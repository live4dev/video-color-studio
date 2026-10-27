"""Versioned local protocol and current-user connection files; no Qt or media imports."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024


class RemoteControlError(Exception):
    def __init__(self, code: str, message: str, hint: str = "", details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = {} if details is None else details


class ProtocolError(RemoteControlError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def decode_json(data: bytes) -> dict:
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            "MESSAGE_TOO_LARGE", "Сообщение превышает 1 МиБ.", "Уменьшите размер запроса."
        )
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        if not isinstance(value, dict):
            raise ValueError("object required")
        json.dumps(value, ensure_ascii=False).encode("utf-8", errors="strict")
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("INVALID_REQUEST", "Ожидается объект JSON в UTF-8.") from exc
    return value


def encode_json(value: dict) -> bytes:
    try:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError(
            "INVALID_RESPONSE", "Ответ приложения не удалось сериализовать."
        ) from exc
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            "MESSAGE_TOO_LARGE",
            "Ответ превышает 1 МиБ.",
            "Используйте постраничный запрос с меньшим limit.",
        )
    return data + b"\n"


class LineDecoder:
    """Bound each line, including an unfinished one, without rejecting coalesced lines."""

    def __init__(self):
        self.pending = bytearray()

    def feed(self, data: bytes) -> list[dict]:
        messages = []
        # Do not hold an arbitrarily large, newline-free chunk before checking it.
        for part in data.splitlines(keepends=True):
            self.pending.extend(part)
            ended = self.pending.endswith(b"\n")
            size = len(self.pending) - int(ended)
            if size > MAX_MESSAGE_BYTES:
                self.pending.clear()
                raise ProtocolError("MESSAGE_TOO_LARGE", "Сообщение превышает 1 МиБ.")
            if ended:
                raw = bytes(self.pending[:-1])
                self.pending.clear()
                messages.append(decode_json(raw))
        return messages


def validate_request(value: dict, token: str) -> tuple[str | int, str, dict]:
    allowed = {"jsonrpc", "id", "method", "params", "api_version", "token"}
    if set(value) != allowed or value.get("jsonrpc") != "2.0":
        raise ProtocolError("INVALID_REQUEST", "Неверная структура локального запроса.")
    identifier = value["id"]
    if type(identifier) not in (str, int) or (
        isinstance(identifier, str) and len(identifier) > 128
    ):
        raise ProtocolError("INVALID_REQUEST", "Неверный идентификатор запроса.")
    supplied = value["token"]
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied.encode("utf-8"), token.encode("utf-8")
    ):
        raise ProtocolError(
            "UNAUTHORIZED",
            "Неверный токен локального подключения.",
            "Подключитесь к текущему экземпляру приложения заново.",
        )
    if type(value["api_version"]) is not int or value["api_version"] != API_VERSION:
        raise ProtocolError(
            "API_VERSION_MISMATCH",
            "Несовместимая версия локального API.",
            "Обновите приложение и MCP-адаптер вместе.",
            {"supported": API_VERSION},
        )
    method, params = value["method"], value["params"]
    if (
        not isinstance(method, str)
        or not method
        or len(method) > 128
        or not isinstance(params, dict)
    ):
        raise ProtocolError("INVALID_REQUEST", "Ожидаются имя операции и объект params.")
    return identifier, method, params


def error_response(identifier, error: Exception) -> dict:
    code = getattr(error, "code", "INTERNAL_ERROR")
    message = getattr(error, "message", "Внутренняя ошибка локального управления.")
    number = {
        "INVALID_REQUEST": -32600,
        "METHOD_NOT_FOUND": -32601,
        "INVALID_PARAMS": -32602,
        "INTERNAL_ERROR": -32603,
    }.get(code, -32000)
    return {
        "jsonrpc": "2.0",
        "id": identifier,
        "error": {
            "code": number,
            "message": message,
            "data": {
                "code": code,
                "hint": getattr(error, "hint", ""),
                "details": getattr(error, "details", {}),
            },
        },
    }


def _windows_sid():
    from ctypes import wintypes

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    adv.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    adv.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    handle = wintypes.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(handle)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        count = wintypes.DWORD()
        adv.GetTokenInformation(handle, 1, None, 0, ctypes.byref(count))
        buffer = ctypes.create_string_buffer(count.value)
        if not adv.GetTokenInformation(handle, 1, buffer, count, ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        string = wintypes.LPWSTR()
        if not adv.ConvertSidToStringSidW(pointer, ctypes.byref(string)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return string.value
        finally:
            kernel.LocalFree(ctypes.cast(string, ctypes.c_void_p))
    finally:
        kernel.CloseHandle(handle)


def _windows_acl(path: Path, *, apply: bool, directory: bool = False) -> None:
    """Protected DACL: only the current token's user SID may have allow ACEs."""
    from ctypes import wintypes

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    sid = _windows_sid()
    if apply:
        adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        adv.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        descriptor = ctypes.c_void_p()
        inheritance = "OICI" if directory else ""
        if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:P(A;{inheritance};FA;;;{sid})", 1, ctypes.byref(descriptor), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not adv.SetFileSecurityW(str(path), 0x80000004, descriptor):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.LocalFree(descriptor)
    # Independently read the ACL: never trust chmod() on Windows.
    adv.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    size = wintypes.DWORD()
    adv.GetFileSecurityW(str(path), 4, None, 0, ctypes.byref(size))
    descriptor = ctypes.create_string_buffer(size.value)
    if not adv.GetFileSecurityW(str(path), 4, descriptor, size, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    present, defaulted, acl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
    adv.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    if (
        not adv.GetSecurityDescriptorDacl(
            descriptor, ctypes.byref(present), ctypes.byref(acl), ctypes.byref(defaulted)
        )
        or not present
        or not acl.value
    ):
        raise PermissionError("A private DACL is required")
    ace_count = ctypes.c_ushort.from_address(acl.value + 4).value
    adv.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    allowed = 0
    for index in range(ace_count):
        ace = ctypes.c_void_p()
        if not adv.GetAce(acl, index, ctypes.byref(ace)):
            raise ctypes.WinError(ctypes.get_last_error())
        kind = ctypes.c_ubyte.from_address(ace.value).value
        if kind == 1:  # A deny ACE grants no extra access.
            continue
        if kind != 0:
            raise PermissionError("Unexpected DACL entry")
        text = wintypes.LPWSTR()
        if not adv.ConvertSidToStringSidW(ace.value + 8, ctypes.byref(text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if text.value != sid:
                raise PermissionError("Connection files must be restricted to the current user")
            allowed += 1
        finally:
            kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    if not allowed:
        raise PermissionError("Current-user access is required")


def private_directory(path: Path, *, create: bool = True) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode) or path.is_symlink():
        raise PermissionError("Connection directory must be a real directory")
    if os.name == "nt":
        _windows_acl(path, apply=create, directory=True)
    else:
        if status.st_uid != os.getuid():
            raise PermissionError("Connection directory belongs to another user")
        if create:
            path.chmod(0o700)
        elif stat.S_IMODE(status.st_mode) & 0o077:
            raise PermissionError("Connection directory must have permissions 0700")


def private_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise PermissionError("Connection file must be a regular, unlinked file")
        if os.name == "nt":
            if path.is_symlink():
                raise PermissionError("Connection file cannot be a symlink")
            _windows_acl(path, apply=False)
        elif status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) & 0o077:
            raise PermissionError(
                "Connection file must be owned by current user with permissions 0600"
            )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(MAX_MESSAGE_BYTES + 1)
        if len(data) > MAX_MESSAGE_BYTES:
            raise PermissionError("Connection file is too large")
        return data
    finally:
        os.close(fd)


def write_private(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise PermissionError("Refusing to replace a symlink")
    temporary = path.with_name(path.name + "." + secrets.token_hex(8))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            _windows_acl(temporary, apply=True)
        else:
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ControlPaths:
    directory: Path

    @classmethod
    def default(cls) -> ControlPaths:
        override = os.environ.get("FREEDIVE_COLOR_CONTROL_DIR")
        if override:
            return cls(Path(override).expanduser().absolute())
        if sys.platform == "win32":
            root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        elif sys.platform == "darwin":
            root = Path.home() / "Library" / "Application Support"
        else:
            root = Path(os.environ.get("XDG_RUNTIME_DIR", Path.home() / ".local" / "share"))
        return cls(root / "Freedive Color Studio" / "control-v1")

    @property
    def registry(self):
        return self.directory / "connection.json"

    @property
    def token(self):
        return self.directory / "token"

    @property
    def lock(self):
        return self.directory / "application.lock"

    @property
    def launch_lock(self):
        return self.directory / "launch.lock"

    def endpoint(self) -> str:
        digest = hashlib.sha256(str(self.directory).encode("utf-8")).hexdigest()[:20]
        if os.name == "nt":
            return "freedive-color-api1-" + digest
        candidate = self.directory / "rpc"
        if len(os.fsencode(candidate)) > 95:
            parent = Path(tempfile.gettempdir()) / ("fcs-" + digest)
            private_directory(parent)
            candidate = parent / "rpc"
        return str(candidate)


def read_connection(paths: ControlPaths) -> tuple[dict, str]:
    private_directory(paths.directory, create=False)
    registry = decode_json(private_file(paths.registry))
    if set(registry) != {"api_version", "endpoint", "pid", "instance_id"}:
        raise ProtocolError("INVALID_REGISTRY", "Неверный локальный реестр приложения.")
    if type(registry["api_version"]) is not int or registry["api_version"] != API_VERSION:
        raise ProtocolError(
            "API_VERSION_MISMATCH",
            "Версия запущенного приложения несовместима.",
            "Закройте старую версию и обновите приложение вместе с адаптером.",
        )
    if (
        not isinstance(registry["endpoint"], str)
        or not registry["endpoint"]
        or type(registry["pid"]) is not int
        or registry["pid"] <= 0
        or not isinstance(registry["instance_id"], str)
    ):
        raise ProtocolError("INVALID_REGISTRY", "Неверные данные локального подключения.")
    try:
        token = private_file(paths.token).decode("ascii").strip()
    except UnicodeError as error:
        raise ProtocolError("INVALID_REGISTRY", "Повреждён служебный токен приложения.") from error
    if len(token) < 32 or len(token) > 128:
        raise ProtocolError("INVALID_REGISTRY", "Неверный служебный токен приложения.")
    return registry, token
