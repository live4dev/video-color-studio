"""Synchronous Qt local-socket client used by the independent stdio adapter."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .protocol import (
    API_VERSION,
    ControlPaths,
    LineDecoder,
    RemoteControlError,
    encode_json,
    private_directory,
    read_connection,
)


def application_command() -> list[str]:
    override = os.environ.get("FREEDIVE_COLOR_APP")
    if override:
        executable = Path(override).expanduser().absolute()
        if executable.suffix == ".app":
            executable = executable / "Contents" / "MacOS" / "Freedive Color Studio"
        if not executable.is_file():
            raise RemoteControlError(
                "APP_NOT_FOUND",
                "Не найдено приложение.",
                "Укажите полный путь к Freedive Color Studio.",
            )
        return [str(executable), "--integration"]
    if getattr(sys, "frozen", False):
        parent = Path(sys.executable).resolve().parent
        names = ("Freedive Color Studio.exe", "Freedive Color Studio", "freedive-color-studio")
        for name in names:
            executable = parent / name
            if executable.is_file() and executable.resolve() != Path(sys.executable).resolve():
                return [str(executable), "--integration"]
        raise RemoteControlError(
            "APP_NOT_FOUND",
            "Не найдено приложение рядом с адаптером.",
            "Распакуйте приложение и MCP-адаптер вместе.",
        )
    return [sys.executable, "-m", "freedive_color", "--integration"]


def mcp_configuration() -> dict:
    portable = os.environ.get("FREEDIVE_COLOR_PORTABLE_ROOT")
    if portable:
        return {"command": str(Path(portable).absolute() / "freedive-color-mcp.exe"), "args": []}
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve().parent / (
            "freedive-color-mcp.exe" if os.name == "nt" else "freedive-color-mcp"
        )
        return {"command": str(executable), "args": []}
    return {
        "command": str(Path(sys.executable).absolute()),
        "args": ["-m", "freedive_color.control.mcp_server"],
    }


class ApplicationClient:
    def __init__(
        self,
        *,
        paths: ControlPaths | None = None,
        timeout: float = 15.0,
        startup_timeout: float = 45.0,
        launch: bool = True,
    ):
        self.paths = paths or ControlPaths.default()
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.launch = launch
        self._mutex = threading.Lock()
        self._counter = 0

    def call(self, method: str, params: dict) -> dict:
        if not isinstance(method, str) or not isinstance(params, dict):
            raise RemoteControlError(
                "INVALID_PARAMS", "Ожидаются имя операции и объект параметров."
            )
        if method == "get_capabilities":
            if params:
                raise RemoteControlError(
                    "INVALID_PARAMS", "get_capabilities не принимает параметры."
                )
            from .schemas import get_capabilities

            return get_capabilities()
        if method == "connect_session" and params:
            raise RemoteControlError("INVALID_PARAMS", "connect_session не принимает параметры.")
        with self._mutex:
            if method == "connect_session" and self.launch:
                return self._connect(params)
            return self._request(method, params)

    def _request(self, method, params, *, timeout=None):
        from PySide6.QtNetwork import QLocalSocket

        try:
            registry, token = read_connection(self.paths)
        except FileNotFoundError as error:
            raise RemoteControlError(
                "APP_NOT_RUNNING", "Приложение ещё не подключено.", "Вызовите connect_session."
            ) from error
        except PermissionError as error:
            raise RemoteControlError(
                "INSECURE_CHANNEL",
                "Недостаточная защита локального подключения.",
                "Проверьте права каталога подключения текущего пользователя.",
            ) from error
        socket = QLocalSocket()
        budget = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + budget

        def remaining():
            return max(1, int((deadline - time.monotonic()) * 1000))

        try:
            socket.connectToServer(registry["endpoint"])
            if not socket.waitForConnected(remaining()):
                raise RemoteControlError(
                    "APP_NOT_RUNNING",
                    "Локальный канал приложения недоступен.",
                    "Вызовите connect_session для переподключения.",
                )
            self._counter += 1
            identifier = self._counter
            socket.write(
                encode_json(
                    {
                        "jsonrpc": "2.0",
                        "id": identifier,
                        "method": method,
                        "params": params,
                        "api_version": API_VERSION,
                        "token": token,
                    }
                )
            )
            while socket.bytesToWrite() and time.monotonic() < deadline:
                if not socket.waitForBytesWritten(min(20, remaining())):
                    break
            parser = LineDecoder()
            while time.monotonic() < deadline:
                if not socket.bytesAvailable() and not socket.waitForReadyRead(
                    min(20, remaining())
                ):
                    if socket.state() == QLocalSocket.LocalSocketState.UnconnectedState:
                        raise RemoteControlError(
                            "CHANNEL_CLOSED",
                            "Приложение закрыло локальный канал.",
                            "Повторите подключение; сохраните request_id при повторе задания.",
                        )
                    time.sleep(0.001)
                    continue
                for response in parser.feed(bytes(socket.readAll())):
                    if response.get("jsonrpc") != "2.0" or response.get("id") != identifier:
                        raise RemoteControlError(
                            "INVALID_RESPONSE", "Неверный ответ локального приложения."
                        )
                    if "error" in response:
                        error = response["error"]
                        if (
                            set(response) != {"jsonrpc", "id", "error"}
                            or not isinstance(error, dict)
                            or not isinstance(error.get("data"), dict)
                            or not isinstance(error.get("message"), str)
                        ):
                            raise RemoteControlError(
                                "INVALID_RESPONSE", "Неверная ошибка локального приложения."
                            )
                        data = error["data"]
                        raise RemoteControlError(
                            data.get("code", "REMOTE_ERROR"),
                            error.get("message", "Ошибка приложения."),
                            data.get("hint", ""),
                            data.get("details", {}),
                        )
                    if set(response) != {"jsonrpc", "id", "result"} or not isinstance(
                        response["result"], dict
                    ):
                        raise RemoteControlError(
                            "INVALID_RESPONSE", "Неверный результат локальной операции."
                        )
                    return response["result"]
            raise RemoteControlError(
                "APP_TIMEOUT",
                "Приложение не ответило вовремя.",
                "Проверьте состояние приложения и повторите запрос.",
            )
        finally:
            socket.abort()

    def _connect(self, params):
        try:
            return self._request("connect_session", params, timeout=min(self.timeout, 2.0))
        except RemoteControlError as error:
            if error.code != "APP_NOT_RUNNING":
                raise
        from PySide6.QtCore import QLockFile

        private_directory(self.paths.directory)
        launch_lock = QLockFile(str(self.paths.launch_lock))
        launch_lock.setStaleLockTime(0)
        lock_deadline = time.monotonic() + self.startup_timeout
        while not launch_lock.tryLock(0):
            if time.monotonic() >= lock_deadline:
                raise RemoteControlError(
                    "APP_STARTING",
                    "Другой адаптер запускает приложение.",
                    "Повторите подключение через несколько секунд.",
                )
            time.sleep(0.025)
        try:
            try:
                return self._request("connect_session", params, timeout=1.0)
            except RemoteControlError as error:
                if error.code != "APP_NOT_RUNNING":
                    raise
            environment = os.environ.copy()
            if not getattr(sys, "frozen", False):
                source_root = str(Path(__file__).resolve().parents[2])
                environment["PYTHONPATH"] = source_root + (
                    os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
                )
            options = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "env": environment,
                "close_fds": True,
            }
            try:
                command = application_command() + ["--control-dir", str(self.paths.directory)]
                if os.name == "nt":
                    from .windows_launch import launch_independent

                    process = launch_independent(command, environment)
                else:
                    options["start_new_session"] = True
                    process = subprocess.Popen(command, **options)
            except OSError as error:
                raise RemoteControlError(
                    "APP_START_FAILED",
                    "Не удалось запустить приложение.",
                    "Проверьте путь и права запуска.",
                ) from error
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                try:
                    return self._request("connect_session", params, timeout=0.5)
                except RemoteControlError as error:
                    if error.code not in ("APP_NOT_RUNNING", "APP_TIMEOUT"):
                        raise
                if process is not None and process.poll() is not None:
                    raise RemoteControlError(
                        "APP_START_FAILED",
                        "Приложение завершилось при запуске.",
                        "Запустите приложение вручную для диагностики.",
                    )
                time.sleep(0.05)
            raise RemoteControlError(
                "APP_START_TIMEOUT",
                "Приложение не запустилось вовремя.",
                "Запустите приложение вручную и подключитесь снова.",
            )
        finally:
            launch_lock.unlock()
