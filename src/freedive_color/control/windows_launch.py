"""Launch the GUI outside the MCP host's kill-on-close Windows job.

First request an ordinary job breakaway. When the host forbids it, use the
existing user's desktop Explorer automation object, following Microsoft's
IShellWindows -> desktop browser/view -> Application.ShellExecute pattern.
No command interpreter, new Explorer process, elevation or global settings.

Native references:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/exdisp/nf-exdisp-ishellwindows-findwindowsw
https://github.com/mhammond/pywin32/blob/main/com/win32com/src/PyIDispatch.cpp

Explorer inherits its own environment. The caller must pass --control-dir in
argv; the portable GUI bootstrap derives its own runtime root from its EXE.
"""

from __future__ import annotations

import ctypes
import ntpath
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from .protocol import RemoteControlError

_BREAKAWAY = 0x01000000
_DETACHED = 0x00000008
_NEW_PROCESS_GROUP = 0x00000200
_CLSID_SHELL_WINDOWS = "{9BA05972-F6A8-11CF-A442-00A0C90A8F39}"
_SID_TOP_LEVEL_BROWSER = "{4C96BE40-915C-11CF-99D3-00AA004AE837}"


def _failure(details=None) -> RemoteControlError:
    return RemoteControlError(
        "APP_START_FAILED",
        "Не удалось независимо запустить окно приложения.",
        "Откройте Freedive Color Studio вручную и повторите connect_session.",
        details or {},
    )


def _executable_exists(value: str) -> bool:
    return Path(value).is_file()


def _validate_command(command: list[str], environment: dict[str, str]) -> list[str]:
    if (
        type(command) is not list
        or not command
        or any(type(arg) is not str or "\0" in arg for arg in command)
    ):
        raise _failure({"reason": "invalid_arguments"})
    executable = command[0]
    if (
        not PureWindowsPath(executable).is_absolute()
        or executable.startswith(("\\\\", "//"))
        or PureWindowsPath(executable).suffix.lower() != ".exe"
        or not _executable_exists(executable)
    ):
        raise _failure({"reason": "absolute_local_exe_required"})
    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str or "\0" in key or "\0" in value
        for key, value in environment.items()
    ):
        raise _failure({"reason": "invalid_environment"})
    return list(command)


def _desktop_window() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetShellWindow.argtypes = []
    user32.GetShellWindow.restype = ctypes.c_void_p
    return user32.GetShellWindow() or 0


def _assert_same_user(hwnd: int) -> None:
    import win32api
    import win32process
    import win32security

    handles = []
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if not pid:
            raise _failure({"reason": "no_desktop_process"})
        process = win32api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        handles.append(process)
        desktop_token = win32security.OpenProcessToken(process, 0x0008)  # TOKEN_QUERY
        handles.append(desktop_token)
        current_token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), 0x0008)
        handles.append(current_token)
        desktop_sid = win32security.GetTokenInformation(desktop_token, win32security.TokenUser)[0]
        current_sid = win32security.GetTokenInformation(current_token, win32security.TokenUser)[0]
        if win32security.ConvertSidToStringSid(desktop_sid) != win32security.ConvertSidToStringSid(
            current_sid
        ):
            raise _failure({"reason": "desktop_user_mismatch"})
    finally:
        for handle in reversed(handles):
            win32api.CloseHandle(handle)


def _broker_in_apartment(command: list[str]) -> None:
    import pythoncom
    from win32com.client.dynamic import Dispatch
    from win32com.shell import shell

    hwnd = _desktop_window()
    if not hwnd:
        raise _failure({"reason": "no_existing_desktop"})
    _assert_same_user(hwnd)
    windows = pythoncom.CoCreateInstance(
        pythoncom.MakeIID(_CLSID_SHELL_WINDOWS),
        None,
        pythoncom.CLSCTX_LOCAL_SERVER,
        pythoncom.IID_IDispatch,
    )
    # InvokeTypes matches ExDisp.Idl exactly. VARIANT* inputs and LONG* output
    # require VT_BYREF; the out-only HWND position still occupies one argument.
    desktop, found_hwnd = windows.InvokeTypes(
        windows.GetIDsOfNames("FindWindowSW"),
        0,
        pythoncom.DISPATCH_METHOD,
        (pythoncom.VT_DISPATCH, 0),
        (
            (pythoncom.VT_VARIANT | pythoncom.VT_BYREF, pythoncom.PARAMFLAG_FIN),
            (pythoncom.VT_VARIANT | pythoncom.VT_BYREF, pythoncom.PARAMFLAG_FIN),
            (pythoncom.VT_I4, pythoncom.PARAMFLAG_FIN),
            (pythoncom.VT_I4 | pythoncom.VT_BYREF, pythoncom.PARAMFLAG_FOUT),
            (pythoncom.VT_I4, pythoncom.PARAMFLAG_FIN),
        ),
        0,
        0,
        8,  # SWC_DESKTOP
        None,
        1,  # SWFO_NEEDDISPATCH
    )
    if desktop is None or not found_hwnd:
        raise _failure({"reason": "desktop_automation_unavailable"})
    _assert_same_user(found_hwnd)
    provider = desktop.QueryInterface(pythoncom.IID_IServiceProvider)
    browser = provider.QueryService(
        pythoncom.MakeIID(_SID_TOP_LEVEL_BROWSER), shell.IID_IShellBrowser
    )
    view = browser.QueryActiveShellView()
    folder_view = Dispatch(view.GetItemObject(0, pythoncom.IID_IDispatch))  # SVGIO_BACKGROUND
    application = folder_view.Application
    application.ShellExecute(
        command[0], subprocess.list2cmdline(command[1:]), ntpath.dirname(command[0]), "open", 1
    )


def _launch_via_explorer(command: list[str]) -> None:
    import pythoncom

    initialized = False
    try:
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        initialized = True
        _broker_in_apartment(command)
    finally:
        if initialized:
            pythoncom.CoUninitialize()


def launch_independent(command: list[str], environment: dict[str, str]) -> subprocess.Popen | None:
    """Return a direct process, or None when Explorer accepted the launch.

    Readiness is always established by the authenticated application registry,
    never by successful Popen/ShellExecute alone. Windows execution is verified
    by the native acceptance runner; tests on other hosts mock Win32 calls.
    """
    if sys.platform != "win32":
        raise _failure({"reason": "windows_only"})
    command = _validate_command(command, environment)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            close_fds=True,
            cwd=ntpath.dirname(command[0]),
            creationflags=_BREAKAWAY | _DETACHED | _NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) != 5:
            raise _failure({"reason": "direct_launch_failed"}) from exc
    try:
        _launch_via_explorer(command)
    except Exception as exc:
        if isinstance(exc, RemoteControlError):
            raise
        raise _failure({"reason": "explorer_broker_unavailable"}) from exc
    return None
