"""Windows launch decisions and COM argument signatures, mocked on other OSes."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from freedive_color.control import windows_launch
from freedive_color.control.protocol import RemoteControlError

EXE = r"C:\Freedive Дайвинг & 100%!\Freedive Color Studio.exe"
COMMAND = [EXE, "--integration", "--control-dir", r"C:\Users\A B\control"]


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(windows_launch.sys, "platform", "win32")
    monkeypatch.setattr(windows_launch, "_executable_exists", lambda _value: True)


def test_breakaway_requests_independent_job_and_preserves_exact_argv_environment(
    windows, monkeypatch
):
    process = Mock()
    popen = Mock(return_value=process)
    monkeypatch.setattr(windows_launch.subprocess, "Popen", popen)
    broker = Mock()
    monkeypatch.setattr(windows_launch, "_launch_via_explorer", broker)
    environment = {
        "LOCALAPPDATA": r"C:\Users\A B\AppData",
        "FREEDIVE_COLOR_PORTABLE_ROOT": "C:\\FCS",
    }
    assert windows_launch.launch_independent(COMMAND, environment) is process
    args, kwargs = popen.call_args
    assert args == (COMMAND,)
    assert kwargs["env"] is environment
    assert kwargs["creationflags"] == 0x01000208
    assert kwargs["close_fds"] is True
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    broker.assert_not_called()


def test_only_access_denied_uses_existing_explorer_broker(windows, monkeypatch):
    denied = OSError("host job denies breakaway")
    denied.winerror = 5
    monkeypatch.setattr(windows_launch.subprocess, "Popen", Mock(side_effect=denied))
    broker = Mock()
    monkeypatch.setattr(windows_launch, "_launch_via_explorer", broker)
    assert windows_launch.launch_independent(COMMAND, {}) is None
    broker.assert_called_once_with(COMMAND)


def test_unrelated_launch_failure_is_explicit_and_does_not_try_broker(windows, monkeypatch):
    missing = OSError("bad executable")
    missing.winerror = 193
    monkeypatch.setattr(windows_launch.subprocess, "Popen", Mock(side_effect=missing))
    broker = Mock()
    monkeypatch.setattr(windows_launch, "_launch_via_explorer", broker)
    with pytest.raises(RemoteControlError) as raised:
        windows_launch.launch_independent(COMMAND, {})
    assert raised.value.code == "APP_START_FAILED"
    assert "вручную" in raised.value.hint
    broker.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["relative.exe"],
        [r"\\host\share\app.exe"],
        ["https://host/app.exe"],
        ["C:\\app.cmd"],
        [EXE, 1],
        [EXE, "a\0b"],
    ],
)
def test_absolute_local_exe_and_separate_string_arguments_required(windows, command):
    with pytest.raises(RemoteControlError):
        windows_launch._validate_command(command, {})


def test_com_broker_uses_desktop_chain_exact_byref_signature_and_no_command_shell(monkeypatch):
    trace = []
    application = SimpleNamespace(
        ShellExecute=Mock(side_effect=lambda *a: trace.append(("execute", a)))
    )
    folder = SimpleNamespace(Application=application)
    view = SimpleNamespace(GetItemObject=Mock(return_value=folder))
    browser = SimpleNamespace(QueryActiveShellView=Mock(return_value=view))
    provider = SimpleNamespace(QueryService=Mock(return_value=browser))
    desktop = SimpleNamespace(QueryInterface=Mock(return_value=provider))
    windows_object = SimpleNamespace(
        GetIDsOfNames=Mock(return_value=42), InvokeTypes=Mock(return_value=(desktop, 123))
    )
    pythoncom = SimpleNamespace(
        MakeIID=lambda value: value,
        CLSCTX_LOCAL_SERVER=4,
        IID_IDispatch="IDispatch",
        IID_IServiceProvider="IServiceProvider",
        DISPATCH_METHOD=1,
        VT_DISPATCH=9,
        VT_VARIANT=12,
        VT_BYREF=0x4000,
        VT_I4=3,
        PARAMFLAG_FIN=1,
        PARAMFLAG_FOUT=2,
        CoCreateInstance=Mock(return_value=windows_object),
    )
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(sys.modules, "win32com", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32com.client", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "win32com.client.dynamic", SimpleNamespace(Dispatch=lambda o: o)
    )
    monkeypatch.setitem(
        sys.modules,
        "win32com.shell",
        SimpleNamespace(shell=SimpleNamespace(IID_IShellBrowser="IShellBrowser")),
    )
    monkeypatch.setattr(windows_launch, "_desktop_window", lambda: 111)
    monkeypatch.setattr(
        windows_launch, "_assert_same_user", lambda hwnd: trace.append(("sid", hwnd))
    )
    windows_launch._broker_in_apartment(COMMAND)
    assert [item[0] for item in trace] == ["sid", "sid", "execute"]
    call = windows_object.InvokeTypes.call_args.args
    assert call[4] == ((16396, 1), (16396, 1), (3, 1), (16387, 2), (3, 1))
    assert call[5:] == (0, 0, 8, None, 1)
    executable, arguments, directory, verb, show = application.ShellExecute.call_args.args
    assert executable == EXE
    assert arguments == subprocess.list2cmdline(COMMAND[1:])
    assert verb == "open" and show == 1
    assert directory == r"C:\Freedive Дайвинг & 100%!"


def test_missing_desktop_and_wrong_user_fail_before_shell_execute(monkeypatch):
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32com", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32com.client", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "win32com.client.dynamic", SimpleNamespace(Dispatch=lambda o: o)
    )
    monkeypatch.setitem(sys.modules, "win32com.shell", SimpleNamespace(shell=SimpleNamespace()))
    monkeypatch.setattr(windows_launch, "_desktop_window", lambda: 0)
    with pytest.raises(RemoteControlError) as raised:
        windows_launch._broker_in_apartment(COMMAND)
    assert raised.value.details["reason"] == "no_existing_desktop"


def test_desktop_sid_mismatch_closes_all_handles_and_fails_before_automation(monkeypatch):
    api = SimpleNamespace(
        OpenProcess=Mock(return_value="desktop-process"),
        GetCurrentProcess=lambda: "current-process",
        CloseHandle=Mock(),
    )
    security = SimpleNamespace(
        TokenUser=1,
        OpenProcessToken=lambda process, _rights: process + "-token",
        GetTokenInformation=lambda token, _kind: (token, 0),
        ConvertSidToStringSid=lambda sid: sid,
    )
    monkeypatch.setitem(sys.modules, "win32api", api)
    monkeypatch.setitem(sys.modules, "win32security", security)
    monkeypatch.setitem(
        sys.modules,
        "win32process",
        SimpleNamespace(GetWindowThreadProcessId=lambda _hwnd: (7, 123)),
    )
    with pytest.raises(RemoteControlError) as raised:
        windows_launch._assert_same_user(111)
    assert raised.value.details["reason"] == "desktop_user_mismatch"
    assert [call.args[0] for call in api.CloseHandle.call_args_list] == [
        "current-process-token",
        "desktop-process-token",
        "desktop-process",
    ]
