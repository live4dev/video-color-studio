import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PySide6.QtCore import QObject
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtWidgets import QWidget

from freedive_color.control.client import ApplicationClient
from freedive_color.control.protocol import ControlPaths, encode_json
from freedive_color.control.runtime import ApplicationRuntime, InstanceLease


class Controller(QObject):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.idle = True
        self.shutdowns = 0

    def execute(self, method, params):
        self.calls.append((method, params))
        return {"session_id": "same-session", "message": "Готово", "method": method}

    def shutdown(self):
        self.shutdowns += 1

    def is_idle(self):
        return self.idle


class Window(QWidget):
    def request_quit(self):
        self.explicit_quit = True


@pytest.fixture
def runtime(qapp, qtbot, tmp_path):
    controller, window = Controller(), Window()
    qtbot.addWidget(window)
    instance = ApplicationRuntime(
        qapp, controller, window, integration_mode=True, paths=ControlPaths(tmp_path / "control")
    )
    yield instance
    instance._closing = True
    instance._quit_timer.stop()
    instance.close()
    window.close()
    instance.deleteLater()


def connect_socket(runtime, qtbot):
    socket = QLocalSocket()
    socket.connectToServer(runtime.endpoint)
    assert socket.waitForConnected(1000)
    qtbot.waitUntil(lambda: len(runtime._clients) > 0)
    return socket


def request(runtime, identifier=1, **changes):
    return encode_json(
        {
            "jsonrpc": "2.0",
            "id": identifier,
            "api_version": 1,
            "token": runtime.token,
            "method": "get_state",
            "params": {},
            **changes,
        }
    )


def read_line(socket, qtbot):
    qtbot.waitUntil(socket.canReadLine, timeout=3000)
    return json.loads(bytes(socket.readLine()))


def test_fragmented_coalesced_and_unauthorized_do_not_dispatch(runtime, qtbot):
    socket = connect_socket(runtime, qtbot)
    data = request(runtime, params={"name": "водолаз"})
    split = data.index("водолаз".encode()) + 1
    socket.write(data[:split])
    socket.flush()
    qtbot.wait(30)
    assert runtime.controller.calls == []
    socket.write(data[split:] + request(runtime, 2))
    socket.flush()
    assert read_line(socket, qtbot)["id"] == 1
    assert read_line(socket, qtbot)["id"] == 2
    socket.write(request(runtime, 3, token="wrong"))
    socket.flush()
    error = read_line(socket, qtbot)["error"]
    assert error["data"]["code"] == "UNAUTHORIZED"
    assert len(runtime.controller.calls) == 2
    socket.abort()


def test_api_version_mismatch(runtime, qtbot):
    socket = connect_socket(runtime, qtbot)
    socket.write(request(runtime, api_version=2))
    socket.flush()
    assert read_line(socket, qtbot)["error"]["data"]["code"] == "API_VERSION_MISMATCH"
    assert runtime.controller.calls == []
    socket.abort()


def test_background_disconnect_reconnect_same_session(runtime, qtbot):
    runtime.controller.idle = False
    runtime.show_window()
    runtime.window.close()
    assert not runtime.window.isVisible()
    assert runtime.controller.shutdowns == 0
    socket = connect_socket(runtime, qtbot)
    socket.write(request(runtime, method="connect_session"))
    socket.flush()
    assert read_line(socket, qtbot)["result"]["session_id"] == "same-session"
    assert runtime.window.isVisible()
    socket.abort()
    qtbot.waitUntil(lambda: not runtime._clients)
    assert not runtime.controller.is_idle()
    assert runtime.controller.shutdowns == 0


def test_client_in_worker_thread(runtime, qtbot):
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(ApplicationClient(paths=runtime.paths).call, "get_state", {})
        qtbot.waitUntil(result.done, timeout=5000)
        assert result.result()["session_id"] == "same-session"


def test_client_process_requires_no_gui_or_qcoreapp(runtime, qtbot):
    environment = os.environ.copy()
    environment["FREEDIVE_COLOR_CONTROL_DIR"] = str(runtime.paths.directory)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json;from freedive_color.control.client import ApplicationClient;"
            "print(json.dumps(ApplicationClient().call('get_state',{})))",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    qtbot.waitUntil(lambda: process.poll() is not None, timeout=8000)
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert json.loads(stdout)["session_id"] == "same-session"
    assert not stderr


def test_singleton_live_lease_and_cleanup(runtime):
    second = InstanceLease(runtime.paths)
    assert not second.acquire()
    runtime.close()
    assert not runtime.paths.registry.exists()
    assert not runtime.paths.token.exists()
    assert second.acquire()
    second.release()


def test_explicit_quit_cancels_then_waits_without_block(runtime, qtbot, monkeypatch):
    called = []
    monkeypatch.setattr(runtime.app, "quit", lambda: called.append(True))
    runtime.controller.idle = False
    runtime.request_quit()
    assert runtime.controller.shutdowns == 1
    assert runtime.window.explicit_quit
    assert not called
    qtbot.wait(70)
    assert not called
    runtime.controller.idle = True
    qtbot.waitUntil(lambda: bool(called), timeout=3000)
    assert not runtime.paths.registry.exists()


def test_stale_process_lock_is_recovered(qtbot, tmp_path):
    paths = ControlPaths(tmp_path / "stale channel")
    lease = InstanceLease(paths)
    ready = tmp_path / "ready"
    script = (
        "import sys,time;from pathlib import Path;from PySide6.QtCore import QLockFile;"
        "lock=QLockFile(sys.argv[1]);lock.setStaleLockTime(0);"
        "assert lock.tryLock(0);Path(sys.argv[2]).touch();time.sleep(30)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(paths.lock), str(ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        qtbot.waitUntil(lambda: ready.exists() or child.poll() is not None, timeout=5000)
        assert ready.exists(), child.communicate(timeout=2)[1].decode()
        assert not lease.acquire()
        child.terminate()
        child.wait(timeout=3)
        assert paths.lock.exists()
        assert lease.acquire()
        lease.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_oversized_request_does_not_dispatch(runtime, qtbot):
    socket = connect_socket(runtime, qtbot)
    socket.write(b"x" * (1024 * 1024 + 1))
    socket.flush()
    assert read_line(socket, qtbot)["error"]["data"]["code"] == "MESSAGE_TOO_LARGE"
    assert runtime.controller.calls == []
    socket.abort()


def test_oversized_response_returns_pagination_hint(runtime, qtbot, monkeypatch):
    monkeypatch.setattr(runtime.controller, "execute", lambda *_args: {"large": "x" * 1048576})
    socket = connect_socket(runtime, qtbot)
    socket.write(request(runtime))
    socket.flush()
    error = read_line(socket, qtbot)["error"]["data"]
    assert error["code"] == "MESSAGE_TOO_LARGE"
    assert "limit" in error["hint"]
    socket.abort()
