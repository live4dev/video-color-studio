import os
import sys
from pathlib import Path

import pytest

from freedive_color.control.client import ApplicationClient, application_command, mcp_configuration
from freedive_color.control.protocol import ControlPaths, RemoteControlError


def test_no_command_launch_for_queries_or_invalid_connect(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A query must not launch the desktop")

    monkeypatch.setattr("freedive_color.control.client.subprocess.Popen", forbidden)
    client = ApplicationClient(paths=ControlPaths(tmp_path / "absent"))
    with pytest.raises(RemoteControlError) as error:
        client.call("get_state", {"session_id": "s"})
    assert error.value.code == "APP_NOT_RUNNING"
    assert not client.paths.directory.exists()


def test_source_configuration_keeps_venv_python_not_resolved_symlink(monkeypatch, tmp_path):
    interpreter = tmp_path / "виртуальное окружение" / "python"
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.delenv("FREEDIVE_COLOR_APP", raising=False)
    config = mcp_configuration()
    assert config["command"] == str(interpreter)
    assert config["args"] == ["-m", "freedive_color.control.mcp_server"]
    assert application_command() == [str(interpreter), "-m", "freedive_color", "--integration"]


def test_frozen_discovery_and_unicode_space_paths(monkeypatch, tmp_path):
    parent = tmp_path / "Приложения с пробелами"
    parent.mkdir()
    suffix = ".exe" if os.name == "nt" else ""
    adapter = parent / f"freedive-color-mcp{suffix}"
    desktop = parent / f"Freedive Color Studio{suffix}"
    adapter.touch()
    desktop.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(adapter))
    monkeypatch.delenv("FREEDIVE_COLOR_APP", raising=False)
    assert application_command() == [str(desktop), "--integration"]
    assert Path(mcp_configuration()["command"]) == adapter
