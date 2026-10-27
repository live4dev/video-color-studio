"""Build-host plans: actual POSIX forwarding, structural Windows command-file checks."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "windows_native_host", ROOT / "tools/windows_native_host.py"
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    binaries = tmp_path / "host tools"
    binaries.mkdir()
    zig, cmake, ninja = (binaries / name for name in ("zig.exe", "cmake.exe", "ninja.exe"))
    for path in (zig, cmake, ninja):
        path.write_bytes(b"developer tool placeholder; never executed")
    monkeypatch.setattr(
        host.shutil, "which", lambda name: str({"cmake": cmake, "ninja": ninja}[name])
    )
    monkeypatch.setattr(host, "_compile_driver", lambda *_args: {"windows_execution": "NOT RUN"})
    return zig, cmake, ninja


def test_windows_plan_has_pe_drivers_ninja_and_an_unpolluted_cmake_prefix(
    tmp_path, monkeypatch, tools
):
    zig, cmake, ninja = tools
    monkeypatch.setattr(host.platform, "system", lambda: "Windows")
    plan = host.host_plan(tmp_path / "target build", zig)
    assert plan["generator"] == "Ninja"
    assert plan["cmake"] == [str(cmake)]
    assert plan["environment"]["PATH"].startswith(str(ninja.parent) + ";")
    assert plan["target"] == "x86_64-windows-gnu"
    for name, subcommand in (
        ("cc", "cc"),
        ("cxx", "c++"),
        ("ar", "ar"),
        ("ranlib", "ranlib"),
        ("rc", "rc"),
    ):
        assert Path(plan[name]).suffix == ".exe"
        path = Path(plan["command_wrappers"][name])
        assert path.suffix == ".cmd"
        content = path.read_bytes().decode()
        assert content.startswith("@echo off\r\n")
        assert f'"{zig}" "{subcommand}"' in content
        assert "%*\r\nexit /b %errorlevel%\r\n" in content
        assert (
            '"-target" "x86_64-windows-gnu"' in content
            if name in ("cc", "cxx")
            else '"-target"' not in content
        )
    assert plan["windows_wrapper_execution"] == "NOT RUN"
    json.dumps(plan)


def test_windows_wrapper_escapes_percent_and_quotes_shell_metacharacters(tmp_path, monkeypatch):
    monkeypatch.setattr(host.platform, "system", lambda: "Windows")
    tool = tmp_path / "tool&100%!" / "zig.exe"
    tool.parent.mkdir()
    tool.touch()
    monkeypatch.setattr(host, "find_host_tool", lambda name, **_kwargs: tool)
    monkeypatch.setattr(host, "_compile_driver", lambda *_args: {"windows_execution": "NOT RUN"})
    plan = host.host_plan(tmp_path / "work", tool)
    command = Path(plan["command_wrappers"]["cc"]).read_bytes().decode()
    assert '"' + str(tool).replace("%", "%%") + '"' in command
    assert "setlocal DisableDelayedExpansion\r\n" in command


def test_pe_driver_uses_unicode_process_api_and_preserves_unparsed_argument_tail():
    source = host._driver_source(
        [r"C:\tools space&100%!\Дайвинг\zig.exe", "cc", "-target", host.TARGET]
    )
    assert "CreateProcessW(zig_path, command" in source
    assert "GetCommandLineW()" in source
    assert "GetExitCodeProcess" in source
    assert "GetStdHandle(STD_ERROR_HANDLE)" in source
    assert r"\u0414" in source
    assert "cmd.exe" not in source
    assert "100%!" in source


@pytest.mark.skipif(os.name == "nt", reason="This case executes a real POSIX shell wrapper")
def test_posix_wrapper_really_preserves_forwarded_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    zig = tmp_path / "fake compiler"
    zig.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    zig.chmod(0o755)
    monkeypatch.setattr(host, "find_host_tool", lambda name, **_kwargs: Path(sys.executable))
    plan = host.host_plan(tmp_path / "cross build", zig)
    arguments = ["path with spaces.c", "literal;$x", 'a"b', ""]
    observed = json.loads(subprocess.check_output([plan["cc"], *arguments], text=True))
    assert observed == ["cc", "-target", "x86_64-windows-gnu", *arguments]
    assert plan["generator"] == "Unix Makefiles"
    assert plan["environment"] == {}


def test_tool_discovery_fails_clearly_without_searching_darwin_cache_on_windows(monkeypatch):
    monkeypatch.setattr(host.shutil, "which", lambda _name: None)
    monkeypatch.setattr(host, "_package_tool", lambda *_args: None)
    with pytest.raises(RuntimeError, match="Host ninja is required"):
        host.find_host_tool("ninja", system="Windows")


def test_tool_discovery_uses_the_installed_host_package_binary(tmp_path, monkeypatch):
    binary = tmp_path / "ninja.exe"
    binary.touch()
    monkeypatch.setattr(host.shutil, "which", lambda _name: None)
    monkeypatch.setattr(host, "_package_tool", lambda name, windows: binary)
    assert host.find_host_tool("ninja", system="Windows") == binary


def test_missing_pinned_zig_is_rejected_before_any_wrapper_is_written(tmp_path):
    with pytest.raises(FileNotFoundError, match="pinned host Zig"):
        host.host_plan(tmp_path / "work", tmp_path / "missing-zig")
    assert not (tmp_path / "work").exists()
