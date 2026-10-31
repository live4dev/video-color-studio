import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_onnxruntime_macos as builder


def wheel(tmp_path: Path, *, native=True, unsafe=False) -> Path:
    path = tmp_path / "onnxruntime-1.23.2-cp312-cp312-macosx_13_0_arm64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "onnxruntime-1.23.2.dist-info/METADATA", "Name: onnxruntime\nVersion: 1.23.2\n"
        )
        if native:
            archive.writestr("onnxruntime/capi/runtime.so", bytes.fromhex("cffaedfe") + b"fixture")
        if unsafe:
            archive.writestr("../outside.txt", "unsafe")
    return path


@pytest.mark.parametrize("minimum", ["13.0", "13.0.0", "12.0"])
def test_native_wheel_is_audited_before_isolated_cpu_inference(tmp_path, monkeypatch, minimum):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "otool" in command[0]:
            output = f"cmd LC_BUILD_VERSION\n    minos {minimum}\n    sdk 15.5\n"
        elif "lipo" in command[0]:
            output = "arm64\n"
        else:
            assert command[1:3] == ["-I", "-c"]
            assert "disable_telemetry_events" in command[3]
            assert "CPUExecutionProvider" in command[3]
            output = json.dumps({"identity_inference": True, "telemetry": False})
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", run)
    report = builder.verify_wheel(wheel(tmp_path), tmp_path / "unpacked", "arm64")
    assert report["target"] == "13.0"
    assert report["binaries"][0]["minimum_macos"] == [minimum]
    assert report["runtime"]["identity_inference"]
    assert len(calls) == 3


@pytest.mark.parametrize(
    "commands",
    [
        "cmd LC_BUILD_VERSION\n minos 13.4\n sdk 15.5\n",
        "cmd LC_BUILD_VERSION\n minos 13.0\ncmd LC_BUILD_VERSION\n minos 14.0\n",
        "cmd LC_VERSION_MIN_MACOSX\n version 13.4\n sdk 15.5\n",
        "cmd LC_BUILD_VERSION\n sdk 15.5\n",
    ],
)
def test_too_new_or_undeclared_macho_target_never_reaches_install(tmp_path, monkeypatch, commands):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=commands)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="newer than 13.0"):
        builder.verify_wheel(wheel(tmp_path), tmp_path / "unpacked", "arm64")
    assert len(calls) == 1 and "otool" in calls[0][0]


def test_wheel_requires_native_payload_and_safe_paths(tmp_path):
    with pytest.raises(RuntimeError, match="no Mach-O"):
        builder.verify_wheel(wheel(tmp_path, native=False), tmp_path / "unpacked", "arm64")
    with pytest.raises(RuntimeError, match="Unsafe wheel"):
        builder.verify_wheel(wheel(tmp_path, unsafe=True), tmp_path / "unpacked", "arm64")
    assert not (tmp_path / "outside.txt").exists()


def test_source_build_compiles_target_without_binary_header_rewriting(tmp_path):
    command = builder.build_command(tmp_path / "source", tmp_path / "work", 3, "arm64")
    assert "CMAKE_OSX_DEPLOYMENT_TARGET=13.0" in command
    assert "CMAKE_OSX_ARCHITECTURES=arm64" in command
    assert "onnxruntime_USE_TELEMETRY=OFF" in command
    assert "onnxruntime_USE_COREML=OFF" in command
    assert "--no_kleidiai" in command
    assert not any("vtool" in item or "install_name_tool" in item for item in command)
    assert builder.SOURCE_COMMIT == "a83fc4d58cb48eb68890dd689f94f28288cf2278"


def test_source_commit_mismatch_stops_before_compilation(tmp_path, monkeypatch):
    def run(command, **_kwargs):
        return SimpleNamespace(stdout="f" * 40)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="commit does not match"):
        builder.prepare_source(tmp_path)


def test_native_build_failure_includes_only_a_bounded_log_tail(tmp_path):
    command = [
        builder.sys.executable,
        "-c",
        "import sys; print('x'*30000); print('compiler-failure-sentinel'); sys.exit(2)",
    ]
    log = tmp_path / "build.log"
    with pytest.raises(RuntimeError, match="compiler-failure-sentinel") as failure:
        builder.run_logged(command, log, cwd=tmp_path, env=builder.os.environ.copy())
    assert len(str(failure.value)) < 17000
    assert log.stat().st_size > 30000
