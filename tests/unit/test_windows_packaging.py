"""Developer packaging checks require no Windows execution or network access."""

import importlib
import json
import struct
import subprocess
import zipfile
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def win(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("tools.build_windows_portable")


def small_pe(*, imported="kernel32.dll", subsystem=2, function_name="GetModuleFileNameW"):
    data = bytearray(0x1000)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", data, 0x84, 0x8664, 1)
    struct.pack_into("<H", data, 0x94, 240)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x140000000)
    struct.pack_into("<I", data, optional + 60, 0x200)
    struct.pack_into("<H", data, optional + 68, subsystem)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<4I", data, optional + 240 + 8, 0xE00, 0x1000, 0xE00, 0x200)
    struct.pack_into("<II", data, optional + 112 + 8, 0x1100, 40)
    struct.pack_into("<5I", data, 0x300, 0x1200, 0, 0, 0x1300, 0x1200)
    struct.pack_into("<QQ", data, 0x400, 0x1400, 0)
    data[0x500 : 0x500 + len(imported) + 1] = imported.encode() + b"\0"
    symbol = b"\0\0" + function_name.encode("ascii") + b"\0"
    data[0x600 : 0x600 + len(symbol)] = symbol
    return bytes(data)


def opencv_pe():
    data = bytearray(small_pe())
    struct.pack_into("<II", data, 0x98 + 112, 0x1500, 0x100)
    struct.pack_into("<5I", data, 0x700 + 20, 1, 1, 0x1560, 0x1570, 0x1580)
    struct.pack_into("<I", data, 0x760, 0x1020)
    struct.pack_into("<I", data, 0x770, 0x1590)
    struct.pack_into("<H", data, 0x780, 0)
    data[0x790 : 0x790 + 11] = b"PyInit_cv2\0"
    return bytes(data)


def test_pe_machine_subsystem_import_names_read_without_execution(win):
    pe = win.PEImage(small_pe())
    assert pe.machine == 0x8664 and pe.subsystem == 2
    assert pe.imports() == {"kernel32.dll": ["GetModuleFileNameW"]}
    assert pe.exports() == {}
    assert pe.imports(delayed=True) == {}
    with pytest.raises(ValueError, match="Windows PE"):
        win.PEImage(b"not executable")


@pytest.mark.parametrize("name", ["../original.jpg", "/absolute", "C:/windows", "folder\\file"])
def test_archive_path_escape_rejected(win, name):
    with pytest.raises(ValueError, match="Unsafe archive"):
        win.archive_path(name)


def test_zip_extraction_never_replaces_original_media(win, tmp_path):
    original = tmp_path / "original.jpg"
    original.write_bytes(b"original media")
    archive = tmp_path / "bad.whl"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../original.jpg", b"replacement")
    with pytest.raises(ValueError):
        win.extract_zip(archive, tmp_path / "target", wheel=True)
    assert original.read_bytes() == b"original media"


def test_cached_hash_mismatch_and_offline_missing_input_fail_before_network(win, tmp_path):
    entry = {"url": "https://files.pythonhosted.org/packages/package.whl", "sha256": "0" * 64}
    with pytest.raises(FileNotFoundError, match="not cached"):
        win.acquire(entry, tmp_path, offline=True)
    (tmp_path / "package.whl").write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="SHA-256"):
        win.acquire(entry, tmp_path, offline=True)
    with pytest.raises(ValueError, match="approved HTTPS"):
        win.acquire({**entry, "url": "http://example.invalid/file.whl"}, tmp_path)


def test_qt_and_opencv_video_plugins_not_used_by_app_are_omitted(win, tmp_path):
    paths = {
        "runtime/Lib/site-packages/PySide6/QtCore.pyd": small_pe(imported="Qt6Core.dll"),
        "runtime/Lib/site-packages/PySide6/Qt6Core.dll": small_pe(),
        "runtime/Lib/site-packages/PySide6/plugins/designer/unneeded.dll": small_pe(
            imported="absent.dll"
        ),
        "runtime/Lib/site-packages/cv2/cv2.pyd": small_pe(),
        "runtime/Lib/site-packages/cv2/opencv_videoio_ffmpeg4140_64.dll": small_pe(
            imported="legacy_av.dll"
        ),
    }
    for relative, contents in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    report = win.native_audit(tmp_path, trim_qt=True)
    assert len(report["native_files"]) == 3
    assert len(report["omitted_unused_binaries"]) == 2
    assert (tmp_path / "runtime/Lib/site-packages/cv2/cv2.pyd").is_file()
    assert not list(tmp_path.rglob("opencv_videoio_ffmpeg*.dll"))
    assert win.native_audit(tmp_path)["system_imports"] == ["kernel32.dll"]


def test_unknown_required_dll_and_unexpected_cv_plugin_import_fail_closed(win, tmp_path):
    path = tmp_path / "program.exe"
    path.write_bytes(small_pe(imported="missing-runtime.dll"))
    with pytest.raises(ValueError, match="Unbundled native"):
        win.native_audit(tmp_path)
    path.write_bytes(small_pe(imported="opencv_videoio_ffmpeg4140_64.dll"))
    (tmp_path / "opencv_videoio_ffmpeg4140_64.dll").write_bytes(small_pe())
    with pytest.raises(ValueError, match="unused video plugin"):
        win.native_audit(tmp_path, trim_qt=True)


def test_ort_dbghelp_import_is_a_reported_windows_os_dependency(win, tmp_path):
    # The locked ORT wheel imports SymFromAddr/ImageDirectoryEntryToData from the
    # Windows-provided DbgHelp 5.1+ API; no SDK/debugger DLL is bundled or downloaded.
    path = tmp_path / "onnxruntime_pybind11_state.pyd"
    path.write_bytes(small_pe(imported="dbghelp.dll", function_name="SymFromAddr"))
    report = win.native_audit(tmp_path)
    assert report["system_imports"] == ["dbghelp.dll"]
    assert len(report["native_files"]) == 1
    assert not (tmp_path / "dbghelp.dll").exists()
    path.write_bytes(small_pe(imported="missing-vendor-runtime.dll"))
    with pytest.raises(ValueError, match="Unbundled native dependency missing-vendor-runtime.dll"):
        win.native_audit(tmp_path)


def test_real_project_resolves_windows_profile_and_stable_abi_wheels(win):
    wheels = win.locked_wheels()
    assert len(wheels) == len({entry["package"] for entry in wheels})
    by_name = {entry["package"]: entry for entry in wheels}
    assert {"mcp", "mcp-types", "pywin32", "cryptography", "cffi"} <= by_name.keys()
    assert by_name["av"]["version"] == "16.1.0"
    assert by_name["onnxruntime"]["version"] == "1.23.2"
    assert "cp313-cp313-win_amd64" in by_name["onnxruntime"]["filename"]
    assert "cp313-cp313-win_amd64" in by_name["av"]["filename"]
    assert "cp39-abi3-win_amd64" in by_name["pyside6-essentials"]["filename"]
    assert by_name["tifffile"]["filename"].endswith("py3-none-any.whl")
    assert all(len(entry["sha256"]) == 64 for entry in wheels)


def test_windows_build_failure_keeps_bounded_unicode_compiler_diagnostic(win, tmp_path):
    opencv = importlib.import_module("tools.build_windows_opencv")
    log = tmp_path / "build.log"
    log.write_bytes(
        b"old configuration output\n" * 2000
        + "\n".join(f"compile {index}" for index in range(80)).encode("utf-8")
        + "\nclang: error: Не найден заголовок\n".encode()
    )
    tail = opencv.build_log_tail(log)
    assert len(tail.splitlines()) == 60
    assert tail.endswith("clang: error: Не найден заголовок")
    assert "old configuration output" not in tail
    assert len(tail.encode("utf-8")) <= 16384


def test_windows_opencv_failed_command_surfaces_compiler_error(win, tmp_path, monkeypatch):
    import sys

    opencv = importlib.import_module("tools.build_windows_opencv")
    work = tmp_path / "work"
    (work / "source/opencv_python_headless-4.14.0.94/opencv").mkdir(parents=True)
    archive = tmp_path / "build/opencv-image-only/source/opencv_python_headless-4.14.0.94.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.touch()
    failure = tmp_path / "failing-cmake.py"
    failure.write_text(
        "import sys\n"
        "sys.stdout.buffer.write('clang: error: native header unavailable\\n'.encode('utf-8'))\n"
        "raise SystemExit(23)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(opencv, "ROOT", tmp_path)
    monkeypatch.setattr(opencv, "digest", lambda path: opencv.SOURCE_SHA)
    monkeypatch.setattr(opencv, "compiler", lambda *args, **kwargs: tmp_path / "zig")
    monkeypatch.setattr(opencv, "locked_wheels", lambda: [{"package": "numpy"}])
    monkeypatch.setattr(opencv, "acquire", lambda *args, **kwargs: tmp_path / "numpy.whl")
    monkeypatch.setattr(
        opencv, "prepare_sdk", lambda *args, **kwargs: (tmp_path / "python", tmp_path / "numpy")
    )
    host = dict.fromkeys(("cc", "cxx", "ar", "ranlib", "rc"), "unused-tool")
    host.update(cmake=[sys.executable, str(failure)], generator="Ninja", environment={})
    monkeypatch.setattr(opencv, "host_plan", lambda *args: host)
    with pytest.raises(RuntimeError, match="clang: error: native header unavailable"):
        opencv.build(work, tmp_path / "cache", None, 1, True)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["returncode"] == 23


@pytest.fixture
def cmake_tool():
    host = importlib.import_module("tools.windows_native_host")
    try:
        return host.find_host_tool("cmake")
    except RuntimeError:
        pytest.skip("This regression executes the developer CMake parser; install cmake to run it")


@pytest.mark.parametrize(
    "original,expected",
    [
        (r"D:\a\build\zig-rc.exe", "D:/a/build/zig-rc.exe"),
        (
            PureWindowsPath(r"C:\Users\Test User\Дайвинг\zig-cxx.exe"),
            "C:/Users/Test User/Дайвинг/zig-cxx.exe",
        ),
        (r"\\server\share\build folder\zig-rc.exe", "//server/share/build folder/zig-rc.exe"),
        (Path("/project with spaces/Дайвинг/zig-rc"), "/project with spaces/Дайвинг/zig-rc"),
    ],
)
def test_cmake_path_preserves_drive_unc_spaces_and_unicode(original, expected):
    opencv = importlib.import_module("tools.build_windows_opencv")
    assert opencv.cmake_path(original) == expected


def test_raw_windows_rc_path_reproduces_the_actual_cmake_parse_failure(cmake_tool, tmp_path):
    script = tmp_path / "broken-compiler.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.10)\nset(CMAKE_RC_COMPILER "'
        + r"D:\a\video-color-studio\build\wrappers\zig-rc.exe"
        + '")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(cmake_tool), "-P", str(script)], capture_output=True, encoding="utf-8"
    )
    assert result.returncode != 0
    assert "Invalid character escape" in result.stderr


@pytest.mark.parametrize(
    "original,expected",
    [
        (
            r"D:\a\video-color-studio\build\wrappers\zig-rc.exe",
            "D:/a/video-color-studio/build/wrappers/zig-rc.exe",
        ),
        (
            PureWindowsPath(r"C:\Users\Test User\Дайвинг\zig-cxx.exe"),
            "C:/Users/Test User/Дайвинг/zig-cxx.exe",
        ),
        (r"\\server\share\build folder\zig-rc.exe", "//server/share/build folder/zig-rc.exe"),
    ],
)
def test_normalized_compiler_path_roundtrips_through_real_cmake(
    cmake_tool, tmp_path, original, expected
):
    opencv = importlib.import_module("tools.build_windows_opencv")
    observed = tmp_path / "parsed-path.txt"
    script = tmp_path / "compiler.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.10)\nset(CMAKE_RC_COMPILER "'
        + opencv.cmake_path(original)
        + '")\nfile(WRITE "'
        + observed.as_posix()
        + '" "${CMAKE_RC_COMPILER}")\n',
        encoding="utf-8",
    )
    subprocess.run([str(cmake_tool), "-P", str(script)], check=True, capture_output=True)
    assert observed.read_text(encoding="utf-8") == expected


def test_final_zip_requires_positive_matching_windows_source_scope(win, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "payload.txt").write_text("staged source", encoding="utf-8")
    manifest = {
        "kind": "freedive_windows_portable",
        "files": win.file_hashes(stage),
        "payload_sha256": "a" * 64,
        "opencv_native": {"status": "built"},
    }
    (stage / win.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    marker = tmp_path / "source.json"
    marker.write_text(
        json.dumps({"complete": True, "scope": "Windows not asserted"}), encoding="utf-8"
    )
    output = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="explicitly cover Windows"):
        win.finalize(stage, output, marker)
    marker.write_text(
        json.dumps(
            {"complete": True, "platforms": ["windows-x64"], "windows_payload_sha256": "b" * 64}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        win.finalize(stage, output, marker)
    assert not output.exists()
    (stage / "payload.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Staged application changed"):
        win.finalize(stage, output, marker)


def test_native_opencv_replaces_entire_stock_package_and_verifies_export(win, tmp_path):
    package = tmp_path / "compiled/cv2"
    package.mkdir(parents=True)
    (package / "cv2.pyd").write_bytes(opencv_pe())
    (package / "__init__.py").write_text("from .cv2 import *\n", encoding="utf-8")
    manifest = {
        "status": "built",
        "platform": "windows-x64",
        "with_ipp": False,
        "with_ffmpeg": False,
        "package_directory": str(package),
        "files": {p.name: win.sha256(p) for p in package.iterdir()},
    }
    manifest_path = tmp_path / "native.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stage = tmp_path / "stage"
    stock = stage / "runtime/Lib/site-packages/cv2"
    stock.mkdir(parents=True)
    (stock / "old-ipp.pyd").write_bytes(b"old binary must not survive")
    (stock / "opencv_videoio_ffmpeg4140_64.dll").write_bytes(b"unneeded plugin")
    result = win.install_native_opencv(stage, manifest_path)
    assert "PyInit_cv2" in win.PEImage((stock / "cv2.pyd").read_bytes()).exports()
    assert set(path.name for path in stock.iterdir()) == {"cv2.pyd", "__init__.py"}
    assert result["manifest_sha256"] == win.sha256(manifest_path)


def test_native_opencv_hash_failure_preserves_previous_package(win, tmp_path):
    package = tmp_path / "compiled"
    package.mkdir()
    (package / "cv2.pyd").write_bytes(opencv_pe())
    manifest_path = tmp_path / "native.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "built",
                "platform": "windows-x64",
                "with_ipp": False,
                "with_ffmpeg": False,
                "package_directory": str(package),
                "files": {"cv2.pyd": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    stage = tmp_path / "stage"
    stock = stage / "runtime/Lib/site-packages/cv2"
    stock.mkdir(parents=True)
    (stock / "prior.pyd").write_bytes(b"prior")
    with pytest.raises(ValueError, match="hashes"):
        win.install_native_opencv(stage, manifest_path)
    assert (stock / "prior.pyd").read_bytes() == b"prior"


def test_no_native_pass_record_is_emitted_on_a_different_os(win, tmp_path, monkeypatch):
    checker = importlib.import_module("tools.check_windows_portable")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "program.exe").write_bytes(small_pe())
    (stage / win.MANIFEST).write_text(
        json.dumps(
            {
                "kind": "freedive_windows_portable",
                "files": win.file_hashes(stage),
                "payload_sha256": "a" * 64,
                "opencv_native": {"status": "built"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(checker.platform, "system", lambda: "Darwin")
    output = tmp_path / "proof.json"
    with pytest.raises(RuntimeError, match="Windows execution is unavailable"):
        checker.check(stage, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["--smoke-test", "--smoke-video"])
def test_packaged_gui_startup_failure_returns_without_dialog_in_both_smoke_modes(
    win, monkeypatch, mode
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "portable_bootstrap_test", ROOT / "packaging/windows/portable_bootstrap.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap.sys, "argv", ["app", "--launcher-gui", mode, "source"])

    def failure():
        raise RuntimeError("Simulated native startup failure")

    monkeypatch.setattr(bootstrap, "configure", failure)
    assert bootstrap.main() == 1


def test_qtnetwork_and_required_pywin32_closure_are_kept(win, tmp_path):
    files = {
        "PySide6/QtNetwork.pyd": small_pe(imported="Qt6Network.dll"),
        "PySide6/Qt6Network.dll": small_pe(),
        "win32/win32api.pyd": small_pe(imported="pywintypes313.dll"),
        "win32/win32job.pyd": small_pe(imported="pywintypes313.dll"),
        "win32/_win32sysloader.pyd": small_pe(),
        "win32comext/shell/shell.pyd": small_pe(imported="pythoncom313.dll"),
        "pywin32_system32/pythoncom313.dll": small_pe(imported="pywintypes313.dll"),
        "pywin32_system32/pywintypes313.dll": small_pe(),
        "win32/win32evtlog.pyd": small_pe(imported="unused-event-feature.dll"),
    }
    for name, data in files.items():
        file = tmp_path / "runtime/Lib/site-packages" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    report = win.native_audit(tmp_path, trim_qt=True)
    assert len(report["native_files"]) == 8
    assert len(report["omitted_unused_binaries"]) == 1
    assert not list(tmp_path.rglob("win32evtlog.pyd"))


def test_mcp_bootstrap_uses_stdio_server_without_desktop_import(win, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "portable_bootstrap_mcp_test", ROOT / "packaging/windows/portable_bootstrap.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    calls = []
    monkeypatch.setattr(bootstrap, "configure", lambda: calls.append("private-runtime"))
    monkeypatch.setattr(bootstrap.sys, "argv", ["app", "--launcher-mcp"])
    monkeypatch.setitem(
        sys.modules,
        "freedive_color.control.mcp_server",
        SimpleNamespace(main=lambda: calls.append("stdio")),
    )
    assert bootstrap.main() == 0
    assert calls == ["private-runtime", "stdio"]


def test_native_checker_rejects_non_protocol_stdio_output(win, tmp_path):
    import os
    import sys

    checker = importlib.import_module("tools.check_windows_portable")
    broken = tmp_path / "wrong-stdout.py"
    broken.write_text(
        "import time;print('diagnostic on stdout',flush=True);time.sleep(30)", encoding="utf-8"
    )
    with pytest.raises((ValueError, json.JSONDecodeError)):
        checker.check_mcp_stdio([sys.executable, str(broken)], dict(os.environ), tmp_path)
