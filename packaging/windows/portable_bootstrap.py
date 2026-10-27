"""Private portable runtime bootstrap; no downloads, installers or user site packages."""

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
SITE = RUNTIME / "Lib" / "site-packages"
DLL_HANDLES = []


def configure() -> None:
    if Path(sys.executable).resolve().parent != RUNTIME:
        raise RuntimeError("Use the application EXE and its included private runtime")
    if not sys.flags.isolated:
        raise RuntimeError("The application runtime must start in isolated mode")
    sys._MEIPASS = str(ROOT)
    os.environ["FREEDIVE_COLOR_PORTABLE_ROOT"] = str(ROOT)
    os.environ["FREEDIVE_COLOR_APP"] = str(ROOT / "Freedive Color Studio.exe")
    # The embeddable interpreter deliberately disables .pth execution. Reproduce
    # only pywin32's required local paths; never run user/site startup hooks.
    for directory in (SITE / "win32", SITE / "win32" / "lib", SITE / "Pythonwin"):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.append(str(directory))
    for directory in [
        RUNTIME,
        SITE / "PySide6",
        SITE / "shiboken6",
        SITE / "cv2",
        SITE / "PIL",
        SITE / "pywin32_system32",
        SITE / "win32",
        *sorted(SITE.glob("*.libs")),
    ]:
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    os.environ["QT_PLUGIN_PATH"] = str(SITE / "PySide6" / "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(SITE / "PySide6" / "plugins" / "platforms")


def diagnose(arguments: list[str]) -> int:
    import importlib.metadata

    import av
    import cv2
    import numpy
    import PIL
    import pythoncom
    import pywintypes
    import win32api
    import win32job
    import win32process
    import win32security
    from PySide6.QtCore import qVersion
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from win32com.shell import shell

    sample = numpy.full((32, 48, 3), (35, 95, 145), dtype=numpy.uint8)
    codecs = {}
    for extension in (".jpg", ".png", ".webp", ".tiff"):
        ok, encoded = cv2.imencode(extension, sample)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if ok else None
        codecs[extension] = decoded is not None and decoded.shape == sample.shape
    high = sample.astype(numpy.uint16) * 257
    ok, encoded = cv2.imencode(".png", high)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if ok else None
    codecs["png16_exact"] = decoded is not None and numpy.array_equal(high, decoded)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    if not all(codecs.values()) or not numpy.isfinite(flow).all():
        raise RuntimeError("Required image codecs or temporal optical flow are unavailable")

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "isolated": bool(sys.flags.isolated),
        "sys_path": sys.path,
        "versions": {
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
            "pyav": av.__version__,
            "qt": qVersion(),
        },
        "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
        "network_required": False,
        "mcp_version": importlib.metadata.version("mcp"),
        "local_transport": {
            "api_version": 1,
            "server": QLocalServer.__name__,
            "socket": QLocalSocket.__name__,
        },
        "windows_control_modules": [
            module.__name__
            for module in (
                pythoncom,
                pywintypes,
                win32api,
                win32job,
                win32process,
                win32security,
                shell,
            )
        ],
        "opencv_build_information": cv2.getBuildInformation(),
        "image_codec_checks": codecs,
        "optical_flow_finite": True,
        "opencv_native_modules": {
            name: module.__file__
            for name, module in sys.modules.items()
            if "cv2" in name and str(getattr(module, "__file__", "")).lower().endswith(".pyd")
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if "--output" in arguments:
        path = Path(arguments[arguments.index("--output") + 1])
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    elif sys.stdout is not None:
        print(text)
    return 0


def main() -> int:
    gui = len(sys.argv) > 1 and sys.argv[1] == "--launcher-gui"
    if len(sys.argv) < 2 or sys.argv[1] not in (
        "--launcher-gui",
        "--launcher-cli",
        "--launcher-mcp",
    ):
        raise RuntimeError("The native launcher mode is missing")
    mcp = sys.argv[1] == "--launcher-mcp"
    del sys.argv[1]
    try:
        configure()
        if mcp:
            from freedive_color.control.mcp_server import main as mcp_main

            mcp_main()
            return 0
        if "--diagnose-portable" in sys.argv:
            return diagnose(sys.argv[1:])
        from freedive_color.__main__ import main as application_main

        return application_main()
    except Exception:
        details = traceback.format_exc()
        if sys.stderr is not None:
            print(details, file=sys.stderr)
        if gui and not {"--smoke-test", "--smoke-video"}.intersection(sys.argv):
            import ctypes
            import tempfile

            directory = (
                Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
                / "Freedive Color Studio"
            )
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / "startup-error.log"
            log.write_text(details, encoding="utf-8")
            ctypes.windll.user32.MessageBoxW(
                None,
                "Не удалось открыть приложение. Подробности:\n" + str(log),
                "Freedive Color Studio",
                0x10,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
