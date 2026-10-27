#!/usr/bin/env python3
"""Verify an extracted portable payload; execute real GUI tests only on Windows."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from build_windows_portable import MANIFEST, ROOT, file_hashes, native_audit, sha256
except ImportError:
    from tools.build_windows_portable import MANIFEST, ROOT, file_hashes, native_audit, sha256


def check(stage: Path, output: Path, *, static_only=False, offscreen=False) -> dict:
    stage, output = stage.resolve(), output.resolve()
    if output.is_relative_to(stage):
        raise ValueError("Write test evidence outside the immutable application payload")
    if output.exists():
        raise FileExistsError("Native check report already exists; use a new output path")
    manifest = json.loads((stage / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("kind") != "freedive_windows_portable":
        raise ValueError("Not a generated Windows payload")
    if file_hashes(stage) != manifest["files"]:
        raise ValueError("Portable payload file hashes changed")
    audit = native_audit(stage)
    record = {
        "kind": "freedive_windows_check",
        "payload_sha256": manifest["payload_sha256"],
        "host_system": platform.system(),
        "static_integrity_passed": True,
        "native_files": len(audit["native_files"]),
        "windows_execution": "NOT RUN",
        "source_manifest_complete": manifest.get("sources_complete", False),
    }
    if not static_only:
        if platform.system() != "Windows":
            raise RuntimeError("Actual Windows execution is unavailable; use --static-only")
        if not manifest.get("opencv_native"):
            raise ValueError("Native acceptance requires the source-built IPP-free OpenCV")
        environment = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
        ):
            environment.pop(name, None)
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))
        if offscreen:
            environment["QT_QPA_PLATFORM"] = "offscreen"
        with tempfile.TemporaryDirectory(prefix="freedive-доказательство ") as temporary:
            diagnose_path = Path(temporary) / "проверка runtime.json"
            subprocess.run(
                [
                    str(stage / "Freedive Color Studio CLI.exe"),
                    "--diagnose-portable",
                    "--output",
                    str(diagnose_path),
                ],
                env=environment,
                check=True,
                timeout=60,
            )
            diagnose = json.loads(diagnose_path.read_text(encoding="utf-8"))
            photo = Path(temporary) / "подводный кадр с пробелами.jpg"
            photo_output = Path(temporary) / "результат 16 бит.png"
            shutil.copyfile(ROOT / "tests/fixtures/freediver_moalboal.jpg", photo)
            source_digest = sha256(photo)
            subprocess.run(
                [
                    str(stage / "Freedive Color Studio CLI.exe"),
                    "--input",
                    str(photo),
                    "--output",
                    str(photo_output),
                    "--preset",
                    "natural",
                    "--bit-depth",
                    "16",
                ],
                env=environment,
                check=True,
                timeout=60,
            )
            recipe = json.loads(
                photo_output.with_suffix(".png.freedive.json").read_text(encoding="utf-8")
            )
            if recipe["source"]["sha256"] != source_digest or sha256(photo) != source_digest:
                raise ValueError("Unicode source path or photo source integrity check failed")
            if recipe["output"]["bit_depth"] != 16:
                raise ValueError("The actual CLI did not export the requested 16-bit image")
        if not diagnose["isolated"] or not Path(diagnose["executable"]).resolve().is_relative_to(
            stage
        ):
            raise ValueError("The product used an external or non-isolated Python")
        if any(not Path(path).resolve().is_relative_to(stage) for path in diagnose["sys_path"]):
            raise ValueError("External Python module search path detected")
        native_modules = diagnose["opencv_native_modules"]
        if not native_modules or any(
            not Path(path).resolve().is_relative_to(stage) for path in native_modules.values()
        ):
            raise ValueError("Cannot confirm the packaged OpenCV extension was loaded")
        for line in diagnose["opencv_build_information"].splitlines():
            if ("Intel IPP:" in line or "FFMPEG:" in line) and line.split(":", 1)[
                1
            ].strip().upper() not in {"NO", "OFF", "DISABLED"}:
                raise ValueError("Unexpected IPP/FFmpeg enabled in the OpenCV extension")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/smoke_bundle.py"),
                "--executable",
                str(stage / "Freedive Color Studio.exe"),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=180,
        )
        smoke = json.loads(
            (ROOT / "artifacts/latest-bundle-smoke.json").read_text(encoding="utf-8")
        )
        if not smoke["passed"] or not smoke["video_gui_playback_and_export_verified"]:
            raise ValueError("Actual packaged photo/video GUI acceptance failed")
        record.update(
            {
                "windows_execution": "PASSED",
                "private_runtime_verified": True,
                "clean_opencv_verified": True,
                "unicode_cli_photo16_verified": True,
                "diagnostics": diagnose,
                "photo_video_gui_acceptance": smoke,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--offscreen", action="store_true", help="Native Windows CI without a display"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check(args.stage, args.output, static_only=args.static_only, offscreen=args.offscreen),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
