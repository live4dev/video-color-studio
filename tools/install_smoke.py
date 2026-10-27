"""Extract the actual distribution into a fresh Unicode/spaced path and exercise it."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    arguments = parser.parse_args()
    archive = arguments.archive.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("Unsafe archive member")
        if bundle.testzip() is not None:
            raise ValueError("Distribution ZIP integrity check failed")
    base = ROOT / "artifacts" / "Проверка установки"
    base.mkdir(parents=True, exist_ok=True)
    destination, index = base / "Отдельная папка", 1
    while destination.exists():
        index += 1
        destination = base / f"Отдельная папка {index}"
    destination.mkdir()
    if sys.platform == "darwin":
        subprocess.run(["ditto", "-x", "-k", str(archive), str(destination)], check=True)
        executable = destination / "Freedive Color Studio.app/Contents/MacOS/Freedive Color Studio"
    elif sys.platform == "win32":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        executable = destination / "Freedive Color Studio/Freedive Color Studio.exe"
    else:
        raise SystemExit("Run distribution checks on macOS or Windows")
    if not executable.is_file():
        raise RuntimeError("The archive does not contain the expected desktop executable")
    if sys.platform == "win32":
        command = [
            sys.executable,
            str(ROOT / "tools/check_windows_portable.py"),
            "--stage",
            str(executable.parent),
            "--output",
            str(destination / "windows-native.json"),
            "--offscreen",
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "tools/smoke_bundle.py"),
            "--executable",
            str(executable),
        ]
    subprocess.run(command, cwd=ROOT, check=True)
    result = {
        "archive": str(archive),
        "installed_in": str(destination),
        "unicode_spaced_path": True,
        "photo_video_gui_smoke_passed": True,
        "execution_platform": sys.platform,
    }
    (destination / "installation-test.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
