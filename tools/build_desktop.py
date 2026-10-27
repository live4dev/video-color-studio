"""Run on the target OS; no cross-compilation or unearned signing claims."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from audit_macos import audit
from collect_licenses import collect

ROOT = Path(__file__).resolve().parents[1]


def build() -> None:
    if sys.platform not in ("darwin", "win32"):
        raise SystemExit("Run this script on macOS or Windows.")
    collect(ROOT / "build" / "licenses")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(ROOT / "packaging" / "freedive.spec"),
        ],
        cwd=ROOT,
        check=True,
    )
    architecture = platform.machine().lower()
    if sys.platform == "darwin":
        audit(
            ROOT / "dist" / "Freedive Color Studio.app", ROOT / "artifacts" / "macos-targets.json"
        )
        archive = ROOT / "dist" / f"Freedive-Color-Studio-macOS-{architecture}.zip"
        subprocess.run(
            [
                "ditto",
                "-c",
                "-k",
                "--sequesterRsrc",
                "--keepParent",
                str(ROOT / "dist" / "Freedive Color Studio.app"),
                str(archive),
            ],
            check=True,
        )
    else:
        shutil.make_archive(
            str(ROOT / "dist" / f"Freedive-Color-Studio-Windows-{architecture}"),
            "zip",
            root_dir=ROOT / "dist",
            base_dir="Freedive Color Studio",
        )


if __name__ == "__main__":
    build()
