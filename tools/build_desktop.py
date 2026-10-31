"""Build desktop distributions and record platform-specific verification limits."""

import hashlib
import json
import platform
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from audit_macos import audit
from collect_licenses import collect
from verify_models import verify_models

ROOT = Path(__file__).resolve().parents[1]


def build() -> None:
    if sys.platform not in ("darwin", "win32"):
        raise SystemExit("Run this script on macOS or Windows.")
    verify_models()
    if sys.platform == "win32":
        stage = ROOT / "build/windows-portable/Freedive Color Studio"
        commands = [
            ["tools/build_windows_opencv.py"],
            [
                "tools/build_windows_portable.py",
                "--stage",
                str(stage),
                "--opencv-manifest",
                str(ROOT / "build/windows-opencv/manifest.json"),
            ],
            [
                "tools/collect_media_sources.py",
                "--windows-bundle",
                str(stage),
                "--work-dir",
                str(ROOT / "build/windows-media-sources"),
                "--output",
                str(ROOT / "dist/Freedive_Color_Studio_Windows_Corresponding_Source.zip"),
            ],
            [
                "tools/build_windows_portable.py",
                "--stage",
                str(stage),
                "--finalize-only",
                "--source-marker",
                str(ROOT / "dist/Freedive_Color_Studio_Windows_Corresponding_Source.json"),
                "--output",
                str(ROOT / "dist/Freedive_Color_Studio_Windows_x64_UNTESTED.zip"),
            ],
        ]
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        return
    import cv2

    if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG) or cv2.videoio_registry.hasBackend(
        cv2.CAP_AVFOUNDATION
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "tools/build_image_opencv.py"), "--install"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "tools/collect_media_sources.py")], cwd=ROOT, check=True
    )
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
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/collect_media_sources.py"),
                "--offline",
                "--bundle",
                str(ROOT / "dist/Freedive Color Studio.app"),
            ],
            cwd=ROOT,
            check=True,
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
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError("macOS distribution ZIP integrity check failed")
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_record = json.loads(
            (ROOT / "dist/Freedive_Color_Studio_Corresponding_Source.json").read_text(
                encoding="utf-8"
            )
        )
        release = {
            "archive": str(archive.relative_to(ROOT)),
            "sha256": digest.hexdigest(),
            "bytes": archive.stat().st_size,
            "platform": "macOS-" + architecture,
            "version": tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
                "project"
            ]["version"],
            "minimum_os": "13.0",
            "signature": "ad-hoc; no Developer ID or notarization",
            "corresponding_source": source_record,
            "native_execution_evidence": "artifacts/latest-bundle-smoke.json",
            "untested_targets": [
                "macOS 13 runtime",
                "clean machine",
                "Windows",
                "Intel in this session",
            ],
        }
        archive.with_suffix(".json").write_text(json.dumps(release, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build()
