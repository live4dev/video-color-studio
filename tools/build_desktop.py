"""Build desktop distributions and record platform-specific verification limits."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

try:
    from audit_macos import audit
    from build_onnxruntime_macos import SOURCE_COMMIT, VERSION
    from collect_licenses import collect
    from verify_models import verify_models
except ImportError:
    from tools.audit_macos import audit
    from tools.build_onnxruntime_macos import SOURCE_COMMIT, VERSION
    from tools.collect_licenses import collect
    from tools.verify_models import verify_models

ROOT = Path(__file__).resolve().parents[1]


def require_macos_onnxruntime(work: Path | None = None) -> None:
    """Freeze only the exact installed CPU binaries audited for the macOS 13 target.

    PyPI's 1.23.2 arm64 wheel declares 13.0 but actually requires 13.4. A source
    build is mandatory; changing Mach-O headers or accepting the wheel filename
    would hide the incompatibility. The final app receives a second full audit.
    """
    work = work or ROOT / "build/onnxruntime-macos"
    remedy = (
        "Build and install the macOS 13.0 CPU runtime first: "
        "uv run --no-sync python tools/build_onnxruntime_macos.py --jobs 2 --install"
    )
    try:
        manifest = json.loads((work / "native-manifest.json").read_text(encoding="utf-8"))
        distribution = importlib.metadata.distribution("onnxruntime")
        verification = manifest["verification"]
        runtime = verification["runtime"]
        binaries = verification["binaries"]
        if (
            manifest["status"] != "built"
            or manifest["source_commit"] != SOURCE_COMMIT
            or manifest["version"] != VERSION
            or distribution.version != VERSION
            or manifest["target"] != "13.0"
            or manifest.get("telemetry") is not False
            or manifest.get("coreml") is not False
            or runtime.get("identity_inference") is not True
            or runtime.get("providers") != ["CPUExecutionProvider"]
            or not binaries
        ):
            raise ValueError("Source build provenance or CPU verification is incomplete")
        for binary in binaries:
            relative = Path(binary["path"])
            versions = binary["minimum_macos"]
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.parts[0] != "onnxruntime"
                or not versions
                or any(tuple(map(int, value.split("."))) > (13, 0, 0) for value in versions)
            ):
                raise ValueError("Native runtime did not pass the macOS 13.0 audit")
            installed = Path(distribution.locate_file(relative))
            if (
                not installed.is_file()
                or hashlib.sha256(installed.read_bytes()).hexdigest() != binary["sha256"]
            ):
                raise ValueError("Installed runtime differs from the audited source build")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        raise RuntimeError(f"{remedy}\n{exc}") from exc


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
    require_macos_onnxruntime()
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
