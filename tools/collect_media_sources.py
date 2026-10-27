#!/usr/bin/env python3
"""Collect pinned corresponding sources; never called by the desktop application.

Archives are retained verbatim, checksum-verified, and never executed. The output
includes upstream recipes/patches, licenses, the current app sources and native
inventory. Missing sources or an unknown bundled Qt module prevent a release ZIP.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "packaging" / "media"
DEFAULT_WORK = ROOT / "build" / "media-sources"
DEFAULT_OUTPUT = ROOT / "dist" / "Freedive_Color_Studio_Corresponding_Source.zip"
QTBASE = {
    "QtCore",
    "QtDBus",
    "QtGui",
    "QtNetwork",
    "QtWidgets",
    "QtOpenGL",
    "QtOpenGLWidgets",
    "QtPrintSupport",
    "QtConcurrent",
    "QtSql",
    "QtTest",
    "QtXml",
}
QT_MODULES = {**dict.fromkeys(QTBASE, "qtbase"), "QtSvg": "qtsvg", "QtSvgWidgets": "qtsvg"}
PROJECT_DIRS = ("src", "tools", "packaging", "docs", "tests", ".github")
PROJECT_FILES = (
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "README.md",
    "DECISIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES.md",
    "TESTING.md",
    "MILESTONES.md",
    "AGENTS.md",
    "FREEDIVE_COLOR_STUDIO_SPEC.md",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside_project(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Keep source downloads and archives inside the project")
    return path


def validate_entry(entry: dict) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError(f"No pinned SHA-256 for {entry['name']}")
    name = entry["filename"]
    if not name or name != Path(name).name or name in {".", ".."}:
        raise ValueError("Source filename must be a single basename")
    if not entry.get("urls") or any(urlparse(u).scheme != "https" for u in entry["urls"]):
        raise ValueError("Source downloads must use HTTPS")
    timeout = entry.get("download_timeout", 90)
    if type(timeout) is not int or not 10 <= timeout <= 300:
        raise ValueError("Source download timeout must be 10..300 seconds")


def merge_mcp_sources(lock: dict, inventory: dict, windows_bundle: Path | None = None) -> dict:
    """Add the separately pinned SDK graph without weakening existing native gates."""
    path = RECORDS / "mcp-sources-lock.json"
    if not path.is_file():
        raise RuntimeError("MCP source lock is missing; no complete source archive can be claimed")
    sdk = json.loads(path.read_text(encoding="utf-8"))
    expected = sdk["required_versions"]
    if windows_bundle:
        versions = {
            row["package"].lower().replace("_", "-"): row["version"] for row in inventory["wheels"]
        }
    else:
        versions = {}
        for name in expected:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(f"Pinned MCP dependency is missing: {name}") from exc
    for name, version in expected.items():
        if versions.get(name) != version:
            raise RuntimeError(f"MCP source lock mismatch: {name} requires {version}")
    if windows_bundle and versions.get("pywin32") != sdk["windows_pywin32_version"]:
        raise RuntimeError("Windows pywin32 source lock mismatch")
    expected_ssl = sdk["openssl_version"]
    if windows_bundle:
        native = windows_bundle / "runtime/Lib/site-packages/cryptography/hazmat/bindings/_rust.pyd"
        ssl_versions = {
            match.decode("ascii")
            for match in re.findall(
                rb"OpenSSL ([0-9]+\.[0-9]+\.[0-9]+) [0-9]{1,2} [A-Za-z]{3} [0-9]{4}",
                native.read_bytes(),
            )
        }
        if ssl_versions != {expected_ssl}:
            raise RuntimeError("Windows cryptography OpenSSL source version is unverified")
    else:
        from cryptography.hazmat.bindings.openssl.binding import Binding

        binding = Binding()
        version_text = binding.ffi.string(binding.lib.OpenSSL_version(0)).decode("ascii")
        if not version_text.startswith(f"OpenSSL {expected_ssl} "):
            raise RuntimeError("Installed cryptography OpenSSL source version changed")
    merged = {**lock, "provenance": {**lock["provenance"]}}
    merged["provenance"]["mcp"] = {
        **sdk["provenance"],
        "source_lock_sha256": sha256(path),
        "required_versions": expected,
        "openssl_version": expected_ssl,
    }
    for key in ("sources", "source_access", "extra_notices"):
        merged[key] = [*lock.get(key, []), *sdk.get(key, [])]
    for entry in sdk["sources"]:
        validate_entry(entry)
    return merged


def acquire(entry: dict, work: Path, *, offline: bool = False) -> dict:
    validate_entry(entry)
    destination = work / "archives" / entry["filename"]
    if destination.is_file():
        if sha256(destination) != entry["sha256"]:
            raise ValueError(f"Cached source hash mismatch: {destination.name}")
        return {**entry, "bytes": destination.stat().st_size, "download": "verified-cache"}
    if offline:
        raise FileNotFoundError(f"Source not cached: {destination.name}")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for developer source collection")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    errors = []
    for url in entry["urls"]:
        process = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "10",
                "--max-time",
                str(entry.get("download_timeout", 90)),
                "--speed-time",
                "25",
                "--speed-limit",
                "1024",
                url,
                "--output",
                str(temporary),
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode == 0 and sha256(temporary) == entry["sha256"]:
            temporary.replace(destination)
            return {**entry, "bytes": destination.stat().st_size, "download": url}
        errors.append(
            {
                "url": url,
                "returncode": process.returncode,
                "reason": process.stderr.strip() or "SHA-256 mismatch",
            }
        )
        temporary.unlink(missing_ok=True)
    raise RuntimeError(json.dumps({"name": entry["name"], "attempts": errors}))


def native_inventory() -> dict:
    import av

    if av.__version__ != "15.1.0":
        raise RuntimeError("Source lock applies to PyAV 15.1.0 only")
    for package in ("PySide6-Essentials", "shiboken6"):
        if importlib.metadata.version(package) != "6.8.3":
            raise RuntimeError("Source lock applies to Qt/PySide/Shiboken 6.8.3 only")
    base = Path(av.__file__).parent
    roots = (base / ".dylibs", base.parent / "av.libs", base)
    natives = sorted(
        {
            p
            for root in roots
            if root.exists()
            for p in root.iterdir()
            if p.is_file() and (p.suffix.lower() in {".dll", ".dylib"} or ".so" in p.name)
        }
    )
    library = next((p for p in natives if re.match(r"(?:lib)?avcodec[.\-]", p.name)), None)
    if library is None:
        raise RuntimeError("Cannot find the actual PyAV libavcodec for provenance")
    handle = ctypes.CDLL(str(library))
    handle.avcodec_configuration.restype = ctypes.c_char_p
    configuration = handle.avcodec_configuration().decode("utf-8")
    if "--enable-libfdk" in configuration or "--enable-nonfree" in configuration:
        raise RuntimeError("This collector does not authorize a nonfree FFmpeg distribution")
    if sys.platform != "darwin":
        raise RuntimeError("Non-macOS vendor DLL sources need their exact runner inventory first")
    return {
        "platform": sys.platform,
        "pyav_version": av.__version__,
        "ffmpeg_libraries": av.library_versions,
        "ffmpeg_configuration": configuration,
        "native_files": [
            {"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in natives
        ],
        "license_basis": (
            "GPLv3-compatible combined distribution; upstream x264/x265 guard patch "
            "does not change their licenses"
        ),
        "windows_status": "Not asserted: MinGW/OpenSSL copied DLLs require exact Windows inventory",
    }


def audit_qt(bundle: Path | None) -> dict:
    if bundle is None:
        return {
            "basis": "Audited milestone-A macOS bundle",
            "modules": ["QtCore", "QtDBus", "QtGui", "QtNetwork", "QtSvg", "QtWidgets"],
        }
    bundle = bundle.resolve()
    frameworks = sorted({p.stem for p in bundle.rglob("Qt*.framework") if p.is_dir()})
    if not frameworks:
        raise RuntimeError("No bundled Qt frameworks found for source audit")
    unknown = sorted(set(frameworks) - QT_MODULES.keys())
    if unknown:
        raise RuntimeError(f"Missing corresponding Qt submodule sources: {unknown}")
    return {"basis": str(bundle.relative_to(ROOT)), "modules": frameworks}


def windows_inventory(bundle: Path) -> dict:
    bundle = inside_project(bundle)
    manifest = json.loads((bundle / "BUILD-MANIFEST.json").read_text())
    if (
        manifest.get("kind") != "freedive_windows_portable"
        or manifest.get("platform") != "windows-x64"
    ):
        raise RuntimeError("Expected a generated Windows portable stage")
    actual = {
        p.relative_to(bundle).as_posix(): sha256(p)
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
        and p.name != "BUILD-MANIFEST.json"
        and "licenses" not in p.relative_to(bundle).parts
    }
    payload = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    if actual != manifest["files"] or payload != manifest["payload_sha256"]:
        raise RuntimeError("Windows stage changed since its native audit")
    native = manifest.get("opencv_native")
    if (
        not native
        or native.get("status") != "built"
        or native.get("with_ipp") is not False
        or native.get("with_ffmpeg") is not False
    ):
        raise RuntimeError(
            "Windows source gate requires the genuine IPP/FFmpeg-free OpenCV extension"
        )
    package = bundle / "runtime/Lib/site-packages/cv2"
    files = {
        p.relative_to(package).as_posix(): sha256(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
    }
    if files != native["files"]:
        raise RuntimeError("Staged OpenCV does not match its source build manifest")
    av = next(row for row in manifest["wheels"] if row["package"] == "av")
    if av["version"] != "16.1.0" or av["sha256"] != (
        "565093ebc93b2f4b76782589564869dadfa83af5b852edebedd8fee746457d06"
    ):
        raise RuntimeError("Source lock applies only to the audited Windows PyAV16.1 wheel")
    media = [name for name in actual if "/av.libs/" in name]
    if any(
        any(token in name.lower() for token in ("ssl", "crypto", "srt", "gmp")) for name in media
    ):
        raise RuntimeError("Windows stage contains an unapproved media dependency")
    qt = sorted({"Qt" + p.stem[3:] for p in bundle.rglob("Qt6*.dll")})
    unknown = sorted(set(qt) - QT_MODULES.keys())
    if unknown:
        raise RuntimeError(f"Missing corresponding Qt submodule sources: {unknown}")
    return {
        "platform": "windows-x64",
        "pyav_version": "16.1.0",
        "ffmpeg_version": "8.0.1",
        "windows_payload_sha256": payload,
        "windows_execution": "NOT RUN",
        "opencv_native": native,
        "qt_bundle": {"modules": qt},
        "native_files": manifest["native_files"],
        "wheels": manifest["wheels"],
    }


def extract_notices(archive: Path, output: Path) -> list[dict]:
    """Read regular license members only; never extract arbitrary archive paths."""
    if archive.name.endswith(".tar.zst"):
        return extract_msys_notices(archive, output)
    if zipfile.is_zipfile(archive):
        notices = []
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if member.is_dir() or member.file_size > 4 * 1024 * 1024:
                    continue
                if not re.match(
                    r"(?i)(COPYING|LICENSE|LICENCE|NOTICE)([.\-_]|$)", Path(member.filename).name
                ):
                    continue
                data = source.read(member)
                output.mkdir(parents=True, exist_ok=True)
                name = (
                    hashlib.sha256(member.filename.encode()).hexdigest()[:12]
                    + "-"
                    + Path(member.filename).name
                )
                (output / name).write_bytes(data)
                notices.append(
                    {
                        "member": member.filename,
                        "path": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        return notices
    notices = []
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            basename = Path(member.name).name
            license_member = re.match(
                r"(?i)(COPYING|LICENSE|LICENCE|NOTICE)([.\-_]|$)", basename
            ) or any(part.upper() == "LICENSES" for part in Path(member.name).parts)
            if not member.isfile() or not license_member:
                continue
            if member.size > 4 * 1024 * 1024:
                continue
            stream = source.extractfile(member)
            if stream is None:
                continue
            data = stream.read()
            token = hashlib.sha256(member.name.encode()).hexdigest()[:12]
            target = output / f"{token}-{basename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            notices.append(
                {
                    "member": member.name,
                    "path": target.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return notices


def extract_msys_notices(archive: Path, output: Path) -> list[dict]:
    """Read original source tarballs/recipes inside a verified MSYS package."""
    names = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
    output.mkdir(parents=True, exist_ok=True)
    notices = []
    for name in names:
        if name.endswith((".tar.xz", ".tar.gz", ".tar.bz2")):
            with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
                nested = Path(temporary) / Path(name).name
                with nested.open("wb") as target:
                    subprocess.run(["tar", "-xOf", str(archive), name], stdout=target, check=True)
                rows = extract_notices(nested, output)
                for row in rows:
                    row["member"] = name + "!/" + row["member"]
                notices.extend(rows)
        elif Path(name).name in {"PKGBUILD", ".SRCINFO"}:
            data = subprocess.check_output(["tar", "-xOf", str(archive), name])
            (output / Path(name).name).write_bytes(data)
            notices.append(
                {
                    "member": name,
                    "path": Path(name).name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "role": "build-recipe",
                }
            )
    return notices


def add_zip_file(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = (
        zipfile.ZIP_STORED
        if path.suffix in {".xz", ".gz", ".bz2", ".zst", ".nupkg"}
        else zipfile.ZIP_DEFLATED
    )
    info.external_attr = 0o100644 << 16
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        shutil.copyfileobj(source, target, 1024 * 1024)


def project_sources() -> list[Path]:
    paths = [ROOT / name for name in PROJECT_FILES if (ROOT / name).is_file()]
    for name in PROJECT_DIRS:
        paths.extend(
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".pyo"}
        )
    return sorted(set(paths))


def collect(
    work: Path,
    output: Path,
    *,
    offline: bool = False,
    bundle: Path | None = None,
    workers: int = 4,
    windows_bundle: Path | None = None,
) -> dict:
    work, output = inside_project(work), inside_project(output)
    (work / "archives").mkdir(parents=True, exist_ok=True)
    lock_name = "windows-sources-lock.json" if windows_bundle else "sources-lock.json"
    lock = json.loads((RECORDS / lock_name).read_text())
    inventory = windows_inventory(windows_bundle) if windows_bundle else native_inventory()
    if not windows_bundle:
        inventory["qt_bundle"] = audit_qt(bundle)
    lock = merge_mcp_sources(lock, inventory, windows_bundle)
    entries = lock["sources"][:]
    shared_archives = ROOT / "build/media-sources/archives"
    for entry in entries:
        destination = work / "archives" / entry["filename"]
        shared = shared_archives / entry["filename"]
        if not destination.exists() and shared.is_file() and sha256(shared) == entry["sha256"]:
            try:
                os.link(shared, destination)
            except OSError:
                shutil.copyfile(shared, destination)
    native_manifest = (
        ROOT
        / "build"
        / (
            "windows-opencv/manifest.json"
            if windows_bundle
            else "opencv-image-only/native-manifest.json"
        )
    )
    if not native_manifest.is_file():
        raise RuntimeError("Build the image-only OpenCV wheel before collecting release sources")
    native = json.loads(native_manifest.read_text())
    image_source = ROOT / "build/opencv-image-only/source/opencv_python_headless-4.14.0.94.tar.gz"
    if native.get("status") != "built" or sha256(image_source) != native["source_sha256"]:
        raise RuntimeError("Image-only OpenCV source manifest is not verified")
    cached_image_source = work / "archives" / image_source.name
    if not cached_image_source.exists():
        shutil.copyfile(image_source, cached_image_source)
    if not any(entry["name"] == "opencv-python-headless" for entry in entries):
        entries.append(
            {
                "name": "opencv-python-headless",
                "filename": image_source.name,
                "sha256": native["source_sha256"],
                "urls": [
                    native.get(
                        "source_url",
                        "https://files.pythonhosted.org/packages/67/8a/"
                        "51ff1c6287be0b6cd3070cf095875da4ff7d6eda38507e6ada5608786a78/"
                        "opencv_python_headless-4.14.0.94.tar.gz",
                    )
                ],
                "role": "Native image-only source; rebuild with tools/build_image_opencv.py",
            }
        )
    completed, missing = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(acquire, e, work, offline=offline): e for e in entries}
        for task in concurrent.futures.as_completed(tasks):
            entry = tasks[task]
            try:
                result = task.result()
                completed.append(result)
                print(f"Verified {entry['name']}: {result['bytes']} bytes", flush=True)
            except Exception as error:
                missing.append({"name": entry["name"], "error": str(error)})
                print(f"Missing {entry['name']}: {error}", file=sys.stderr, flush=True)
    manifest = {
        "complete": not missing,
        "sources": sorted(completed, key=lambda e: e["name"]),
        "missing": missing,
        "inventory": inventory,
        "provenance": lock["provenance"],
        "source_access": lock.get("source_access", []),
        "source_access_scope": (
            "Included archives are verified bytes. Separate immutable upstream source URLs "
            "provide the explicitly identified third-party source access; they are not "
            "claimed to be additional bundled archive bytes."
        ),
    }
    manifest_path = work / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if missing:
        raise RuntimeError("Corresponding source set is incomplete; no release ZIP was produced")
    for entry in manifest["sources"]:
        entry["notices"] = extract_notices(
            work / "archives" / entry["filename"], work / "notices" / entry["name"]
        )
    for notice in lock.get("extra_notices", []):
        source = ROOT / notice["path"]
        if sha256(source) != notice["sha256"]:
            raise RuntimeError("Explicit native notice checksum mismatch")
        target = work / "notices/explicit" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.zip")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        add_zip_file(archive, manifest_path, "source-manifest.json")
        add_zip_file(archive, native_manifest, "native/opencv-native-manifest.json")
        included = {}
        for entry in manifest["sources"]:
            previous = included.get(entry["filename"])
            if previous is not None:
                if previous != entry["sha256"]:
                    raise RuntimeError("Conflicting source checksums for one archive filename")
                continue
            included[entry["filename"]] = entry["sha256"]
            add_zip_file(
                archive, work / "archives" / entry["filename"], "archives/" + entry["filename"]
            )
        for path in sorted((work / "notices").rglob("*")):
            if path.is_file():
                add_zip_file(archive, path, path.relative_to(work).as_posix())
        for path in project_sources():
            add_zip_file(archive, path, "application/" + path.relative_to(ROOT).as_posix())
        if windows_bundle:
            for path in sorted((windows_bundle / "licenses").rglob("*")):
                if path.is_file():
                    name = (
                        "notices/windows-distribution/"
                        + path.relative_to(windows_bundle / "licenses").as_posix()
                    )
                    add_zip_file(archive, path, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Source ZIP integrity verification failed")
    temporary.replace(output)
    release = {
        "archive": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "complete": True,
        "platforms": ["windows-x64"]
        if windows_bundle
        else ["macos-arm64" if platform.machine() == "arm64" else "macos-x64"],
        "scope": "Source/native inventory verified; Windows execution NOT RUN"
        if windows_bundle
        else "Verified macOS vendor natives and audited Qt modules; Windows not asserted",
    }
    if windows_bundle:
        release["windows_payload_sha256"] = inventory["windows_payload_sha256"]
    output.with_suffix(".json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(json.dumps(release), flush=True)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path, help="Audit exact new macOS app frameworks")
    parser.add_argument("--windows-bundle", type=Path, help="Audit exact Windows portable stage")
    parser.add_argument(
        "--offline", action="store_true", help="Verify cached sources without network"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    arguments = parser.parse_args()
    collect(
        arguments.work_dir,
        arguments.output,
        offline=arguments.offline,
        bundle=arguments.bundle,
        workers=arguments.workers,
        windows_bundle=arguments.windows_bundle,
    )


if __name__ == "__main__":
    main()
