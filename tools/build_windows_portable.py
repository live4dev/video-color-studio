#!/usr/bin/env python3
"""Prepare an actual Windows x64 portable app from locked wheels and a native launcher.

Windows execution is never inferred from cross-compilation or PE inspection. This
build script may download developer inputs; the resulting application never does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

try:
    from windows_pe import PEImage, inspect_pe
except ImportError:
    from tools.windows_pe import PEImage, inspect_pe

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.13.17"
PYTHON_RUNTIME = {
    "url": "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-amd64.zip",
    "sha256": "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38",
}
ZIG_VERSION = "0.14.1"
ZIG_ARCHIVES = {
    ("Darwin", "arm64"): (
        "aarch64-macos",
        "39f3dc5e79c22088ce878edc821dedb4ca5a1cd9f5ef915e9b3cc3053e8faefa",
    ),
    ("Linux", "x86_64"): (
        "x86_64-linux",
        "24aeeec8af16c381934a6cd7d95c807a8cb2cf7df9fa40d359aa884195c4716c",
    ),
    ("Windows", "AMD64"): (
        "x86_64-windows",
        "554f5378228923ffd558eac35e21af020c73789d87afeabf4bfd16f2e6feed2c",
    ),
}
SYSTEM_DLLS = set(
    [
        "advapi32",
        "authz",
        "avicap32",
        "bcrypt",
        "comdlg32",
        "crypt32",
        "d2d1",
        "d3d11",
        "d3d12",
        "d3d9",
        "d3dcompiler_47",
        "dnsapi",
        "dwmapi",
        "dwrite",
        "dxgi",
        "gdi32",
        "imagehlp",
        "imm32",
        "iphlpapi",
        "kernel32",
        "mpr",
        "msvcrt",
        "ncrypt",
        "netapi32",
        "odbc32",
        "ole32",
        "oleaut32",
        "opengl32",
        "propsys",
        "psapi",
        "rpcrt4",
        "secur32",
        "setupapi",
        "shell32",
        "shlwapi",
        "uiautomationcore",
        "user32",
        "userenv",
        "uxtheme",
        "version",
        "winhttp",
        "winmm",
        "ws2_32",
        "wsock32",
        "wtsapi32",
    ]
)
QT_ROOTS = {"QtCore.pyd", "QtGui.pyd", "QtWidgets.pyd"}
NATIVE_SUFFIXES = {".dll", ".pyd", ".exe"}
MANIFEST = "BUILD-MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def acquire(entry: dict, cache: Path, *, offline: bool = False) -> Path:
    url, expected = entry["url"], entry["sha256"]
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.python.org", "files.pythonhosted.org", "ziglang.org"}
        or not re.fullmatch("[0-9a-f]{64}", expected)
    ):
        raise ValueError("Build inputs require an approved HTTPS origin and a pinned SHA-256")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(parsed.path).name
    if path.exists():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Cached input failed SHA-256 verification: {path.name}")
        return path
    if offline:
        raise FileNotFoundError(f"Input not cached: {path.name}")
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=cache)
    os.close(fd)
    temporary = Path(temporary)
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded input failed SHA-256 verification: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def locked_wheels(project: Path = ROOT) -> list[dict]:
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    lock = tomllib.loads((project / "uv.lock").read_text())
    environment = {
        **default_environment(),
        "sys_platform": "win32",
        "os_name": "nt",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "python_version": "3.13",
        "python_full_version": PYTHON_VERSION,
        "implementation_name": "cpython",
        "implementation_version": PYTHON_VERSION,
    }
    packages = {}
    for package in lock["package"]:
        packages.setdefault(canonicalize_name(package["name"]), []).append(package)
    supported = list(cpython_tags((3, 13), abis=["cp313"], platforms=["win_amd64"]))
    supported += list(compatible_tags((3, 13), interpreter="cp313", platforms=["win_amd64"]))
    priorities = {tag: index for index, tag in enumerate(supported)}
    selected, pending = {}, [Requirement(text) for text in metadata["project"]["dependencies"]]
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        matches = [
            p for p in packages.get(name, []) if Version(p["version"]) in requirement.specifier
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one locked Windows version for {name}")
        package = matches[0]
        if name in selected:
            if selected[name]["version"] != package["version"]:
                raise ValueError(f"Conflicting locked versions for {name}")
            continue
        candidates = []
        for wheel in package.get("wheels", []):
            filename = Path(urlparse(wheel["url"]).path).name
            tags = parse_wheel_filename(filename)[3]
            rank = min((priorities[tag] for tag in tags if tag in priorities), default=None)
            if rank is not None:
                candidates.append((rank, filename, wheel))
        if not candidates:
            raise ValueError(f"No locked CPython3.13 Windows x64 wheel for {name}")
        _rank, filename, wheel = min(candidates)
        selected[name] = {
            "package": name,
            "version": package["version"],
            "filename": filename,
            "url": wheel["url"],
            "sha256": wheel["hash"].removeprefix("sha256:"),
        }
        for dependency in package.get("dependencies", []):
            if dependency.get("marker") and not Marker(dependency["marker"]).evaluate(environment):
                continue
            pending.append(
                Requirement(
                    dependency["name"]
                    + ("==" + dependency["version"] if dependency.get("version") else "")
                )
            )
    if selected.get("av", {}).get("version") != "16.1.0":
        raise ValueError("Windows release currently audits official PyAV16.1.0 only")
    return [selected[name] for name in sorted(selected)]


def archive_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*pure.parts)


def extract_zip(path: Path, destination: Path, *, wheel: bool = False) -> list[Path]:
    paths = []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            relative = archive_path(item.filename)
            if item.is_dir():
                continue
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not allowed in Windows build inputs")
            if wheel and relative.parts[0].endswith(".data"):
                if len(relative.parts) < 3 or relative.parts[1] not in {"purelib", "platlib"}:
                    continue
                relative = Path(*relative.parts[2:])
            target = destination / relative
            if target.exists():
                raise FileExistsError(f"Overlapping build input: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            paths.append(target)
    return paths


def system_dll(name: str) -> bool:
    return (
        name.startswith(("api-ms-win-", "ext-ms-win-"))
        or (name.endswith(".dll") and name[:-4] in SYSTEM_DLLS)
        or name == "winspool.drv"
    )


def native_audit(stage: Path, *, trim_qt: bool = False) -> dict:
    rows, by_name = [], {}
    for path in sorted(stage.rglob("*")):
        if path.suffix.lower() not in NATIVE_SUFFIXES or not path.is_file():
            continue
        pe = PEImage(path.read_bytes())
        if pe.machine != 0x8664:
            raise ValueError(f"Non-AMD64 native file: {path}")
        relative = path.relative_to(stage).as_posix()
        row = {
            "path": relative,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "machine": hex(pe.machine),
            "subsystem": pe.subsystem,
            "imports": sorted(set(pe.imports()) | set(pe.imports(delayed=True))),
        }
        rows.append(row)
        by_name.setdefault(path.name.lower(), []).append(row)
    pending = []
    unused_video_plugins = {
        Path(row["path"]).name.lower()
        for row in rows
        if Path(row["path"]).name.lower().startswith("opencv_videoio_ffmpeg")
    }
    for row in rows:
        name = row["path"]
        if Path(name).name.lower() in unused_video_plugins:
            continue
        if "/PySide6/" not in name or (
            Path(name).name in QT_ROOTS
            or any(
                "/PySide6/plugins/" + part + "/" in name
                for part in ("platforms", "styles", "iconengines", "imageformats")
            )
            and not name.endswith("/qpdf.dll")
        ):
            pending.append(row)
    selected, external = {}, set()
    while pending:
        row = pending.pop()
        if row["path"] in selected:
            continue
        selected[row["path"]] = row
        for dependency in row["imports"]:
            if dependency in unused_video_plugins:
                raise ValueError(
                    "An unexpected loader import requires OpenCV's unused video plugin"
                )
            if dependency in by_name:
                pending.extend(by_name[dependency])
            elif system_dll(dependency):
                external.add(dependency)
            else:
                raise ValueError(f"Unbundled native dependency {dependency} from {row['path']}")
    omitted = [row["path"] for row in rows if row["path"] not in selected]
    if trim_qt:
        for relative in omitted:
            (stage / relative).unlink()
        qml = stage / "runtime/Lib/site-packages/PySide6/qml"
        if qml.is_dir():
            shutil.rmtree(qml)
    elif omitted:
        raise ValueError("Unexpected unused/unvalidated native binaries remain")
    return {
        "native_files": sorted(selected.values(), key=lambda row: row["path"]),
        "system_imports": sorted(external),
        "omitted_unused_binaries": omitted,
    }


def compiler(cache: Path, supplied: Path | None, *, offline: bool) -> Path:
    if supplied:
        executable = supplied.resolve()
    else:
        target, digest = ZIG_ARCHIVES[(platform.system(), platform.machine())]
        suffix = ".zip" if "windows" in target else ".tar.xz"
        filename = f"zig-{target}-{ZIG_VERSION}{suffix}"
        entry = {"url": f"https://ziglang.org/download/{ZIG_VERSION}/{filename}", "sha256": digest}
        archive = acquire(entry, cache, offline=offline)
        folder = cache / filename.removesuffix(suffix)
        executable = folder / ("zig.exe" if os.name == "nt" else "zig")
        if not executable.exists():
            if suffix == ".zip":
                extract_zip(archive, cache)
            else:
                with tarfile.open(archive) as source:
                    source.extractall(cache, filter="data")
    version = subprocess.check_output([str(executable), "version"], text=True).strip()
    if version != ZIG_VERSION:
        raise ValueError(f"Use the audited compiler Zig{ZIG_VERSION}, got{version}")
    return executable


def compile_launchers(stage: Path, zig: Path, cache: Path) -> list[dict]:
    environment = {**os.environ, "ZIG_GLOBAL_CACHE_DIR": str(cache / "zig-cache")}
    results = []
    for gui, name in (
        (True, "Freedive Color Studio.exe"),
        (False, "Freedive Color Studio CLI.exe"),
    ):
        destination = stage / name
        arguments = [
            str(zig),
            "cc",
            "-target",
            "x86_64-windows-gnu",
            "-O2",
            "-static",
            "-s",
            "-municode",
            f"-DFREEDIVE_GUI={int(gui)}",
            "-Wl,--subsystem," + ("windows" if gui else "console"),
            str(ROOT / "packaging/windows/portable_launcher.c"),
            "-luser32",
            "-o",
            str(destination),
        ]
        subprocess.run(arguments, check=True, env=environment)
        result = inspect_pe(destination)
        if result["machine"] != "0x8664" or result["subsystem"] != (2 if gui else 3):
            raise ValueError("Launcher machine/subsystem does not match Windows x64")
        results.append(
            {
                "path": name,
                "sha256": sha256(destination),
                "machine": result["machine"],
                "subsystem": result["subsystem"],
            }
        )
    return results


def collect_wheel_notices(stage: Path) -> None:
    site = stage / "runtime/Lib/site-packages"
    destination = stage / "licenses"
    destination.mkdir()
    for source in sorted(site.rglob("*")):
        if source.is_file() and any(
            text in source.name.lower() for text in ("license", "copying", "copyright", "notice")
        ):
            target = destination / "wheels" / source.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(stage / "runtime/LICENSE.txt", destination / "PYTHON_LICENSE.txt")
    for name in ("LICENSE", "THIRD_PARTY_LICENSES.md", "README.md"):
        path = ROOT / name
        if path.is_file():
            shutil.copyfile(path, stage / name)
    shutil.copyfile(ROOT / "packaging/media/NOTICE.md", destination / "NOTICE.md")


def install_native_opencv(stage: Path, manifest_path: Path) -> dict:
    """Replace the complete stock package; no binary-only IPP implementation remains."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        data.get("status") != "built"
        or data.get("platform") != "windows-x64"
        or data.get("with_ipp") is not False
        or data.get("with_ffmpeg") is not False
    ):
        raise ValueError("Windows requires the verified IPP/FFmpeg-free native OpenCV build")
    package = Path(data["package_directory"])
    package = package if package.is_absolute() else ROOT / package
    actual = {
        path.relative_to(package).as_posix(): sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
    }
    if not data["files"] or actual != data["files"]:
        raise ValueError("Native OpenCV package hashes do not match its build manifest")
    extensions = list(package.rglob("*.pyd"))
    if len(extensions) != 1:
        raise ValueError("Expected exactly one genuine OpenCV extension")
    pe = PEImage(extensions[0].read_bytes())
    if pe.machine != 0x8664 or "PyInit_cv2" not in pe.exports():
        raise ValueError("Native OpenCV must export AMD64 PyInit_cv2")
    destination = stage / "runtime/Lib/site-packages/cv2"
    if destination.is_dir():
        shutil.rmtree(destination)
    shutil.copytree(package, destination)
    return {**data, "manifest_sha256": sha256(manifest_path)}


def file_hashes(stage: Path) -> dict[str, str]:
    return {
        p.relative_to(stage).as_posix(): sha256(p)
        for p in sorted(stage.rglob("*"))
        if p.is_file() and p.name != MANIFEST and "licenses" not in p.relative_to(stage).parts
    }


def prepare(
    stage: Path,
    cache: Path,
    zig: Path | None,
    *,
    offline: bool,
    opencv_manifest: Path | None = None,
) -> dict:
    if stage.exists():
        raise FileExistsError("Stage already exists; choose a new folder or use --finalize-only")
    wheels = locked_wheels()
    runtime = acquire(PYTHON_RUNTIME, cache, offline=offline)
    wheel_paths = [(entry, acquire(entry, cache, offline=offline)) for entry in wheels]
    zig = compiler(cache, zig, offline=offline)
    stage.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".windows-stage-", dir=stage.parent))
    try:
        extract_zip(runtime, temporary / "runtime")
        site = temporary / "runtime/Lib/site-packages"
        for _entry, path in wheel_paths:
            extract_zip(path, site, wheel=True)
        native_cv = install_native_opencv(temporary, opencv_manifest) if opencv_manifest else None
        app = temporary / "app/freedive_color"
        shutil.copytree(
            ROOT / "src/freedive_color",
            app,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        if not list((app / "assets/models").glob("*.onnx")):
            raise ValueError("The offline segmentation model is missing")
        shutil.copyfile(
            ROOT / "packaging/windows/portable_bootstrap.py", temporary / "portable_bootstrap.py"
        )
        (temporary / "runtime/python313._pth").write_text(
            "python313.zip\n.\nLib/site-packages\n../app\n", encoding="ascii"
        )
        launchers = compile_launchers(temporary, zig, cache)
        audit = native_audit(temporary, trim_qt=True)
        collect_wheel_notices(temporary)
        (temporary / "WINDOWS-STATUS.txt").write_text(
            "Windows 10/11 x64 portable build\n"
            "Prepared by cross-compilation and PE dependency inspection.\n"
            "WINDOWS EXECUTION: NOT RUN. This is an untested Windows build.\n"
            "Extract the entire folder, then open Freedive Color Studio.exe.\n"
            "Python, administrator privileges, API keys and internet are not required.\n"
            "CLI: Freedive Color Studio CLI.exe --help\n"
            "License notices are in licenses/. Corresponding sources accompany this ZIP.\n",
            encoding="utf-8",
        )
        hashes = file_hashes(temporary)
        payload = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        manifest = {
            "kind": "freedive_windows_portable",
            "schema_version": 1,
            "platform": "windows-x64",
            "windows_execution": "NOT RUN",
            "host": {"system": platform.system(), "machine": platform.machine()},
            "python": {"version": PYTHON_VERSION, **PYTHON_RUNTIME},
            "compiler": {
                "version": ZIG_VERSION,
                "launcher_source_sha256": sha256(ROOT / "packaging/windows/portable_launcher.c"),
            },
            "wheels": wheels,
            "opencv_native": native_cv,
            "launchers": launchers,
            **audit,
            "files": hashes,
            "payload_sha256": payload,
            "sources_complete": False,
            "signature": "unsigned",
        }
        json_write(temporary / MANIFEST, manifest)
        temporary.rename(stage)
        print(
            json.dumps(
                {
                    "stage": str(stage),
                    "native_files": len(audit["native_files"]),
                    "payload_sha256": payload,
                    "windows_execution": "NOT RUN",
                }
            ),
            flush=True,
        )
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def finalize(stage: Path, output: Path, marker: Path) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Release ZIP or JSON exists; select a new output name")
    manifest = json.loads((stage / MANIFEST).read_text())
    if manifest.get("kind") != "freedive_windows_portable":
        raise ValueError("Not a generated Windows portable stage")
    if file_hashes(stage) != manifest["files"]:
        raise ValueError("Staged application changed; rebuild before release")
    if not manifest.get("opencv_native"):
        raise ValueError("Stock OpenCV is provisional; build the source-based IPP-free extension")
    sources = json.loads(marker.read_text())
    if sources.get("complete") is not True or "windows-x64" not in sources.get("platforms", []):
        raise ValueError("Verified corresponding sources must explicitly cover Windows x64")
    if sources.get("windows_payload_sha256") != manifest["payload_sha256"]:
        raise ValueError("Corresponding source inventory does not match this Windows stage")
    archive = Path(sources["archive"])
    archive = archive if archive.is_absolute() else ROOT / archive
    if not archive.is_file() or sha256(archive) != sources["sha256"]:
        raise ValueError("Corresponding source ZIP hash mismatch")
    with zipfile.ZipFile(archive) as source_zip:
        for path in sorted((stage / "app/freedive_color").rglob("*")):
            if path.is_file():
                name = "application/src/" + path.relative_to(stage / "app").as_posix()
                if hashlib.sha256(source_zip.read(name)).hexdigest() != sha256(path):
                    raise ValueError("Source archive has different application code/assets")
        notices = [
            name
            for name in source_zip.namelist()
            if name.startswith("notices/") and not name.endswith("/")
        ]
        if not notices:
            raise ValueError("Corresponding source notices are missing")
        for name in notices:
            target = stage / "licenses/native-media" / archive_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_zip.read(name))
    native_audit(stage)
    manifest["sources_complete"] = True
    manifest["corresponding_source"] = {
        "archive": archive.name,
        "sha256": sources["sha256"],
        "bytes": archive.stat().st_size,
    }
    json_write(stage / MANIFEST, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    companion = output.parent / archive.name
    if companion.resolve() != archive.resolve():
        if companion.exists() and sha256(companion) != sources["sha256"]:
            raise FileExistsError("A different source archive already exists beside the release")
        if not companion.exists():
            shutil.copyfile(archive, companion)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".windows-release-", suffix=".zip", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive_zip:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    name = stage.name + "/" + path.relative_to(stage).as_posix()
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.external_attr = 0o100644 << 16
                    archive_zip.writestr(
                        entry,
                        path.read_bytes(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=6,
                    )
        with zipfile.ZipFile(temporary) as archive_zip:
            if archive_zip.testzip() is not None:
                raise ValueError("Release ZIP integrity verification failed")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    release = {
        "archive": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "windows_execution": "NOT RUN",
        "sources": manifest["corresponding_source"],
    }
    json_write(output.with_suffix(".json"), release)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", type=Path, default=ROOT / "build/windows-portable/Freedive Color Studio"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "build/windows-inputs")
    parser.add_argument(
        "--zig", type=Path, help="Audited Zig0.14.1 executable; otherwise fetched with pinned hash"
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--opencv-manifest", type=Path, help="Verified clean Windows OpenCV package"
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Create release ZIP only after source verification"
    )
    parser.add_argument(
        "--source-marker",
        type=Path,
        default=ROOT / "dist/Freedive_Color_Studio_Corresponding_Source.json",
    )
    arguments = parser.parse_args()
    if not arguments.finalize_only:
        prepare(
            arguments.stage.resolve(),
            arguments.cache.resolve(),
            arguments.zig,
            offline=arguments.offline,
            opencv_manifest=arguments.opencv_manifest,
        )
    if arguments.output:
        print(
            json.dumps(
                finalize(
                    arguments.stage.resolve(),
                    arguments.output.resolve(),
                    arguments.source_marker.resolve(),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
