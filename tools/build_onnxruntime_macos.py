"""Build official CPU ONNX Runtime 1.23.2 for macOS 13.0, without binary retargeting.

Developer-only network access fetches pinned source/dependency archives. The application
ships the resulting wheel's native libraries and performs no downloads. Run with Python
3.12 and CMake>=3.28, Ninja, setuptools, wheel, packaging and NumPy installed. All source,
compiler output, notices and provenance remain below the selected project build folder.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.23.2"
SOURCE_COMMIT = "a83fc4d58cb48eb68890dd689f94f28288cf2278"
SOURCE_URL = "https://github.com/microsoft/onnxruntime.git"
TARGET = "13.0"
MAGICS = {
    bytes.fromhex(value)
    for value in ("cffaedfe", "feedfacf", "cefaedfe", "feedface", "cafebabe", "bebafeca")
}
IDENTITY_MODEL_HEX = (
    "08083a3f0a100a017812017922084964656e7469747912096370752d736d6f6b65"
    "5a0f0a0178120a0a08080112040a020802620f0a0179120a0a08080112040a02080242040a001011"
)
CMAKE_DEFINES = (
    "CMAKE_OSX_DEPLOYMENT_TARGET=13.0",
    "onnxruntime_BUILD_UNIT_TESTS=OFF",
    "onnxruntime_USE_TELEMETRY=OFF",
    "onnxruntime_USE_COREML=OFF",
    "onnxruntime_USE_XNNPACK=OFF",
    "onnxruntime_USE_MIMALLOC=OFF",
    "onnxruntime_ENABLE_LTO=OFF",
)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(data)
    return checksum.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_logged(command: list[str], log: Path, *, cwd: Path, env: dict) -> None:
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        raise RuntimeError(f"Native build failed ({process.returncode}); {log}\n{tail[-16384:]}")


def build_command(source: Path, work: Path, jobs: int, architecture: str) -> list[str]:
    return [
        sys.executable,
        str(source / "tools/ci_build/build.py"),
        "--build_dir",
        str(work / "compile"),
        "--config",
        "Release",
        "--update",
        "--build",
        "--build_wheel",
        "--build_shared_lib",
        "--skip_tests",
        "--skip_submodule_sync",
        "--compile_no_warning_as_error",
        "--no_kleidiai",
        "--parallel",
        str(jobs),
        "--cmake_generator",
        "Ninja",
        "--osx_arch",
        architecture,
        "--cmake_extra_defines",
        *CMAKE_DEFINES,
        f"CMAKE_OSX_ARCHITECTURES={architecture}",
    ]


def minimum_versions(commands: str) -> list[str]:
    versions = re.findall(r"\bminos\s+(\d+(?:\.\d+)+)", commands)
    versions += re.findall(
        r"LC_VERSION_MIN_MACOSX.*?\bversion\s+(\d+(?:\.\d+)+)", commands, flags=re.DOTALL
    )
    return versions


def verify_wheel(wheel: Path, unpacked: Path, architecture: str) -> dict:
    """Reject every too-new Mach-O slice before installing anything into Python."""
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Built wheel failed ZIP integrity verification")
        for member in archive.infolist():
            destination = (unpacked / member.filename).resolve()
            if not destination.is_relative_to(unpacked.resolve()):
                raise RuntimeError("Unsafe wheel member path")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    metadata = list(unpacked.glob("onnxruntime-*.dist-info/METADATA"))
    if len(metadata) != 1 or f"Version: {VERSION}\n" not in metadata[0].read_text(encoding="utf-8"):
        raise RuntimeError("Built wheel has unexpected package metadata")
    binaries = []
    for path in sorted(unpacked.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(4) not in MAGICS:
                continue
        commands = subprocess.run(
            ["/usr/bin/otool", "-l", str(path)], check=True, capture_output=True, text=True
        ).stdout
        versions = minimum_versions(commands)
        if not versions or any(tuple(map(int, v.split("."))) > (13, 0, 0) for v in versions):
            raise RuntimeError(f"Native library requires macOS newer than 13.0: {path}: {versions}")
        architectures = subprocess.run(
            ["/usr/bin/lipo", "-archs", str(path)], check=True, capture_output=True, text=True
        ).stdout.split()
        if architectures != [architecture]:
            raise RuntimeError(f"Unexpected native wheel architectures: {architectures}")
        binaries.append(
            {
                "path": str(path.relative_to(unpacked)),
                "sha256": digest(path),
                "minimum_macos": versions,
                "architectures": architectures,
            }
        )
    if not binaries:
        raise RuntimeError("Built wheel contains no Mach-O libraries")
    # Run in a new isolated interpreter, forcing imports from the audited wheel rather
    # than the source tree or an already-installed PyPI wheel.
    probe = """
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
import onnxruntime as ort
ort.disable_telemetry_events()
session = ort.InferenceSession(bytes.fromhex(sys.argv[2]), providers=['CPUExecutionProvider'])
sample = np.array([.25, .75], np.float32)
assert np.array_equal(session.run(None, {'x': sample})[0], sample)
assert ort.__version__ == '1.23.2'
assert session.get_providers() == ['CPUExecutionProvider']
print(json.dumps({'version':ort.__version__, 'file':ort.__file__,
                  'providers':session.get_providers(), 'identity_inference':True,
                  'telemetry':False}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(unpacked), IDENTITY_MODEL_HEX],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return {"binaries": binaries, "runtime": json.loads(result.stdout), "target": TARGET}


def source_dependencies(source: Path) -> list[dict]:
    dependencies = []
    for line in (source / "cmake/deps.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, url, sha1 = line.split(";")[:3]
        dependencies.append({"name": name, "url": url, "upstream_sha1": sha1})
    return dependencies


def prepare_source(work: Path) -> tuple[Path, str]:
    """Git verifies the immutable commit and every fetched tree/blob object by its hash.

    Omit only unused test-model data. No submodules are synced: this CPU configuration
    fetches its C++ dependencies using the hash-locked upstream cmake/deps.txt.
    """
    source = work / "source" / f"onnxruntime-{SOURCE_COMMIT}"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    commands = [
        ["git", "init", str(source)],
        ["git", "remote", "add", "origin", SOURCE_URL],
        ["git", "config", "remote.origin.promisor", "true"],
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        ["git", "sparse-checkout", "init", "--no-cone"],
        ["git", "sparse-checkout", "set", "--no-cone", "/*", "!/onnxruntime/test/testdata/"],
        ["git", "fetch", "--depth=1", "--filter=blob:none", "--no-tags", "origin", SOURCE_COMMIT],
        ["git", "checkout", "--detach", "FETCH_HEAD"],
    ]
    with (work / "source.log").open("w", encoding="utf-8") as log:
        for command in commands:
            subprocess.run(command, cwd=source, check=True, stdout=log, stderr=subprocess.STDOUT)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError("Official ONNX Runtime source commit does not match its lock")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    return source, tree


def collect_notices(source: Path, work: Path) -> list[str]:
    result = []
    roots = [source, *sorted((work / "compile/Release/_deps").glob("*-src"))]
    for index, root in enumerate(roots):
        # Upstream's aggregate notice covers its embedded third-party source; fetched
        # dependency roots additionally retain their own exact copyright/license files.
        paths = root.iterdir() if index == 0 else root.rglob("*")
        for path in paths:
            if path.is_file() and any(
                word in path.name.lower() for word in ("license", "copying", "notice", "copyright")
            ):
                relative = Path(root.name) / path.relative_to(root)
                destination = work / "notices" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
                result.append(str(destination.relative_to(work)))
    if not any(name.endswith("/LICENSE") for name in result):
        raise RuntimeError("ONNX Runtime source license was not retained")
    return result


def install_existing(work: Path) -> dict:
    manifest = json.loads((work / "native-manifest.json").read_text(encoding="utf-8"))
    wheel = (work / manifest["wheel"]).resolve()
    if (
        manifest.get("status") != "built"
        or not wheel.is_relative_to(work)
        or digest(wheel) != manifest["wheel_sha256"]
    ):
        raise RuntimeError("No verified source-built ONNX Runtime wheel is available")
    manifest["verification"] = verify_wheel(wheel, work / "wheel-audit", platform.machine())
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for installing the verified wheel")
    subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--reinstall", str(wheel)],
        check=True,
    )
    manifest["installed"] = True
    write_json(work / "native-manifest.json", manifest)
    return manifest


def build(work: Path, jobs: int, install: bool = False) -> dict:
    work = work.expanduser().resolve()
    if not work.is_relative_to(ROOT / "build"):
        raise RuntimeError("Use an ONNX Runtime work directory below the project's build folder")
    if sys.platform != "darwin" or platform.machine() not in ("arm64", "x86_64"):
        raise RuntimeError("This CPU source build requires a native macOS arm64 or x86_64 host")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for the standalone macOS runtime wheel")
    work.mkdir(parents=True, exist_ok=True)
    source, source_tree = prepare_source(work)
    if (source / "VERSION_NUMBER").read_text(encoding="utf-8").strip() != VERSION:
        raise RuntimeError("Unexpected ONNX Runtime source version")
    environment = os.environ.copy()
    environment["MACOSX_DEPLOYMENT_TARGET"] = TARGET
    command = build_command(source, work, jobs, platform.machine())
    manifest = {
        "package": "onnxruntime",
        "version": VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "source_tree": source_tree,
        "source_integrity": "Git commit/tree/blob object hashes; no source patches",
        "dependency_lock_sha256": digest(source / "cmake/deps.txt"),
        "upstream_dependencies": source_dependencies(source),
        "target": TARGET,
        "architecture": platform.machine(),
        "python": sys.version,
        "build_command": command,
        "telemetry": False,
        "coreml": False,
        "kleidiai": False,
        "status": "building",
        "installed": False,
        "runtime_on_macos13_tested": False,
    }
    write_json(work / "native-manifest.json", manifest)
    started = time.perf_counter()
    try:
        run_logged(command, work / "build.log", cwd=source, env=environment)
        built = list((work / "compile/Release/dist").glob(f"onnxruntime-{VERSION}-*.whl"))
        if len(built) != 1:
            raise RuntimeError("Expected exactly one source-built ONNX Runtime wheel")
        verification = verify_wheel(built[0], work / "wheel-audit", platform.machine())
        wheels = work / "wheels"
        wheels.mkdir(exist_ok=True)
        wheel = wheels / built[0].name
        shutil.copy2(built[0], wheel)
        manifest.update(
            status="built",
            wheel=str(wheel.relative_to(work)),
            wheel_sha256=digest(wheel),
            wheel_bytes=wheel.stat().st_size,
            verification=verification,
            notices=collect_notices(source, work),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    except Exception as exc:
        manifest.update(
            status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started
        )
        write_json(work / "native-manifest.json", manifest)
        raise
    write_json(work / "native-manifest.json", manifest)
    if install:
        manifest = install_existing(work)
    print(json.dumps({"status": manifest["status"], "wheel": manifest["wheel"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/onnxruntime-macos")
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=3)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-existing", action="store_true")
    args = parser.parse_args()
    if args.install_existing:
        install_existing(args.work_dir.expanduser().resolve())
    else:
        build(args.work_dir, args.jobs, args.install)
