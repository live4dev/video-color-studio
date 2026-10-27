"""Build a local image-only OpenCV wheel without FFmpeg/AVFoundation class collisions.

Run with the project Python. All build/cache/log/source/wheel files live under the workspace.
The default builds and verifies a wheel; --install explicitly replaces only opencv-python-headless
in this Python environment after the wheel succeeds. Project dependency files are not edited.
"""

import argparse
import hashlib
import importlib.metadata
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
PACKAGE = "opencv-python-headless"
VERSION = "4.14.0.94"
PYPI_METADATA = f"https://pypi.org/pypi/{PACKAGE}/{VERSION}/json"
BASE_FLAGS = (
    "-DWITH_FFMPEG=OFF",
    "-DWITH_AVFOUNDATION=OFF",
    "-DWITH_GSTREAMER=OFF",
    "-DWITH_V4L=OFF",
    "-DWITH_MSMF=OFF",
    "-DWITH_DSHOW=OFF",
    "-DWITH_COCOA=OFF",
    "-DWITH_QT=OFF",
    "-DWITH_GTK=OFF",
    "-DWITH_OPENGL=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_IPP=OFF",
    "-DWITH_ITT=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DWITH_AVIF=OFF",
    "-DWITH_JPEGXL=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_opencv_apps=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_JPEG=ON",
    "-DBUILD_PNG=ON",
    "-DBUILD_TIFF=ON",
    "-DBUILD_ZLIB=ON",
    "-DBUILD_WEBP=ON",
    "-DBUILD_OPENJPEG=ON",
    "-DOPENCV_ENABLE_NONFREE=OFF",
)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(url: str, destination: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for the build-time source download")
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--retry",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            "600",
            url,
            "--output",
            str(destination),
        ],
        check=True,
    )


def verify_install(work: Path) -> dict:
    # Verification occurs in a clean child so an older mapped cv2 library cannot hide a bad install.
    probe = r"""
import cv2, json, numpy as np
try:
    import av
    av_version=av.__version__
except ImportError:
    av_version=None
information=cv2.getBuildInformation()
sample=np.full((8,11,3),[24,70,130],np.uint8)
checks={}
for extension in (".jpg", ".png", ".tiff"):
    ok,encoded=cv2.imencode(extension,sample)
    decoded=cv2.imdecode(encoded,cv2.IMREAD_UNCHANGED) if ok else None
    checks[extension]=bool(ok and decoded is not None and decoded.shape==sample.shape)
high=np.arange(16*20*3,dtype=np.uint16).reshape(16,20,3)*40
ok,encoded=cv2.imencode(".png",high)
decoded=cv2.imdecode(encoded,cv2.IMREAD_UNCHANGED)
checks["png16"]=bool(ok and decoded.dtype==np.uint16 and np.array_equal(decoded,high))
gray=np.tile(np.arange(32,dtype=np.uint8),(24,1))
flow=cv2.calcOpticalFlowFarneback(gray,gray,None,.5,2,15,2,5,1.1,0)
assert np.isfinite(flow).all()
assert hasattr(cv2,"gapi") and hasattr(cv2,"dnn")
print(json.dumps({"version":cv2.__version__,"file":cv2.__file__,"codecs":checks,
                  "gapi":hasattr(cv2,"gapi"),"dnn":hasattr(cv2,"dnn"),
                  "ffmpeg_backend":bool(cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG)),
                  "avfoundation_backend":bool(cv2.videoio_registry.hasBackend(cv2.CAP_AVFOUNDATION)),
                  "farneback":True,"pyav_version":av_version,"build_information":information}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )
    record = json.loads(result.stdout)
    information = record["build_information"]
    (work / "opencv-pyav-interop.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if "implemented in both" in result.stderr:
        raise RuntimeError("Native classes still collide when OpenCV and PyAV import together")
    # OpenCV 4.14 omits disabled backend lines rather than printing NO. Also ask
    # the native registry, which catches both linked and available plugin backends.
    for label, key in (("FFMPEG", "ffmpeg_backend"), ("AVFoundation", "avfoundation_backend")):
        if record[key] or re.search(rf"^\s*{label}:\s+YES", information, re.MULTILINE):
            raise RuntimeError(f"Custom wheel unexpectedly enables {label}")
    if not all(record["codecs"].values()):
        raise RuntimeError("Required photo codecs are unavailable in the custom build")
    native_files = list(Path(record["file"]).parent.glob("*.so"))
    if sys.platform == "darwin" and len(native_files) == 1:
        dependencies = subprocess.run(
            ["otool", "-L", str(native_files[0])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if re.search(r"libav(codec|format|device|util)|libsw(scale|resample)", dependencies):
            raise RuntimeError("Image OpenCV still links FFmpeg libraries")
        (work / "opencv-linked-libraries.txt").write_text(dependencies, encoding="utf-8")
        record["native_binary_sha256"] = digest(native_files[0])
        record["linked_libraries"] = dependencies.splitlines()[1:]
        load_commands = subprocess.run(
            ["otool", "-l", str(native_files[0])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        minimum = re.search(r"^\s*minos\s+(\d+(?:\.\d+)*)", load_commands, re.MULTILINE)
        if minimum is None or tuple(map(int, minimum[1].split(".")[:2])) > (13, 0):
            raise RuntimeError("Custom OpenCV does not declare macOS 13.0 compatibility")
        record["minimum_macos"] = minimum[1]
    (work / "opencv-build-information.txt").write_text(information, encoding="utf-8")
    return record


def install_built(work: Path) -> dict:
    work = work.expanduser().resolve()
    if not work.is_relative_to(ROOT):
        raise RuntimeError("Use a build directory inside the project")
    manifest_path = work / "native-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel = (ROOT / manifest["wheel"]).resolve()
    if (
        manifest.get("status") != "built"
        or not wheel.is_relative_to(ROOT)
        or digest(wheel) != manifest["wheel_sha256"]
    ):
        raise RuntimeError("No verified built wheel is available for installation")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install the custom wheel")
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            "--cache-dir",
            str(work / "cache"),
            str(wheel),
        ],
        check=True,
    )
    manifest["verification"] = verify_install(work)
    manifest["installed_distribution_version"] = importlib.metadata.version(PACKAGE)
    manifest["installed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps({"installed": True, "wheel": str(wheel), "manifest": str(manifest_path)}),
        flush=True,
    )
    return manifest


def build(work: Path, jobs: int, install: bool) -> dict:
    work = work.expanduser().resolve()
    if not work.is_relative_to(ROOT):
        raise RuntimeError("Use a work directory inside the project so build files remain scoped")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the isolated source wheel")
    for name in ("source", "cache", "tmp", "wheels", "notices"):
        (work / name).mkdir(parents=True, exist_ok=True)
    metadata_path = work / "source" / "pypi-metadata.json"
    if not metadata_path.exists():
        download(PYPI_METADATA, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources = [item for item in metadata["urls"] if item["packagetype"] == "sdist"]
    if len(sources) != 1:
        raise RuntimeError("PyPI metadata does not identify one immutable source archive")
    source = sources[0]
    if Path(source["filename"]).name != source["filename"]:
        raise RuntimeError("Unexpected source archive name")
    archive = work / "source" / source["filename"]
    if not archive.exists():
        download(source["url"], archive)
    source_sha256 = digest(archive)
    if source_sha256 != source["digests"]["sha256"]:
        raise RuntimeError("PyPI source archive SHA-256 does not match")
    flags = list(BASE_FLAGS)
    if sys.platform == "darwin":
        flags += [
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
            f"-DCMAKE_OSX_ARCHITECTURES={platform.machine()}",
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "ENABLE_HEADLESS": "1",
            "ENABLE_CONTRIB": "0",
            "CMAKE_ARGS": " ".join(flags),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs),
            "MAKEFLAGS": f"-j{jobs}",
            "UV_CACHE_DIR": str(work / "cache"),
            "PIP_CACHE_DIR": str(work / "cache" / "pip"),
            "TMPDIR": str(work / "tmp"),
            "SKBUILD_BUILD_OPTIONS": f"-j{jobs}",
        }
    )
    if sys.platform == "darwin":
        environment["MACOSX_DEPLOYMENT_TARGET"] = "13.0"
    command = [
        uv,
        "build",
        "--wheel",
        "--python",
        sys.executable,
        "--out-dir",
        str(work / "wheels"),
        "--cache-dir",
        str(work / "cache"),
        str(archive),
    ]
    manifest = {
        "package": PACKAGE,
        "version": VERSION,
        "source_url": source["url"],
        "source_sha256": source_sha256,
        "source_bytes": archive.stat().st_size,
        "pypi_metadata": PYPI_METADATA,
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "flags": flags,
        "parallel_jobs": jobs,
        "headless": True,
        "purpose": "Remove image OpenCV's FFmpeg and AVFoundation native backends; video uses PyAV",
        "installed": False,
        "status": "building",
        "build_command": command,
    }
    manifest_path = work / "native-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Building {PACKAGE} {VERSION} from verified source; log: {work / 'build.log'}", flush=True
    )
    started = time.perf_counter()
    with (work / "build.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    if process.returncode:
        manifest.update(
            status="failed",
            returncode=process.returncode,
            elapsed_seconds=round(time.perf_counter() - started, 2),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError(f"OpenCV source build failed; inspect {work / 'build.log'}")
    wheels = list((work / "wheels").glob("opencv_python_headless-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one built OpenCV wheel")
    wheel = wheels[0]
    notices = []
    with zipfile.ZipFile(wheel) as built:
        if built.testzip() is not None:
            raise RuntimeError("Wheel ZIP integrity check failed")
        for name in built.namelist():
            if any(token in Path(name).name.lower() for token in ("license", "copying", "notice")):
                target = work / "notices" / name
                if not target.resolve().is_relative_to(work / "notices"):
                    raise RuntimeError("Unexpected wheel notice path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(built.read(name))
                notices.append(str(target.relative_to(work)))
    manifest.update(
        status="built",
        wheel=str(wheel.relative_to(ROOT)),
        wheel_sha256=digest(wheel),
        wheel_bytes=wheel.stat().st_size,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        notices=notices,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if install:
        print("Installing the verified custom wheel into this Python environment only", flush=True)
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--reinstall",
                "--cache-dir",
                str(work / "cache"),
                str(wheel),
            ],
            check=True,
            env=environment,
        )
        manifest["verification"] = verify_install(work)
        manifest["installed_distribution_version"] = importlib.metadata.version(PACKAGE)
        manifest["installed"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "wheel": manifest["wheel"],
                "installed": manifest["installed"],
                "manifest": str(manifest_path),
            }
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build" / "opencv-image-only")
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=4)
    parser.add_argument(
        "--install", action="store_true", help="Replace this environment's OpenCV after build"
    )
    parser.add_argument(
        "--install-existing",
        action="store_true",
        help="Install and verify the already built wheel without rebuilding",
    )
    arguments = parser.parse_args()
    if arguments.install_existing:
        install_built(arguments.work_dir)
    else:
        build(arguments.work_dir, arguments.jobs, arguments.install)
