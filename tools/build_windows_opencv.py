#!/usr/bin/env python3
"""Genuine IPP-free Windows cv2 cross build with target Python/NumPy headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    from build_windows_portable import acquire, compiler, locked_wheels
    from windows_native_host import host_plan
    from windows_pe import PEImage
except ImportError:
    from tools.build_windows_portable import acquire, compiler, locked_wheels
    from tools.windows_native_host import host_plan
    from tools.windows_pe import PEImage

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "4afa2ea1214453648be88259f035712454faa9039b686de7753569ba8eec1577"
SOURCE_URL = (
    "https://files.pythonhosted.org/packages/67/8a/"
    "51ff1c6287be0b6cd3070cf095875da4ff7d6eda38507e6ada5608786a78/"
    "opencv_python_headless-4.14.0.94.tar.gz"
)
SDK_SHA = "dce409744b088f94c6e37ba41f8100c384c27e029e843237364c602d30d475e5"
SDK_URL = "https://api.nuget.org/v3-flatcontainer/python/3.13.17/python.3.13.17.nupkg"
MODULES = "core,imgproc,imgcodecs,photo,dnn,video,python3"
BASE_FLAGS = [
    "-DCMAKE_SYSTEM_NAME=Windows",
    "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DBUILD_LIST={MODULES}",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_opencv_world=OFF",
    "-DWITH_IPP=OFF",
    "-DWITH_ITT=OFF",
    "-DWITH_FFMPEG=OFF",
    "-DWITH_AVFOUNDATION=OFF",
    "-DWITH_GSTREAMER=OFF",
    "-DWITH_MSMF=OFF",
    "-DWITH_DSHOW=OFF",
    "-DWITH_WIN32UI=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_OPENGL=OFF",
    "-DWITH_QT=OFF",
    "-DWITH_GTK=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DWITH_AVIF=OFF",
    "-DWITH_JPEGXL=OFF",
    "-DWITH_CUDA=OFF",
    "-DWITH_LAPACK=OFF",
    "-DWITH_VULKAN=OFF",
    "-DOPENCV_ENABLE_NONFREE=OFF",
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
    "-DPYTHON3_LIMITED_API=ON",
    "-DPYTHON3_LIMITED_API_VERSION=0x03090000",
    "-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++",
    "-DCMAKE_MODULE_LINKER_FLAGS=-static-libstdc++",
]


def cmake_path(value: os.PathLike[str] | str) -> str:
    """CMake serializes compiler paths into quoted source: use portable separators.

    Native Windows backslashes such as D:\\a are otherwise emitted unescaped into
    CMakeRCCompiler.cmake and become invalid escape sequences when it is included.
    Subprocess arguments remain individual strings, preserving spaces and Unicode.
    """
    return os.fspath(value).replace("\\", "/")


def loader_source(extension: str = "cv2.pyd") -> str:
    """Native initialization must use cv2, rather than the invalid cv2.cv2 name."""
    if Path(extension).name != extension:
        raise ValueError("Expected a native extension basename")
    literal = json.dumps(extension)
    return (
        '"""Minimal OpenCV loader; compiled image/DNN/flow modules only."""\n'
        "import numpy as _numpy\nimport importlib.util as _util\n"
        "from pathlib import Path as _Path\n"
        "_spec = _util.spec_from_file_location(__name__, "
        f"_Path(__file__).with_name({literal}))\n"
        "_native = _util.module_from_spec(_spec)\n_spec.loader.exec_module(_native)\n"
        "globals().update({k:v for k,v in vars(_native).items() if not k.startswith('__')})\n"
        "__version__ = _native.__version__\n__doc__ = _native.__doc__\n"
    )


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_log_tail(path: Path) -> str:
    """Keep the actual compiler failure visible when a CI build directory is ephemeral."""
    with path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - 16384))
        return "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-60:])


def prepare_sdk(work: Path, numpy_wheel: Path, *, offline: bool = False) -> tuple[Path, Path]:
    sdk = work / "sdk"
    sdk.mkdir(exist_ok=True)
    package = sdk / "python.3.13.17.nupkg"
    if not package.exists():
        if offline:
            raise FileNotFoundError("Python target SDK is not cached")
        with urllib.request.urlopen(SDK_URL, timeout=90) as source, package.open("wb") as target:
            shutil.copyfileobj(source, target)
    if digest(package) != SDK_SHA:
        raise ValueError("Official Python target SDK hash mismatch")
    python = sdk / "python"
    if not python.exists():
        with zipfile.ZipFile(package) as archive:
            archive.extractall(python)
    numpy = sdk / "numpy"
    if not numpy.exists():
        with zipfile.ZipFile(numpy_wheel) as archive:
            for name in archive.namelist():
                if name.startswith("numpy/_core/include/"):
                    archive.extract(name, numpy)
    return python / "tools", numpy / "numpy/_core/include"


def build(work: Path, cache: Path, zig_path: Path | None, jobs: int, offline: bool) -> dict:
    work = work.expanduser().resolve()
    if not work.is_relative_to(ROOT):
        raise ValueError("Keep native build files in the project")
    work.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "build/opencv-image-only/source/opencv_python_headless-4.14.0.94.tar.gz"
    if not archive.exists():
        if offline:
            raise FileNotFoundError("OpenCV source archive is not cached")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as source, archive.open("wb") as output:
            shutil.copyfileobj(source, output)
    if digest(archive) != SOURCE_SHA:
        raise ValueError("OpenCV source hash mismatch")
    zig = compiler(cache, zig_path, offline=offline)
    numpy_record = next(row for row in locked_wheels() if row["package"] == "numpy")
    numpy_wheel = acquire(numpy_record, cache, offline=offline)
    python, numpy = prepare_sdk(work, numpy_wheel, offline=offline)
    source = work / "source/opencv_python_headless-4.14.0.94/opencv"
    if not source.exists():
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(work / "source", filter="data")
    target = "x86_64-windows-gnu"
    host = host_plan(work, zig)
    flags = BASE_FLAGS + [
        f"-DCMAKE_C_COMPILER={cmake_path(host['cc'])}",
        f"-DCMAKE_CXX_COMPILER={cmake_path(host['cxx'])}",
        f"-DCMAKE_AR={cmake_path(host['ar'])}",
        f"-DCMAKE_RANLIB={cmake_path(host['ranlib'])}",
        f"-DCMAKE_RC_COMPILER={cmake_path(host['rc'])}",
        f"-DPYTHON3_EXECUTABLE={cmake_path(sys.executable)}",
        f"-DPYTHON_DEFAULT_EXECUTABLE={cmake_path(sys.executable)}",
        f"-DPYTHON3_INCLUDE_PATH={cmake_path(python / 'include')}",
        f"-DPYTHON3_INCLUDE_DIR={cmake_path(python / 'include')}",
        f"-DPYTHON3_LIBRARY={cmake_path(python / 'libs/python3.lib')}",
        f"-DPYTHON3_LIBRARIES={cmake_path(python / 'libs/python3.lib')}",
        f"-DPYTHON3_NUMPY_INCLUDE_DIRS={cmake_path(numpy)}",
        "-DPYTHON3_NUMPY_VERSION=2.5.4",
        f"-DOPENCV_PYTHON3_INSTALL_PATH={cmake_path(work / 'install')}",
        "-DOPENCV_SKIP_PYTHON_LOADER=ON",
        "-DOPENCV_PYTHON3_INSTALL_PATH_SETUPVARS=OFF",
    ]
    configure = host["cmake"] + [
        "-S",
        str(source),
        "-B",
        str(work / "cmake"),
        "-G",
        host["generator"],
        *flags,
    ]
    compile_command = host["cmake"] + ["--build", str(work / "cmake"), "--parallel", str(jobs)]
    environment = os.environ.copy()
    environment.update(
        ZIG_LOCAL_CACHE_DIR=str(work / "zig-cache"),
        ZIG_GLOBAL_CACHE_DIR=str(ROOT / "build/windows-srt/zig-global-cache"),
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
    )
    environment.update(host["environment"])
    manifest = {
        "status": "building",
        "platform": "windows-x64",
        "version": "4.14.0.94",
        "target": target,
        "source_sha256": SOURCE_SHA,
        "python_sdk_url": SDK_URL,
        "python_sdk_sha256": SDK_SHA,
        "numpy_wheel": numpy_record,
        "zig_version": "0.14.1",
        "host_tools": host,
        "flags": flags,
        "modules": MODULES.split(","),
        "configure_command": configure,
        "build_command": compile_command,
        "with_ipp": False,
        "with_ffmpeg": False,
        "with_opencl": False,
        "windows_execution": "NOT RUN",
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    with (work / "build.log").open("w", encoding="utf-8") as log:
        for command in (configure, compile_command):
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                manifest.update(status="failed", returncode=process.returncode)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                log.flush()
                raise RuntimeError(
                    f"Windows OpenCV build failed; inspect {work / 'build.log'}\n"
                    f"Last build output:\n{build_log_tail(work / 'build.log')}"
                )
    binaries = list((work / "cmake").rglob("cv2*.pyd"))
    if not binaries:
        # The binding generator gets its suffix from the host interpreter. The
        # cross-linker's bytes are still PE; validate before installing as .pyd.
        binaries = list((work / "cmake/lib/python3").glob("cv2*.so"))
    if len(binaries) != 1:
        raise RuntimeError(f"Expected one Windows cv2 extension: {binaries}")
    image = PEImage(binaries[0].read_bytes())
    imports = image.imports() | image.imports(delayed=True)
    if image.machine != 0x8664 or "PyInit_cv2" not in image.exports():
        raise RuntimeError("Windows extension ABI audit failed")
    if any(
        any(token in name for token in ("ipp", "avcodec", "avformat", "libstdc", "libgcc"))
        for name in imports
    ):
        raise RuntimeError("Extension still imports an excluded native backend/runtime")
    package = work / "package/cv2"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binaries[0], package / "cv2.pyd")
    shutil.copyfile(source / "LICENSE", package / "LICENSE-OpenCV.txt")
    (package / "__init__.py").write_text(loader_source(), encoding="utf-8")
    files = {
        p.relative_to(package).as_posix(): digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
    }
    manifest.update(
        status="built",
        elapsed_seconds=round(time.perf_counter() - started, 2),
        package_directory=str(package),
        files=files,
        imports=imports,
        cv2_pyd_sha256=digest(package / "cv2.pyd"),
        exports=sorted(image.exports()),
        native_execution_tested=False,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"status": "built", "manifest": str(manifest_path), "package_directory": str(package)}
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/windows-opencv")
    parser.add_argument("--cache", type=Path, default=ROOT / "build/windows-inputs")
    parser.add_argument("--zig", type=Path)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    build(args.work_dir, args.cache, args.zig, args.jobs, args.offline)


if __name__ == "__main__":
    main()
