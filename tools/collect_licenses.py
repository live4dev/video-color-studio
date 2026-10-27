"""Collect exact installed dependency notices, including wheel-vendored native libraries."""

import importlib.metadata
import json
import shutil
import sys
import sysconfig
from pathlib import Path

DISTRIBUTIONS = (
    "numpy",
    "opencv-python-headless",
    "pillow",
    "PySide6-Essentials",
    "shiboken6",
    "tifffile",
    "pyinstaller",
)


def collect(destination: Path) -> None:
    if destination.is_dir():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"python": sys.version, "distributions": []}
    for name in DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        notices = []
        for entry in distribution.files or ():
            if any(
                part.lower() in ("licenses", "license") for part in entry.parts[:-1]
            ) or entry.name.lower().startswith(("license", "copying", "copyright", "notice")):
                source = Path(distribution.locate_file(entry))
                if source.is_file():
                    target = destination / name / str(entry).replace("..", "parent")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    notices.append(str(target.relative_to(destination)))
        manifest["distributions"].append(
            {"name": name, "version": distribution.version, "notices": notices}
        )
    # Include the actual runtime license; a bundled app also redistributes CPython.
    candidates = (
        Path(sys.base_prefix) / "LICENSE",
        Path(sys.base_prefix) / "LICENSE.txt",
        Path(sysconfig.get_path("stdlib")) / "LICENSE.txt",
    )
    runtime_license = next((path for path in candidates if path.is_file()), None)
    if runtime_license is None:
        raise RuntimeError(
            "CPython license not found in this runtime; supply it before distribution."
        )
    shutil.copyfile(runtime_license, destination / "PYTHON_LICENSE.txt")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    collect(Path(__file__).resolve().parents[1] / "build" / "licenses")
