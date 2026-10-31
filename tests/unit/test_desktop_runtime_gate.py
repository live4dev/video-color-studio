"""Desktop packaging must not silently ship the too-new stock macOS runtime."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools import build_desktop, collect_licenses


@pytest.fixture
def audited_runtime(tmp_path, monkeypatch):
    work = tmp_path / "build"
    work.mkdir()
    site = tmp_path / "site"
    library = site / "onnxruntime/capi/runtime.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"audited source-built runtime")
    manifest = {
        "status": "built",
        "source_commit": build_desktop.SOURCE_COMMIT,
        "version": "1.23.2",
        "target": "13.0",
        "telemetry": False,
        "coreml": False,
        "verification": {
            "runtime": {"identity_inference": True, "providers": ["CPUExecutionProvider"]},
            "binaries": [
                {
                    "path": "onnxruntime/capi/runtime.so",
                    "minimum_macos": ["13.0"],
                    "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    distribution = SimpleNamespace(version="1.23.2", locate_file=lambda relative: site / relative)
    monkeypatch.setattr(
        build_desktop.importlib.metadata, "distribution", lambda _name: distribution
    )
    return work, library, manifest


def save_manifest(work, manifest):
    (work / "native-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_freeze_accepts_exact_installed_audited_source_runtime(audited_runtime):
    work, _library, manifest = audited_runtime
    save_manifest(work, manifest)
    build_desktop.require_macos_onnxruntime(work)


def test_stock_runtime_without_source_provenance_has_actionable_build_error(audited_runtime):
    work, _library, _manifest = audited_runtime
    with pytest.raises(RuntimeError, match="Build and install the macOS 13.0 CPU runtime first"):
        build_desktop.require_macos_onnxruntime(work)


def test_replaced_or_modified_runtime_cannot_reuse_an_old_audit(audited_runtime):
    work, library, manifest = audited_runtime
    save_manifest(work, manifest)
    library.write_bytes(b"different or too-new installed wheel")
    with pytest.raises(RuntimeError, match="differs from the audited source build"):
        build_desktop.require_macos_onnxruntime(work)


@pytest.mark.parametrize("failure", ["minimum", "source", "cpu", "telemetry", "no-binaries"])
def test_native_runtime_provenance_failures_stop_packaging(audited_runtime, failure):
    work, _library, manifest = audited_runtime
    if failure == "minimum":
        manifest["verification"]["binaries"][0]["minimum_macos"] = ["13.4"]
    elif failure == "source":
        manifest["source_commit"] = "f" * 40
    elif failure == "cpu":
        manifest["verification"]["runtime"]["providers"] = ["CoreMLExecutionProvider"]
    elif failure == "telemetry":
        manifest["telemetry"] = True
    else:
        manifest["verification"]["binaries"] = []
    save_manifest(work, manifest)
    with pytest.raises(RuntimeError, match="--jobs 2 --install"):
        build_desktop.require_macos_onnxruntime(work)


def test_native_notices_and_same_named_manifests_are_both_retained(tmp_path):
    root, destination = tmp_path / "project", tmp_path / "licenses"
    destination.mkdir()
    for directory, component in (
        ("opencv-image-only", "opencv"),
        ("onnxruntime-macos", "onnxruntime"),
    ):
        folder = root / "build" / directory
        (folder / "notices").mkdir(parents=True)
        (folder / "notices/LICENSE.txt").write_text(component + " license", encoding="utf-8")
        (folder / "native-manifest.json").write_text(component + " source record", encoding="utf-8")
    collect_licenses.collect_native_provenance(root, destination)
    assert (destination / "native-manifest.json").read_text() == "opencv source record"
    assert (
        destination / "onnxruntime-native-manifest.json"
    ).read_text() == "onnxruntime source record"
    assert (destination / "native-opencv/LICENSE.txt").read_text() == "opencv license"
    assert (destination / "native-onnxruntime/LICENSE.txt").read_text() == "onnxruntime license"
