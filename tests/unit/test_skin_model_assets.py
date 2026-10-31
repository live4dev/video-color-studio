"""Model supply and packaging failures are explicit and never trigger a download."""

import importlib
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def assets(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("tools.verify_models")


def test_bundled_skin_model_has_verified_source_license_and_numerical_parity(assets):
    hashes = assets.verify_models()
    folder = ROOT / "src/freedive_color/assets/models"
    metadata = json.loads((folder / "SELFIE_MULTICLASS_METADATA.json").read_text(encoding="utf-8"))
    assert hashes["selfie_multiclass_256x256.onnx"] == metadata["onnx_sha256"]
    assert metadata["input"]["shape"] == [1, 256, 256, 3]
    assert metadata["input"]["normalization_mean"] == [127.5]
    assert metadata["input"]["normalization_std"] == [127.5]
    assert metadata["input"]["range"] == [-1, 1]
    assert metadata["output"]["shape"] == [1, 256, 256, 6]
    assert metadata["output"]["classes"][2:4] == ["body-skin", "face-skin"]
    assert metadata["source_url"].startswith("https://storage.googleapis.com/mediapipe-models/")
    assert len(metadata["samples"]) >= 4
    assert all(sample["max_abs_error"] <= 0.002 for sample in metadata["samples"])
    assert all(sample["argmax_agreement"] >= 0.999 for sample in metadata["samples"])


@pytest.mark.parametrize("failure", ["weights", "license", "provenance"])
def test_packaging_rejects_modified_or_incomplete_skin_model(assets, tmp_path, failure):
    folder = tmp_path / "models"
    shutil.copytree(ROOT / "src/freedive_color/assets/models", folder)
    if failure == "weights":
        (folder / "selfie_multiclass_256x256.onnx").write_bytes(b"modified weights")
    elif failure == "license":
        (folder / "SELFIE_MULTICLASS_LICENSE.txt").unlink()
    else:
        metadata = folder / "SELFIE_MULTICLASS_METADATA.json"
        values = json.loads(metadata.read_text(encoding="utf-8"))
        values["source_sha256"] = "0" * 64
        metadata.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or modified|provenance/license"):
        assets.verify_models(folder)


def test_conversion_refuses_unverified_existing_source_without_downloading(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    converter = importlib.import_module("tools.convert_skin_model")
    source = tmp_path / "unverified.tflite"
    source.write_bytes(b"unverified")

    def unexpected_network(*args, **kwargs):
        pytest.fail("A source hash failure must not start a hidden download")

    monkeypatch.setattr(converter.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        converter.acquire_source(source)
