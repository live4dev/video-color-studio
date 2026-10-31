import json

import numpy as np
import pytest
from PIL import Image

from freedive_color.core.masks import MaskSet
from freedive_color.core.project import sha256_file
from tools import verify_skin_photo as verifier


def fixture(tmp_path):
    source = tmp_path / "fixture.png"
    Image.new("RGB", (30, 20), (10, 90, 110)).save(source)
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "source_sha256": sha256_file(source),
                "source_size": [30, 20],
                "points": {"photo": {"skin_arm": [5, 5, 2], "dress": [20, 10, 2]}},
            }
        )
    )
    return source, samples


def test_photo_audit_reports_scalar_failures_without_changing_source(tmp_path, monkeypatch):
    source, samples = fixture(tmp_path)
    before = source.read_bytes()
    alpha = np.zeros((20, 30), np.float32)
    alpha[1:10, 1:10] = 0.9
    ones = np.ones_like(alpha)
    monkeypatch.setattr(
        verifier, "build_masks", lambda _rgb: MaskSet(ones, alpha, ones - alpha, ones, 0.9, 0.9)
    )
    report = verifier.evaluate_photo(source, samples)
    assert report["acceptance_passed"]
    assert report["source_unchanged"] and source.read_bytes() == before
    assert report["regions"]["skin_arm"]["mean_alpha"] == pytest.approx(0.9)
    assert report["regions"]["dress"]["mean_alpha"] == 0
    json.dumps(report, allow_nan=False)
    alpha[10, 20] = 1
    assert verifier.evaluate_photo(source, samples)["failed_regions"] == ["dress"]


@pytest.mark.parametrize(
    "field,value,message",
    [("source_sha256", "wrong", "SHA"), ("source_size", [20, 30], "geometry")],
)
def test_photo_audit_checks_immutable_input_pins(tmp_path, field, value, message):
    source, samples = fixture(tmp_path)
    data = json.loads(samples.read_text())
    data[field] = value
    samples.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        verifier.evaluate_photo(source, samples)


def test_photo_samples_reject_additional_frames(tmp_path):
    source, samples = fixture(tmp_path)
    data = json.loads(samples.read_text())
    data["points"]["1"] = {"skin_leg": [10, 10, 2]}
    samples.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly"):
        verifier.evaluate_photo(source, samples)
