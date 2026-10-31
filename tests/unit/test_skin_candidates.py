"""Candidate input/class contracts; no large weights, downloads or desktop side effects."""

import json

import numpy as np
import pytest

from freedive_color.core.errors import StudioError
from tools.skin_candidates import (
    CandidateSegmenter,
    InputGeometry,
    make_predictor,
    map_candidate_logits,
    prepare_candidate_input,
    restore_candidate_logits,
)


@pytest.mark.parametrize(
    "name,mean,std,channel",
    [
        ("fcn-skin", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [0.2, 0.6, 0.8]),
        ("schp-atr", [0.406, 0.456, 0.485], [0.225, 0.224, 0.229], [0.8, 0.6, 0.2]),
    ],
)
def test_original_channel_order_normalization_and_input_immutability(name, mean, std, channel):
    rgb = np.full((97, 183, 3), [0.2, 0.6, 0.8], np.float32)
    before = rgb.copy()
    tensor, geometry = prepare_candidate_input(name, rgb)
    assert tensor.shape == (1, 3, 512, 512) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 256, 256], (np.array(channel) - mean) / std, atol=1e-6)
    np.testing.assert_array_equal(rgb, before)
    assert geometry.shape == (97, 183)


def test_fcn_padding_preserves_narrow_source_edges_instead_of_truncating():
    rgb = np.ones((511, 17, 3), np.float32)
    _tensor, geometry = prepare_candidate_input("fcn-skin", rgb)
    left, top, right, bottom = geometry.content
    assert right - left >= 17 and bottom - top == 512
    logits = np.full((1, 1, 512, 512), -12, np.float32)
    logits[:, :, top:bottom, left:right] = 7
    restored = restore_candidate_logits(logits, geometry)
    assert restored.shape == (511, 17, 1)
    np.testing.assert_allclose(restored, 7)


def test_schp_inverse_affine_retains_all_four_source_corners():
    rgb = np.ones((123, 201, 3), np.float32)
    _tensor, geometry = prepare_candidate_input("schp-atr", rgb)
    assert geometry.inverse_affine is not None
    maps = restore_candidate_logits(np.ones((1, 18, 512, 512), np.float32), geometry)
    assert maps.shape == (123, 201, 18)
    for y, x in [(0, 0), (0, 200), (122, 0), (122, 200)]:
        np.testing.assert_allclose(maps[y, x], 1, atol=1e-6)


@pytest.mark.parametrize(
    "label,target",
    [
        (0, 0),
        (1, 5),
        (2, 1),
        (3, 5),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 4),
        (9, 4),
        (10, 4),
        (11, 3),
        (12, 2),
        (13, 2),
        (14, 2),
        (15, 2),
        (16, 5),
        (17, 4),
    ],
)
def test_atr_all_original_classes_have_exactly_one_semantic_target(label, target):
    logits = np.full((3, 5, 18), -30, np.float32)
    logits[..., label] = 30
    maps = map_candidate_logits("schp-atr", logits)
    assert maps.shape == (3, 5, 6)
    np.testing.assert_allclose(maps.sum(axis=2), 1)
    assert np.all(maps.argmax(axis=2) == target)


def test_binary_skin_uses_sigmoid_and_does_not_invent_hair_or_clothing_classes():
    maps = map_candidate_logits("fcn-skin", np.array([[[-80], [0], [80]]], np.float32))
    np.testing.assert_allclose(maps[0, :, 2], [0, 0.5, 1], atol=1e-7)
    np.testing.assert_allclose(maps.sum(axis=2), 1)
    assert not maps[..., [1, 3, 4, 5]].any()


def test_missing_candidate_fails_closed_without_network(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A runtime candidate attempted a network download")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(StudioError, match="отсутствует"):
        make_predictor("fcn-skin", tmp_path)


def test_unknown_candidate_never_silently_falls_back():
    with pytest.raises(ValueError, match="Unknown candidate"):
        make_predictor("not-a-candidate")


def test_modified_candidate_rejected_before_session_creation(monkeypatch, tmp_path):
    import onnxruntime

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unverified candidate reached ONNX Runtime")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    (tmp_path / "fcn-skin.onnx").write_bytes(b"unverified")
    (tmp_path / "fcn-skin.json").write_text(json.dumps({"onnx_sha256": "unknown"}))
    with pytest.raises(StudioError, match="сумма"):
        CandidateSegmenter("fcn-skin", tmp_path)


@pytest.mark.parametrize(
    "bad", [np.full((1, 1, 512, 512), np.nan, np.float32), np.zeros((1, 512, 512), np.float32)]
)
def test_invalid_logits_are_not_accepted(bad):
    with pytest.raises(StudioError, match="logits"):
        restore_candidate_logits(bad, InputGeometry((3, 4), (0, 0, 512, 512)))
