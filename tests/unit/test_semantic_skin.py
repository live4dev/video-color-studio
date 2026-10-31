from threading import Lock

import cv2
import numpy as np
import pytest

from freedive_color.core.errors import StudioError
from freedive_color.core.masks import BrushStroke, MaskEdits, build_masks
from freedive_color.core.skin import (
    SKIN_MODEL_ID,
    SelfieSkinSegmenter,
    SkinPrediction,
    _regions,
)


def semantic_scene():
    rgb = np.full((120, 180, 3), [0.04, 0.35, 0.5], np.float32)
    person = np.zeros(rgb.shape[:2], np.float32)
    person[10:110, 20:160] = 0.99
    skin = np.zeros_like(person)
    for x, color in (
        (30, [0.07, 0.28, 0.42]),
        (70, [0.06, 0.15, 0.08]),
        (110, [0.035, 0.06, 0.05]),
    ):
        rgb[20:50, x : x + 25] = color  # blue, green, dark visible skin
        skin[20:50, x : x + 25] = 0.96
    rgb[65:95, 30:55] = [0.25, 0.17, 0.11]  # hair shares warm colors with skin
    rgb[65:95, 70:95] = [0.75, 0.3, 0.16]  # belt
    rgb[65:95, 110:135] = [0.12, 0.14, 0.13]  # diving mask
    exclusion = (person > 0).astype(np.float32) * (skin == 0) * 0.99
    return rgb, person, SkinPrediction(skin, person, exclusion, "a" * 64)


def test_semantic_blue_green_dark_skin_and_warm_exclusions():
    rgb, person, prediction = semantic_scene()
    original = rgb.copy()
    masks = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    for x in (40, 80, 120):
        assert masks.skin[35, x] > 0.9
        assert masks.skin[80, x] == 0
        assert masks.skin_exclusion[80, x] > 0.9
    assert masks.skin_status == "automatic"
    assert masks.skin_source == SKIN_MODEL_ID
    assert masks.skin_confidence == pytest.approx(0.96)
    assert masks.skin_confidence != masks.person_confidence
    np.testing.assert_array_equal(rgb, original)


def test_high_person_probability_cannot_make_uncertain_skin_confident():
    rgb, person, prediction = semantic_scene()
    prediction.skin[:] = 0.7
    prediction.exclusion[:] = 0
    masks = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    assert masks.person_confidence > 0.95
    assert masks.skin_status == "needs-review"
    assert masks.skin_confidence == pytest.approx(0.7)
    assert masks.skin.max() == 0
    assert masks.warnings


def test_overly_broad_semantic_skin_abstains_even_with_high_raw_score():
    rgb = np.full((80, 100, 3), [0.05, 0.3, 0.6], np.float32)
    probability = np.full(rgb.shape[:2], 0.99, np.float32)
    prediction = SkinPrediction(probability, probability, np.zeros_like(probability))
    masks = build_masks(rgb, use_model=False, skin_prediction=prediction)
    assert masks.skin_confidence > 0.98
    assert masks.skin_status == "needs-review"
    assert masks.skin.max() == 0
    assert masks.skin_diagnostics["selected_fraction"] == 1
    assert masks.warnings


def test_ambiguous_accessory_boundary_is_excluded_without_losing_clear_skin():
    rgb, person, prediction = semantic_scene()
    prediction.skin[20:50, 30:55] = 0.74
    prediction.exclusion[20:50, 30:55] = 0.23
    masks = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    assert masks.skin[35, 40] == 0
    assert masks.skin[35, 80] > 0.9
    assert masks.skin_status == "automatic"
    assert masks.skin_diagnostics["ambiguous_pixels"] >= 750
    assert any("Неоднозначные" in value for value in masks.warnings)


def test_semantic_skin_recovers_limb_outside_coarse_person_hint():
    rgb, person, prediction = semantic_scene()
    coarse = person.copy()
    coarse[20:50, 30:55] = 0
    masks = build_masks(rgb, person_probability=coarse, skin_prediction=prediction)
    assert masks.skin[35, 40] > 0.9
    assert masks.person[35, 40] > 0.9
    assert masks.water[35, 40] < 0.01


def test_explicit_person_box_and_subtraction_are_not_overwritten_by_skin_model():
    rgb, person, prediction = semantic_scene()
    edits = MaskEdits(
        person_box=(0.3, 0.1, 0.9, 0.9),
        strokes=(BrushStroke("person", "subtract", ((0.45, 0.3),), 0.1),),
    )
    masks = build_masks(rgb, edits, person_probability=person, skin_prediction=prediction)
    assert masks.skin[35, 40] == 0  # outside requested box
    assert masks.skin[35, 80] == 0  # explicit person subtraction
    assert masks.skin[35, 120] > 0.9


def test_model_unavailable_does_not_fall_back_to_warm_rgb_rules(monkeypatch):
    rgb, person, _prediction = semantic_scene()

    def unavailable(*_args):
        raise StudioError("Локальная модель кожи отсутствует.")

    monkeypatch.setattr("freedive_color.core.masks.infer_skin", unavailable)
    masks = build_masks(rgb, person_probability=person)
    assert masks.skin.max() == 0
    assert masks.skin_status == "needs-review"
    assert masks.skin_source == "none"
    assert any("отсутствует" in warning for warning in masks.warnings)


def test_manual_brush_overrides_semantic_exclusion_and_subtraction():
    rgb, person, prediction = semantic_scene()
    edits = MaskEdits(
        strokes=(
            BrushStroke("skin", "add", ((0.46, 0.67),), 0.09),
            BrushStroke("skin", "subtract", ((0.46, 0.67),), 0.04),
        )
    )
    masks = build_masks(rgb, edits, person_probability=person, skin_prediction=prediction)
    assert masks.skin_status == "manual"
    assert masks.skin_source == "manual"
    assert masks.skin[80, 83] < 0.1
    assert masks.skin[80, 90] > 0.8


def test_subtract_only_skin_edit_is_reported_as_manual():
    rgb, person, prediction = semantic_scene()
    edits = MaskEdits(strokes=(BrushStroke("skin", "subtract", ((0.22, 0.3),), 0.1),))
    masks = build_masks(rgb, edits, person_probability=person, skin_prediction=prediction)
    assert masks.skin_status == "manual"
    assert masks.skin_source == "manual"
    assert masks.skin_confidence == pytest.approx(0.96)


@pytest.mark.parametrize("bad", ["shape", "nan", "range"])
def test_invalid_semantic_prediction_rejected(bad):
    rgb, person, prediction = semantic_scene()
    skin = prediction.skin.copy()
    if bad == "shape":
        skin = skin[:10]
    elif bad == "nan":
        skin[0, 0] = np.nan
    else:
        skin[0, 0] = 2
    with pytest.raises(StudioError, match="семантическая"):
        build_masks(
            rgb,
            person_probability=person,
            skin_prediction=SkinPrediction(skin, person, prediction.exclusion),
        )


def test_no_network_and_provenance_metadata_contains_no_rasters(monkeypatch):
    rgb, person, prediction = semantic_scene()
    import socket

    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("network"))
    metadata = build_masks(rgb, person_probability=person, skin_prediction=prediction).to_metadata()
    assert metadata["skin_model_sha256"] == "a" * 64
    assert metadata["skin_source"] == SKIN_MODEL_ID
    assert not any(isinstance(value, np.ndarray) for value in metadata.values())


def test_missing_or_changed_skin_model_is_explicit(tmp_path):
    with pytest.raises(StudioError, match="отсутствует"):
        SelfieSkinSegmenter(tmp_path / "missing.onnx")
    modified = tmp_path / "changed.onnx"
    modified.write_bytes(b"modified")
    with pytest.raises(StudioError, match="сумма"):
        SelfieSkinSegmenter(modified)


def test_model_input_uses_verified_minus_one_to_one_normalization():
    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.input_name, segmenter.output_name = "input", "output"
    segmenter._runtime_errors = (RuntimeError,)
    seen = []

    class Session:
        def run(self, _names, inputs):
            seen.append(inputs["input"])
            return [np.zeros((1, 256, 256, 6), np.float32)]

    segmenter.session = Session()
    rgb = np.full((40, 60, 3), [0, 0.5, 1], np.float32)
    result = segmenter._predict(rgb)
    np.testing.assert_array_equal(seen[0][0, 128, 128], [-1, 0, 1])
    np.testing.assert_array_equal(seen[0][0, 0, 0], [-1, -1, -1])
    assert seen[0].dtype == np.float32 and seen[0].flags.c_contiguous
    assert result.shape == (*rgb.shape[:2], 6)
    np.testing.assert_allclose(result.sum(axis=2), 1, atol=2e-7)


def test_native_runtime_failure_is_reported_as_recoverable():
    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.input_name, segmenter.output_name = "input", "output"

    class NativeFailure(Exception):
        pass

    class Session:
        def run(self, *_args):
            raise NativeFailure("Native inference unavailable")

    segmenter.session = Session()
    segmenter._runtime_errors = (NativeFailure,)
    with pytest.raises(StudioError, match="локальную модель"):
        segmenter._predict(np.zeros((40, 60, 3), np.float32))


def test_person_crops_are_bounded_and_include_context():
    hint = np.zeros((200, 300), np.float32)
    hint[50:100, 100:130] = 0.99
    boxes = _regions(hint)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 < 100 and y0 < 50 and x1 > 130 and y1 > 100
    assert _regions(np.zeros_like(hint)) == []
    assert _regions(np.ones_like(hint)) == []


def test_letterbox_probability_map_returns_to_narrow_source_geometry():
    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.input_name, segmenter.output_name = "input", "output"
    segmenter._runtime_errors = (RuntimeError,)

    class Session:
        def run(self, _names, inputs):
            marker = (inputs["input"][..., 0] + 1) / 2
            result = np.zeros((1, 256, 256, 6), np.float32)
            result[..., 0], result[..., 2] = 1 - marker, marker
            return [result]

    segmenter.session = Session()
    rgb = np.zeros((200, 40, 3), np.float32)
    rgb[80:120, 10:30, 0] = 1
    result = segmenter._predict(rgb)
    assert result.shape == (200, 40, 6)
    assert result[100, 20, 2] > 0.99
    assert result[30, 20, 2] == 0
    assert result[100, 2, 2] == 0


@pytest.mark.parametrize("best_rotation", [0, 1, 2, 3])
def test_four_orientations_are_inverted_to_source_coordinates(best_rotation):
    # This fake model recognizes a spatial marker; it tests rotation/fusion only.
    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.lock = Lock()
    seen = []

    def predict(rgb):
        seen.append(rgb.shape)
        marker = cv2.resize(rgb[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
        if len(seen) - 1 != best_rotation:
            marker *= 0
        output = np.zeros((256, 256, 6), np.float32)
        output[..., 0] = 1 - marker
        output[..., 2] = marker
        return output

    segmenter._predict = predict
    rgb = np.zeros((80, 120, 3), np.float32)
    rgb[20:55, 25:55, 0] = 1
    result = segmenter.infer(rgb)
    assert len(seen) == 4
    assert result.skin[30, 35] > 0.99
    assert result.skin[65, 90] == 0
    assert result.diagnostics["inference_calls"] == 4
    assert result.diagnostics["selected_quarter_turns"] == [best_rotation]
