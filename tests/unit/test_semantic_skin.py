from threading import Lock

import cv2
import numpy as np
import pytest

from freedive_color.core.errors import StudioError
from freedive_color.core.masks import BrushStroke, MaskEdits, build_masks
from freedive_color.core.skin import (
    SKIN_CLASS_NAMES,
    SKIN_MODEL_ID,
    RawPredictionCache,
    SelfieSkinSegmenter,
    SkinPrediction,
    SkinView,
    _regions,
    fuse_skin_views,
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
    assert masks.skin_status == "needs-review"
    assert masks.skin_diagnostics["automatic_accepted"]
    assert masks.skin_diagnostics["accepted_components"] == 2
    assert masks.skin_diagnostics["rejected_components"] >= 1
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
        if len(seen) - 1 not in {best_rotation, (best_rotation + 1) % 4}:
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
    assert result.consensus[30, 35] == 1
    assert [v.quarter_turns for v in result.views] == [0, 1, 2, 3]
    assert set(result.class_maps) == set(SKIN_CLASS_NAMES)


def probabilities(skin=0.0, clothes=0.0, shape=(30, 40)):
    result = np.zeros((*shape, 6), np.float32)
    result[..., 0] = 1 - skin - clothes
    result[..., 2] = skin
    result[..., 4] = clothes
    return result


def view(p, rotation, *, source="full-frame", region=None):
    return SkinView(p, region or (0, 0, p.shape[1], p.shape[0]), rotation, source=source)


def test_one_skin_orientation_does_not_pass_without_geometric_corroboration():
    prediction = fuse_skin_views(
        [
            view(probabilities(skin=0.96), 0),
            view(probabilities(), 1),
            view(probabilities(), 2),
            view(probabilities(), 3),
        ],
        (30, 40),
    )
    assert not prediction.consensus.any()
    assert prediction.diagnostics["unconfirmed_pixels"] == 1200


def test_two_supporting_orientations_recover_skin_despite_background_views():
    prediction = fuse_skin_views(
        [
            view(probabilities(skin=0.96), 0),
            view(probabilities(skin=0.94), 1),
            view(probabilities(), 2),
            view(probabilities(), 3),
        ],
        (30, 40),
    )
    assert prediction.consensus.all()
    np.testing.assert_allclose(prediction.skin, 0.95)
    assert prediction.person.min() > 0.9
    assert prediction.class_maps["body_skin"].mean() == pytest.approx(0.95)


def test_duplicate_geometry_cannot_count_as_two_independent_observations():
    p = probabilities(skin=0.96)
    prediction = fuse_skin_views([view(p, 0), view(p.copy(), 0)], (30, 40))
    assert not prediction.consensus.any()
    assert prediction.diagnostics["view_count"] == 1


def test_conflicting_skin_and_clothing_views_abstain_instead_of_max_union():
    prediction = fuse_skin_views(
        [
            view(probabilities(skin=0.96), 0),
            view(probabilities(skin=0.94), 1),
            view(probabilities(clothes=0.96), 2),
            view(probabilities(clothes=0.96), 3),
        ],
        (30, 40),
    )
    assert not prediction.consensus.any()
    assert prediction.exclusion.min() == 1
    assert prediction.diagnostics["conflicting_pixels"] == 1200


def test_worse_crop_cannot_overwrite_local_full_frame_clothing_consensus():
    views = [view(probabilities(clothes=0.96), rotation) for rotation in range(4)]
    crop = probabilities(skin=0.99, shape=(20, 20))
    views += [
        view(crop.copy(), rotation, source="person-crop", region=(10, 5, 30, 25))
        for rotation in range(4)
    ]
    prediction = fuse_skin_views(views, (30, 40))
    assert not prediction.consensus[15, 20]
    assert prediction.exclusion[15, 20] == 1
    assert prediction.class_maps["clothes"][15, 20] > 0


def test_partial_low_score_component_does_not_erase_confident_components():
    rgb, person, prediction = semantic_scene()
    prediction.skin[20:50, 30:55] = 0.7
    masks = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    assert masks.skin[35, 40] == 0
    assert masks.skin[35, 80] > 0.9
    assert masks.skin_status == "needs-review"
    assert masks.skin_confidence == pytest.approx(0.96)
    assert masks.skin_diagnostics["automatic_accepted"]
    assert masks.skin_diagnostics["accepted_components"] == 2
    assert masks.skin_diagnostics["rejected_components"] == 1
    assert masks.skin_diagnostics["rejection_reasons"]["low_score"] == 1
    assert any("Уверенные участки обрабатываются" in x for x in masks.warnings)
    assert not any("автоматическая коррекция отключена" in x for x in masks.warnings)


def test_unconfirmed_prediction_does_not_pass_mask_builder_even_with_high_average():
    rgb, person, prediction = semantic_scene()
    from dataclasses import replace

    unconfirmed = replace(prediction, consensus=np.zeros_like(person))
    masks = build_masks(rgb, person_probability=person, skin_prediction=unconfirmed)
    assert masks.skin.max() == 0
    assert not masks.skin_diagnostics["automatic_accepted"]


def test_proxy_cache_is_bounded_and_isolated_from_mutable_results():
    p = SkinPrediction(
        np.ones((5, 5), np.float32), np.ones((5, 5), np.float32), np.zeros((5, 5), np.float32)
    )
    cache = RawPredictionCache(max_bytes=601)
    cache.put("one", p)
    cache.put("two", p)
    assert cache.bytes == 600
    first = cache.get("one")
    first.skin[:] = 0
    assert cache.get("one").skin.min() == 1
    p.skin[:] = 0
    assert cache.get("two").skin.min() == 1
    assert cache.get("one") is not None  # one is now most recently used
    cache.put("three", p)
    assert cache.get("two") is None
    assert cache.get("one") is not None
    assert cache.bytes <= 601
    cache.clear()
    assert cache.bytes == 0
    assert cache.get("one") is None


def test_inference_cache_uses_source_and_hint_content_and_does_not_alias():
    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.lock = Lock()
    calls = []

    def predict(rgb):
        calls.append(rgb.shape)
        return probabilities(skin=0.96, shape=rgb.shape[:2])

    segmenter._predict = predict
    rgb = np.full((40, 50, 3), 0.3, np.float32)
    first = segmenter.infer(rgb)
    assert len(calls) == 4
    first.skin[:] = 0
    first.class_maps["body_skin"][:] = 0
    first.views[0].probabilities[:] = 0
    second = segmenter.infer(rgb)
    assert len(calls) == 4 and second.diagnostics["cache_hit"]
    assert second.skin.min() > 0.9
    assert second.class_maps["body_skin"].min() > 0.9
    assert second.views[0].probabilities.max() > 0.9
    changed = rgb.copy()
    changed[0, 0] = 0.31
    segmenter.infer(changed)
    assert len(calls) == 8
    segmenter.infer(rgb, np.full(rgb.shape[:2], 0.1, np.float32))
    assert len(calls) == 12


def test_alternative_model_provenance_survives_masks_without_rasters():
    from dataclasses import replace

    rgb, person, prediction = semantic_scene()
    candidate = replace(
        prediction,
        model_id="candidate",
        algorithm_version="candidate-v2",
        models=({"id": "candidate", "sha256": "a" * 64},),
    )
    masks = build_masks(rgb, person_probability=person, skin_prediction=candidate)
    metadata = masks.to_metadata()
    assert metadata["skin_source"] == "candidate"
    assert metadata["skin_algorithm_version"] == "candidate-v2"
    assert metadata["skin_models"] == [{"id": "candidate", "sha256": "a" * 64}]
    metadata["skin_models"][0]["id"] = "changed"
    assert masks.skin_models[0]["id"] == "candidate"


def test_strong_hair_evidence_is_not_outvoted_by_skin_majority():
    hair = probabilities()
    hair[..., 0] = 0.02
    hair[..., 1] = 0.98
    prediction = fuse_skin_views(
        [view(probabilities(skin=0.98), rotation) for rotation in (0, 1, 2)] + [view(hair, 3)],
        (30, 40),
    )
    assert not prediction.consensus.any()
    assert prediction.exclusion.min() == 1
    assert prediction.diagnostics["conflicting_pixels"] == 1200
    assert prediction.class_maps["hair"].mean() > 0


def test_numpy_crop_coordinates_and_realistic_view_metadata_are_json_safe():
    import json

    segmenter = object.__new__(SelfieSkinSegmenter)
    segmenter.lock = Lock()
    segmenter._predict = lambda rgb: probabilities(skin=0.96, shape=rgb.shape[:2])
    rgb = np.full((80, 120, 3), 0.3, np.float32)
    hint = np.zeros(rgb.shape[:2], np.float32)
    hint[25:45, 40:60] = 0.99
    prediction = segmenter.infer(rgb, hint)
    assert prediction.diagnostics["crop_count"] > 0
    metadata = build_masks(rgb, person_probability=hint, skin_prediction=prediction).to_metadata()
    assert json.loads(json.dumps(metadata))["skin_models"][0]["id"] == SKIN_MODEL_ID
    for source in metadata["skin_diagnostics"]["views"]:
        assert all(type(value) is int for value in source["region"])


def test_small_confirmed_hand_survives_whole_frame_area_floor_with_other_skin():
    rgb = np.full((1000, 1600, 3), [0.05, 0.3, 0.4], np.float32)
    person = np.zeros(rgb.shape[:2], np.float32)
    person[100:500, 100:700] = 0.99
    skin = np.zeros_like(person)
    skin[150:350, 150:350] = 0.95
    skin[400:408, 600:608] = 0.95  # 64 pixels, below the old total-area floor of80
    prediction = SkinPrediction(skin, person, np.zeros_like(person))
    masks = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    assert masks.skin[404, 604] > 0.5  # normal edge feathering remains active
    assert masks.skin_diagnostics["accepted_components"] == 2
    skin[150:350, 150:350] = 0
    alone = build_masks(rgb, person_probability=person, skin_prediction=prediction)
    assert not alone.skin.any()  # the original total-evidence safeguard remains
