from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from freedive_color.core.color_pipeline import LUMA
from freedive_color.core.errors import StudioError
from freedive_color.core.local_pipeline import LocalSettings, apply_local_grade
from freedive_color.core.masks import (
    MODEL_PATH,
    BrushStroke,
    MaskEdits,
    PPHumanSegmenter,
    build_masks,
)
from freedive_color.core.media_io import load_photo


@pytest.fixture
def scene():
    rgb = np.full((160, 240, 3), [0.08, 0.4, 0.65], np.float32)
    person = np.zeros(rgb.shape[:2], np.float32)
    person[20:152, 60:185] = 0.99
    rgb[40:70, 90:120] = [0.68, 0.50, 0.4]  # light skin target
    rgb[80:110, 90:120] = [0.25, 0.18, 0.13]  # dark skin target
    rgb[30:65, 135:170] = [0.85, 0.08, 0.55]  # saturated swimsuit
    rgb[100:130, 140:172] = [0.02, 0.02, 0.02]  # black mask
    rgb[125:150, 80:120] = [0.03, 0.12, 0.38]  # blue fin
    return rgb, person


def test_mask_and_control_serialization():
    edits = MaskEdits(
        strokes=(BrushStroke("skin", "add", ((0.3, 0.4), (0.5, 0.6)), 0.04),),
        waterline=((0, 0.4), (0.5, 0.45), (1, 0.5)),
        person_box=(0.1, 0.2, 0.7, 0.9),
    )
    assert MaskEdits.from_dict(edits.to_dict()) == edits
    settings = LocalSettings(skin_strength=0.7, water_hue=-0.25)
    assert LocalSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BrushStroke(points=((float("nan"), 0.1),)),
        lambda: BrushStroke(points=((-0.1, 0.4),)),
        lambda: BrushStroke(points=((0.1, 0.4),), radius=1),
        lambda: MaskEdits(waterline=((0.8, 0.2), (0.4, 0.2))),
        lambda: MaskEdits(person_box=(0.5, 0.1, 0.3, 0.4)),
        lambda: LocalSettings(skin_strength=float("nan")),
        lambda: LocalSettings(water_hue=2),
        lambda: LocalSettings(enabled="yes"),
        lambda: MaskEdits.from_dict({"execute": "bad"}),
        lambda: LocalSettings.from_dict({"execute": "bad"}),
    ],
)
def test_invalid_local_settings_and_edits_rejected(factory):
    with pytest.raises(StudioError):
        factory()


def test_skin_is_distinct_from_person_and_excludes_equipment(scene):
    rgb, person_probability = scene
    masks = build_masks(rgb, person_probability=person_probability)
    assert masks.person[50, 150] > 0.95
    assert masks.skin[55, 105] > 0.7
    assert masks.skin[95, 105] > 0.7
    assert masks.skin[45, 150] == 0  # magenta swimsuit
    assert masks.skin[115, 155] == 0  # black mask
    assert masks.skin[138, 100] == 0  # fin
    assert masks.water[138, 100] < 0.01  # person silhouette protects blue fin from water controls
    assert masks.water[10, 20] > 0.9
    assert masks.skin_confidence >= 0.8
    assert masks.person_confidence > masks.skin_confidence


def test_no_trustworthy_person_disables_auto_skin(scene):
    rgb, _person = scene
    masks = build_masks(rgb, use_model=False)
    assert masks.skin_confidence == 0
    assert masks.skin.max() == 0
    assert masks.warnings


def test_local_skin_edit_preserves_water_and_clothing_and_luminance(scene):
    rgb, probability = scene
    original = rgb.copy()
    masks = build_masks(rgb, person_probability=probability)
    result = apply_local_grade(
        rgb, rgb, LocalSettings(skin_strength=1, skin_warmth=0.8), masks=masks
    )
    assert result.rgb[55, 105, 0] > rgb[55, 105, 0]
    assert result.rgb[95, 105, 0] > rgb[95, 105, 0]
    for y, x in ((45, 150), (115, 155), (138, 100), (10, 20)):
        np.testing.assert_array_equal(result.rgb[y, x], rgb[y, x])
    np.testing.assert_allclose(result.rgb @ LUMA, rgb @ LUMA, atol=3e-7)
    np.testing.assert_array_equal(rgb, original)
    assert result.rgb.shape == rgb.shape


def test_water_controls_preserve_person_and_existing_gradient(scene):
    rgb, probability = scene
    rgb[:, :40] *= np.linspace(0.5, 1, 160, dtype=np.float32)[:, None, None]
    masks = build_masks(rgb, person_probability=probability)
    result = apply_local_grade(
        rgb, rgb, LocalSettings(skin_strength=0, water_hue=0.8, water_saturation=0.4), masks=masks
    )
    assert np.max(np.abs(result.rgb[10, 20] - rgb[10, 20])) > 0.005
    np.testing.assert_array_equal(result.rgb[138, 100], rgb[138, 100])
    assert result.rgb[140, 20].mean() > result.rgb[20, 20].mean()
    np.testing.assert_allclose(result.rgb @ LUMA, rgb @ LUMA, atol=5e-7)


def test_manual_skin_brush_works_without_a_model_and_subtraction_is_respected(scene):
    rgb, _probability = scene
    strokes = (
        BrushStroke("skin", "add", ((0.5, 0.5),), 0.15),
        BrushStroke("skin", "subtract", ((0.5, 0.5),), 0.035),
    )
    masks = build_masks(rgb, MaskEdits(strokes=strokes), use_model=False)
    assert masks.skin_confidence == 1
    assert masks.skin[80, 120] < 0.02
    assert masks.skin[80, 135] > 0.9
    assert np.all(masks.skin <= masks.person)
    assert "manual-skin" in masks.method


def test_strokes_scale_with_image_geometry():
    edits = MaskEdits(strokes=(BrushStroke("skin", "add", ((0.2, 0.3), (0.7, 0.65)), 0.05),))
    small = np.full((128, 192, 3), 0.3, np.float32)
    large = np.full((512, 768, 3), 0.3, np.float32)
    a = build_masks(small, edits, use_model=False).skin
    b = build_masks(large, edits, use_model=False).skin
    reduced = cv2.resize(b, (192, 128), interpolation=cv2.INTER_AREA)
    assert np.mean(np.abs(a - reduced)) < 0.02


def test_curved_waterline_preserves_sky_and_applies_underwater_grade():
    original = np.full((120, 200, 3), [0.2, 0.5, 0.7], np.float32)
    graded = original * [1.15, 0.95, 1.0]
    graded = graded.astype(np.float32)
    edits = MaskEdits(waterline=((0, 0.35), (0.5, 0.6), (1, 0.4)))
    masks = build_masks(original, edits, use_model=False)
    result = apply_local_grade(original, graded, LocalSettings(skin_strength=0), edits, masks)
    np.testing.assert_array_equal(result.rgb[:25], original[:25])
    np.testing.assert_array_equal(result.rgb[95:], graded[95:])
    assert masks.underwater[55, 5] > 0.95
    assert masks.underwater[55, 100] == 0
    # Upper and lower pixels stay in place; only their colors are blended.
    assert result.rgb.shape == original.shape


def test_above_water_exposure_is_independent():
    original = np.full((100, 100, 3), 0.3, np.float32)
    edits = MaskEdits(waterline=((0, 0.5), (1, 0.5)))
    masks = build_masks(original, edits, use_model=False)
    settings = LocalSettings(skin_strength=0, above_exposure=1)
    result = apply_local_grade(original, original, settings, edits, masks)
    assert result.rgb[10, 10, 0] > 0.4
    np.testing.assert_array_equal(result.rgb[90, 10], original[90, 10])


def test_waterline_protects_above_water_skin_even_with_manual_brush_and_reference():
    from freedive_color.core.reference import PaletteReference

    original = np.full((120, 160, 3), [0.68, 0.5, 0.4], np.float32)
    edits = MaskEdits(
        strokes=(
            BrushStroke("skin", "add", ((0.5, 0.25),), 0.12),
            BrushStroke("skin", "add", ((0.5, 0.75),), 0.12),
        ),
        waterline=((0, 0.5), (1, 0.5)),
    )
    masks = build_masks(original, edits, use_model=False)
    settings = LocalSettings(skin_strength=1, skin_warmth=1, reference_strength=1)
    result = apply_local_grade(
        original, original, settings, edits, masks, PaletteReference(skin_ab=(30, 40))
    )
    np.testing.assert_array_equal(result.rgb[:40], original[:40])
    assert result.masks.skin[:40].max() == 0
    assert result.rgb[90, 80, 0] > original[90, 80, 0]


def test_zero_local_controls_and_disabled_are_identity(scene):
    rgb, probability = scene
    masks = build_masks(rgb, person_probability=probability)
    zero = LocalSettings(skin_strength=0, water_strength=0)
    np.testing.assert_array_equal(apply_local_grade(rgb, rgb, zero, masks=masks).rgb, rgb)
    disabled = replace(zero, enabled=False, water_hue=1)
    np.testing.assert_array_equal(apply_local_grade(rgb, rgb, disabled, masks=masks).rgb, rgb)


def test_metadata_contains_stats_not_raster_arrays(scene):
    rgb, probability = scene
    metadata = build_masks(rgb, person_probability=probability).to_metadata()
    assert metadata["coverage"]["skin"] < metadata["coverage"]["person"]
    assert not any(isinstance(value, np.ndarray) for value in metadata.values())


def test_missing_or_modified_model_is_explicit(tmp_path):
    with pytest.raises(StudioError, match="отсутствует"):
        PPHumanSegmenter(tmp_path / "missing.onnx")
    fake = tmp_path / "wrong.onnx"
    fake.write_bytes(b"not the licensed model")
    with pytest.raises(StudioError, match="сумма"):
        PPHumanSegmenter(fake)


def test_unicode_model_path_uses_verified_memory_buffer(tmp_path, monkeypatch):
    folder = tmp_path / "Сборка и модель"
    folder.mkdir()
    model = folder / MODEL_PATH.name
    model.write_bytes(MODEL_PATH.read_bytes())
    original = cv2.dnn.readNetFromONNX
    observed = []

    def load(buffer):
        assert isinstance(buffer, np.ndarray) and buffer.dtype == np.uint8
        observed.append(buffer.size)
        return original(buffer)

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    segmenter = PPHumanSegmenter(model)
    assert not segmenter.net.empty()
    assert observed == [model.stat().st_size]


def test_real_photo_semantic_masks_are_local_and_do_not_select_black_gear():
    assert MODEL_PATH.is_file()
    source = Path(__file__).parents[1] / "fixtures" / "freediver_moalboal.jpg"
    photo = load_photo(source, max_edge=750)
    masks = build_masks(photo.rgb)
    assert masks.method == "pphumanseg-opencv-cpu"
    assert masks.person.shape == photo.rgb.shape[:2]
    assert 0.005 < masks.person.mean() < 0.5
    assert masks.skin.mean() < masks.person.mean()
    # Black gear cannot enter automatic skin correction, including edge-feathered alpha.
    assert masks.skin[photo.rgb.max(axis=2) < 0.07].max(initial=0) == 0
    assert masks.skin[(photo.rgb[..., 2] > photo.rgb[..., 0] * 2)].max(initial=0) == 0
    # The blue right arm is part of the person even though it has a water-like hue.
    # This avoids changing the wetsuit with the water hue control.
    y, x = round(0.64 * photo.rgb.shape[0]), round(0.46 * photo.rgb.shape[1])
    assert photo.rgb[y, x, 2] > photo.rgb[y, x, 0] * 1.5
    assert masks.person[y, x] > 0.9
    assert masks.water[y, x] < 0.05
