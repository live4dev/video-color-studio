from dataclasses import replace

import numpy as np
import pytest

from freedive_color.core.color_pipeline import LUMA
from freedive_color.core.errors import StudioError
from freedive_color.core.masks import MaskSet
from freedive_color.core.reference import (
    PaletteReference,
    ReferenceTransform,
    create_reference,
    freeze_reference,
    match_reference,
)


def water_masks(shape):
    zeros = np.zeros(shape, np.float32)
    ones = np.ones(shape, np.float32)
    return MaskSet(zeros, zeros, ones, ones, 0, 0)


def test_reference_serialization_and_validation():
    reference = PaletteReference(water_ab=(-15, -20), skin_ab=(8, 11), source_name="Dive")
    assert PaletteReference.from_dict(reference.to_dict()) == reference
    with pytest.raises(StudioError):
        replace(reference, water_ab=(float("nan"), 3))
    with pytest.raises(StudioError):
        PaletteReference.from_dict({"execute": "bad"})


def test_reference_matching_changes_chroma_without_copying_exposure():
    x = np.linspace(0.5, 1.0, 180, dtype=np.float32)[None, :, None]
    scene = np.repeat(x * np.array([0.08, 0.4, 0.65], np.float32), 100, axis=0)
    target = np.full(scene.shape, [0.15, 0.5, 0.64], np.float32)
    masks = water_masks(scene.shape[:2])
    reference = create_reference(target, masks)
    assert reference.water_ab is not None and reference.skin_ab is None
    assert reference.warnings
    matched = match_reference(scene, masks, reference, strength=0.7)
    assert np.mean(np.abs(matched - scene)) > 0.001
    np.testing.assert_allclose(matched @ LUMA, scene @ LUMA, atol=4e-7)
    assert np.all(np.diff((matched @ LUMA)[50]) > 0)


def test_reference_adjustment_is_bounded_and_zero_strength_is_identity():
    rgb = np.full((100, 120, 3), [0.1, 0.4, 0.65], np.float32)
    masks = water_masks(rgb.shape[:2])
    extreme = PaletteReference(water_ab=(120, 120))
    matched = match_reference(rgb, masks, extreme, strength=1)
    assert np.max(np.abs(matched - rgb)) < 0.15
    np.testing.assert_array_equal(match_reference(rgb, masks, extreme, 0), rgb)


def test_absent_mask_does_not_copy_another_region_palette():
    rgb = np.full((70, 100, 3), [0.08, 0.4, 0.65], np.float32)
    masks = water_masks(rgb.shape[:2])
    skin_only = PaletteReference(skin_ab=(30, 30))
    np.testing.assert_array_equal(match_reference(rgb, masks, skin_only, 1), rgb)


def test_frozen_reference_matches_photo_path_and_applies_strength_once():
    rgb = np.full((80, 120, 3), [0.2, 0.45, 0.65], np.float32)
    masks = water_masks(rgb.shape[:2])
    target = create_reference(np.full_like(rgb, [0.16, 0.5, 0.6]), masks)
    frozen = freeze_reference(rgb, masks, target)
    assert ReferenceTransform.from_dict(frozen.to_dict()) == frozen
    np.testing.assert_array_equal(
        match_reference(rgb, masks, target, 0.45), match_reference(rgb, masks, frozen, 0.45)
    )
    full = match_reference(rgb, masks, frozen, 1)
    half = match_reference(rgb, masks, frozen, 0.5)
    np.testing.assert_allclose(half, rgb * 0.5 + full * 0.5, atol=1e-7)
    assert set(frozen.to_dict()) == {
        "water_delta_ab",
        "skin_delta_ab",
        "source_name",
        "source_sha256",
        "warnings",
        "algorithm",
    }


def test_frozen_video_reference_does_not_reanalyze_frame_chroma_or_mask_quantization(monkeypatch):
    import freedive_color.core.reference as module

    first = np.full((80, 120, 3), [0.2, 0.45, 0.65], np.float32)
    masks = water_masks(first.shape[:2])
    reference = create_reference(np.full_like(first, [0.16, 0.5, 0.6]), masks)
    frozen = freeze_reference(first, masks, reference)
    a = match_reference(first, masks, frozen, 0.8)
    second = np.full_like(first, [0.3, 0.6, 0.7])
    second[35:45, 55:65] = first[35:45, 55:65]
    later_masks = water_masks(first.shape[:2])
    later_masks.water[:, :40] = 0.54
    later_masks.water[:, 80:] = 0.56

    def cannot_sample(*_args):
        raise AssertionError("Frozen video transform reanalyzed per-frame statistics")

    monkeypatch.setattr(module, "_sample_lab", cannot_sample)
    b = match_reference(second, later_masks, frozen, 0.8)
    np.testing.assert_array_equal(a[40, 60], b[40, 60])
    np.testing.assert_allclose(b @ LUMA, second @ LUMA, atol=5e-7)


@pytest.mark.parametrize(
    "settings",
    [
        {"water_delta_ab": (7, 0)},
        {"skin_delta_ab": (0, 5)},
        {"water_delta_ab": (float("nan"), 0)},
        {"exposure": 1},
        {"algorithm": "unrecognized"},
    ],
)
def test_invalid_frozen_reference_is_rejected(settings):
    with pytest.raises(StudioError):
        ReferenceTransform.from_dict(settings)
