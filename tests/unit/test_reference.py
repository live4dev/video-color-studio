from dataclasses import replace

import numpy as np
import pytest

from freedive_color.core.color_pipeline import LUMA
from freedive_color.core.errors import StudioError
from freedive_color.core.masks import MaskSet
from freedive_color.core.reference import PaletteReference, create_reference, match_reference


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
