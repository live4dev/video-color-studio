from dataclasses import replace

import numpy as np
import pytest

from freedive_color.core.analyze import analyze_scene, linear_to_srgb, srgb_to_linear
from freedive_color.core.color_pipeline import grade_image
from freedive_color.core.errors import StudioError
from freedive_color.core.presets import PRESETS, RANGES, GradeSettings


def neutral_settings():
    return GradeSettings(**dict.fromkeys(RANGES, 0))


def test_zero_controls_are_identity_and_input_immutable(rgb):
    original = rgb.copy()
    result = grade_image(rgb, neutral_settings())
    np.testing.assert_allclose(result, rgb, atol=3e-7)
    np.testing.assert_array_equal(original, rgb)
    assert not np.shares_memory(result, rgb)


def test_srgb_roundtrip():
    ramp = np.linspace(0, 1, 65536, dtype=np.float32)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(ramp)), ramp, atol=2e-7)


@pytest.mark.parametrize("name", PRESETS)
def test_presets_finite_deterministic_preserve_dimensions(rgb, name):
    result = grade_image(rgb, PRESETS[name])
    assert result.shape == rgb.shape and result.dtype == np.float32
    assert np.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1
    np.testing.assert_array_equal(result, grade_image(rgb, PRESETS[name]))


def test_presets_are_distinguishable(rgb):
    outputs = [grade_image(rgb, setting) for setting in PRESETS.values()]
    assert (
        min(np.mean(np.abs(outputs[a] - outputs[b])) for a, b in ((0, 1), (1, 2), (0, 2))) > 0.002
    )


def test_missing_red_is_not_fabricated():
    water = np.full((50, 80, 3), [0, 0.35, 0.55], dtype=np.float32)
    analysis = analyze_scene(water)
    assert analysis.red_gain == 1 and analysis.red_confidence == 0
    assert analysis.warnings
    corrected = grade_image(water, replace(neutral_settings(), auto_strength=1, red_recovery=1))
    assert corrected[..., 0].max() < 1e-6
    assert corrected[..., 2].mean() > corrected[..., 1].mean()


@pytest.mark.parametrize("color", ([0, 0, 0], [1, 1, 1], [0, 0.5, 0.7]))
def test_constant_water_has_no_spatial_artifacts(color):
    image = np.full((100, 140, 3), color, dtype=np.float32)
    out = grade_image(image, PRESETS["vivid"])
    assert np.max(out.max(axis=(0, 1)) - out.min(axis=(0, 1))) < 1e-6


def test_no_geometry_shift():
    board = np.full((100, 140, 3), 0.2, np.float32)
    board[25:70, 40:100] = [0.45, 0.55, 0.6]
    result = grade_image(board, neutral_settings())
    np.testing.assert_array_equal(result[..., 1] > 0.4, board[..., 1] > 0.4)


@pytest.mark.parametrize(
    "key,value",
    [
        ("exposure", float("nan")),
        ("detail", 2),
        ("auto_strength", True),
        ("temperature", float("inf")),
    ],
)
def test_invalid_controls_rejected(key, value):
    with pytest.raises(StudioError):
        replace(PRESETS["natural"], **{key: value})


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 2, 3), np.float32),
        np.zeros((5, 5), np.float32),
        np.full((5, 5, 3), np.nan, np.float32),
        np.full((5, 5, 3), 2, np.float32),
    ],
)
def test_bad_pixels_rejected(value):
    with pytest.raises(StudioError):
        grade_image(value, PRESETS["natural"])
