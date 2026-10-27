"""Controlled color comparisons; these are not semantic skin-mask acceptance tests."""

import numpy as np
import pytest

from freedive_color.core.color_pipeline import grade_image
from freedive_color.core.presets import PRESETS


@pytest.mark.parametrize("name", PRESETS)
def test_water_stays_blue_and_gradient_is_preserved(name):
    gradient = np.linspace(0.1, 0.9, 150, dtype=np.float32)[:, None, None]
    water = np.repeat(gradient * np.array([0.2, 0.7, 1], np.float32), 120, axis=1)
    result = grade_image(water, PRESETS[name])
    assert np.all(result[..., 2] > result[..., 1])
    assert np.all(result[..., 1] > result[..., 0])
    assert np.all(np.diff(result[:, 40, 1]) > 0)
    assert np.mean(np.abs(result - water)) < 0.08


@pytest.mark.parametrize("skin", ([0.68, 0.5, 0.4], [0.25, 0.18, 0.13]))
def test_skin_color_targets_are_not_pushed_to_orange_or_purple(skin):
    board = np.full((100, 200, 3), [0.08, 0.4, 0.65], np.float32)
    board[30:70, 50:150] = skin
    result = grade_image(board, PRESETS["natural"])
    center = result[50, 100]
    assert center[0] > center[1] > center[2]
    assert center[0] / center[1] < 1.7
    assert np.max(np.abs(center - skin)) < 0.08


def test_black_equipment_and_existing_highlights_remain_in_place():
    board = np.full((100, 200, 3), [0.08, 0.4, 0.65], np.float32)
    board[30:70, 50:100] = 0
    board[10:15, 150:170] = 0.98
    result = grade_image(board, PRESETS["natural"])
    np.testing.assert_array_equal(result[40:60, 60:90], np.zeros((20, 30, 3), np.float32))
    assert np.unravel_index(np.argmax(result[..., 1]), result.shape[:2]) == (10, 150)
