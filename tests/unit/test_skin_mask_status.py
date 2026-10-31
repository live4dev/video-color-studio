from types import SimpleNamespace

import numpy as np
import pytest

from freedive_color.core.local_pipeline import LocalSettings, apply_local_grade
from freedive_color.core.masks import MaskEdits, MaskSet


@pytest.mark.parametrize(
    "status,expected",
    [
        ("automatic", "автоматическая"),
        ("manual", "ручная"),
        ("needs-review", "требует проверки"),
        ("unknown", "требует проверки"),
    ],
)
def test_ui_skin_state_does_not_present_model_score_as_accuracy(status, expected):
    from freedive_color.ui.main_window import MainWindow

    labels = []
    masks = SimpleNamespace(
        skin_status=status, skin_confidence=0.99, warnings=("Проверьте границы",)
    )
    window = SimpleNamespace(
        local_controls=SimpleNamespace(
            overlay_combo=SimpleNamespace(currentData=lambda: "none"),
            confidence_label=SimpleNamespace(setText=labels.append),
        ),
        preview=SimpleNamespace(masks=masks),
        viewer=SimpleNamespace(set_mask_overlay=lambda *_args: None),
    )
    MainWindow._update_mask_overlay(window)
    assert labels == [f"Маска кожи: {expected}\nПроверьте границы"]
    assert "%" not in labels[0]


def test_underwater_clipping_keeps_semantic_provenance_and_exclusions():
    rgb = np.full((20, 20, 3), 0.3, np.float32)
    ones = np.ones(rgb.shape[:2], np.float32)
    underwater = ones.copy()
    underwater[:10] = 0
    exclusion = np.zeros_like(ones)
    exclusion[2:5, 2:5] = 1
    diagnostics = {"automatic_accepted": True, "selected_pixels": 400}
    masks = MaskSet(
        ones,
        ones,
        ones * 0,
        underwater,
        0.92,
        0.99,
        skin_source="selfie-multiclass-v1",
        skin_status="automatic",
        skin_model_sha256="a" * 64,
        skin_diagnostics=diagnostics,
        skin_exclusion=exclusion,
    )
    result = apply_local_grade(
        rgb,
        rgb,
        LocalSettings(skin_strength=0, water_strength=0),
        MaskEdits(waterline=((0, 0.5), (1, 0.5))),
        masks,
    )
    assert not result.masks.skin[:10].any()
    assert result.masks.skin[10:].min() == 1
    assert result.masks.skin_source == masks.skin_source
    assert result.masks.skin_status == masks.skin_status
    assert result.masks.skin_model_sha256 == masks.skin_model_sha256
    assert result.masks.skin_diagnostics == diagnostics
    np.testing.assert_array_equal(result.masks.skin_exclusion, exclusion)
    np.testing.assert_array_equal(masks.skin, ones)
