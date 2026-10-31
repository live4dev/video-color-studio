from dataclasses import replace

import cv2
import numpy as np
import pytest

from freedive_color.core.analyze import analyze_scene
from freedive_color.core.errors import StudioError
from freedive_color.core.masks import BrushStroke, MaskEdits, MaskSet
from freedive_color.core.temporal import (
    FrameAnchor,
    ShotDetector,
    ShotPlan,
    TemporalAnalyzer,
    TemporalMaskTracker,
    build_shot_plan,
)


def blue(shape=(64, 96)):
    return np.full((*shape, 3), [0.1, 0.4, 0.65], np.float32)


def green(shape=(64, 96)):
    return np.full((*shape, 3), [0.12, 0.6, 0.2], np.float32)


def test_shot_detector_ignores_sensor_noise_and_detects_color_cut():
    detector = ShotDetector()
    assert detector.update(blue()) is False
    noise = np.random.default_rng(31).normal(0, 0.004, blue().shape).astype(np.float32)
    assert detector.update(np.clip(blue() + noise, 0, 1)) is False
    assert detector.update(green()) is True
    assert detector.last_score > 0.42
    assert detector.update(green()) is False


def test_steady_shot_uses_representative_samples_not_every_frame():
    analyzer = TemporalAnalyzer()
    for index in range(180):
        assert analyzer.add(blue(), index / 30, index) is False
        assert len(analyzer._pending) <= 5
    plan = analyzer.finish(6)
    assert plan.sampled_frames == 12
    assert plan.frame_count == 180
    assert len(plan.anchors) == 3
    assert plan.cuts == ()
    values = [plan.analysis_at(index / 30).red_gain for index in range(180)]
    assert max(values) - min(values) < 1e-10
    assert not any(
        isinstance(value, np.ndarray)
        for anchor in plan.anchors
        for value in anchor.analysis.to_dict().values()
    )


def test_color_noise_does_not_pump_analysis_gains():
    analyzer = TemporalAnalyzer()
    raw = []
    for index in range(180):
        frame = blue()
        frame[..., 0] += 0.012 if index % 2 else -0.012
        raw.append(analyze_scene(frame).red_gain)
        analyzer.add(frame, index / 30, index)
    plan = analyzer.finish(6)
    assert plan.cuts == ()
    smooth = [plan.analysis_at(index / 30).red_gain for index in range(180)]
    assert np.std(smooth) < np.std(raw) * 0.1


def test_gradual_depth_change_has_smooth_bounded_coefficients():
    analyzer = TemporalAnalyzer()
    for index in range(300):
        frame = blue()
        frame[..., 0] = 0.17 - 0.13 * index / 299
        analyzer.add(frame, index / 30, index)
    plan = analyzer.finish(10)
    assert plan.cuts == ()
    gains = np.array([plan.analysis_at(index / 30).red_gain for index in range(300)])
    assert np.max(np.abs(np.diff(gains))) < 0.0021
    assert gains[0] > gains[-1] + 0.03
    assert np.all((gains >= 1) & (gains <= 1.75))


def test_plan_does_not_interpolate_across_hard_cut():
    samples = [(index / 10, blue() if index < 20 else green()) for index in range(40)]
    plan = build_shot_plan(samples, 4)
    assert plan.cuts == (2.0,)
    assert plan.shot_at(1.999) == 0 and plan.shot_at(2) == 1
    before, after = plan.analysis_at(1.999), plan.analysis_at(2)
    assert before.red_gain == pytest.approx(analyze_scene(blue()).red_gain)
    assert after.red_gain == pytest.approx(analyze_scene(green()).red_gain)
    assert before.red_gain != after.red_gain


def test_nonzero_source_pts_origin_and_plan_serialization():
    analyzer = TemporalAnalyzer()
    for index in range(40):
        analyzer.add(blue() if index < 20 else green(), 100 + index / 10, index)
    plan = analyzer.finish(4)
    assert plan.start_time == 100
    assert plan.duration == 4
    assert plan.cuts == (102,)
    restored = ShotPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.analysis_at(102.5) == plan.analysis_at(102.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TemporalAnalyzer(sample_interval=0),
        lambda: TemporalAnalyzer(anchor_interval=0.2),
        lambda: TemporalMaskTracker(refresh_seconds=0),
        lambda: TemporalMaskTracker(proxy_edge=8),
        lambda: ShotDetector(threshold=float("nan")),
        lambda: FrameAnchor(float("inf"), analyze_scene(blue())),
        lambda: ShotPlan((), (), 1),
    ],
)
def test_invalid_temporal_parameters_are_explicit(factory):
    with pytest.raises(StudioError):
        factory()


def test_analyzer_rejects_backwards_time_and_empty_stream():
    analyzer = TemporalAnalyzer()
    with pytest.raises(StudioError, match="не содержит"):
        analyzer.finish()
    analyzer.add(blue(), 1)
    with pytest.raises(StudioError, match="обратном"):
        analyzer.add(blue(), 0.9)
    analyzer.finish(1)
    with pytest.raises(StudioError, match="завершён"):
        analyzer.add(blue(), 1.1)


def test_inconsistent_serialized_shot_anchors_rejected():
    anchor = FrameAnchor(0, analyze_scene(blue()))
    plan = ShotPlan((anchor,), (), 1)
    data = plan.to_dict()
    data["anchors"][0]["shot_id"] = 2
    with pytest.raises(StudioError):
        ShotPlan.from_dict(data)


def textured_subject(offset=0, *, shape=(96, 160)):
    rng = np.random.default_rng(23)
    rgb = np.clip(blue(shape) + rng.normal(0, 0.007, (*shape, 3)), 0, 1).astype(np.float32)
    tile = np.full((24, 24, 3), [0.65, 0.48, 0.38], np.float32)
    tile += rng.normal(0, 0.018, tile.shape).astype(np.float32)
    rgb[35:59, 35 + offset : 59 + offset] = tile
    return np.clip(rgb, 0, 1).astype(np.float32)


class FakeSemanticBuilder:
    def __init__(self, alternating=False):
        self.calls = 0
        self.alternating = alternating

    def __call__(self, rgb, _edits=None, **_kwargs):
        self.calls += 1
        person = ((rgb[..., 0] > rgb[..., 1]) & (rgb[..., 0] > rgb[..., 2])).astype(np.float32)
        strength = (0.8 if self.calls % 2 else 0.95) if self.alternating else 0.9
        ones = np.ones(rgb.shape[:2], np.float32)
        return MaskSet(
            person,
            person * strength,
            1 - person,
            ones,
            0.9,
            0.95,
            skin_source="selfie-multiclass-v1",
            skin_status="automatic",
            skin_model_sha256="a" * 64,
            skin_diagnostics={"score_kind": "uncalibrated-model-probability"},
        )


def centroid(mask):
    y, x = np.mgrid[: mask.shape[0], : mask.shape[1]]
    return (
        float((x * mask).sum() / max(float(mask.sum()), 1e-6)),
        float((y * mask).sum() / max(float(mask.sum()), 1e-6)),
    )


def test_backward_flow_tracks_only_mask_not_source_geometry():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(refresh_seconds=10, proxy_edge=192, mask_builder=builder)
    for index in range(8):
        rgb = textured_subject(index * 2)
        original = rgb.copy()
        masks = tracker.update(rgb, index / 30, shot_id=0)
        np.testing.assert_array_equal(rgb, original)
    assert builder.calls == 1
    x, y = centroid(masks.skin)
    assert x == pytest.approx(46.5 + 14, abs=3)
    assert y == pytest.approx(46.5, abs=2)
    assert masks.person.shape == rgb.shape[:2]


def test_periodic_semantic_updates_are_bounded_and_alpha_is_smoothed():
    builder = FakeSemanticBuilder(alternating=True)
    tracker = TemporalMaskTracker(refresh_seconds=0.25, proxy_edge=192, mask_builder=builder)
    strengths = []
    for index in range(60):
        masks = tracker.update(textured_subject(), index / 30, shot_id=0)
        strengths.append(float(masks.skin[47, 47]))
    assert 2 <= builder.calls <= 9
    assert np.max(np.abs(np.diff(strengths))) < 0.035
    assert tracker.to_metadata()["semantic_refreshes"] == builder.calls


def test_hard_cut_resets_without_previous_person_ghost():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(refresh_seconds=10, proxy_edge=192, mask_builder=builder)
    first = tracker.update(textured_subject(), 0, shot_id=0)
    assert first.skin[47, 47] > 0.8
    after = tracker.update(textured_subject(70), 1 / 30, shot_id=1)
    assert after.skin[47, 47] == 0
    assert after.skin[47, 117] > 0.8
    assert builder.calls == 2 and tracker.resets == 2


def test_unreliable_or_black_occlusion_abstains_from_skin():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(refresh_seconds=10, proxy_edge=192, mask_builder=builder)
    rgb = textured_subject()
    tracker.update(rgb, 0, shot_id=0)
    rgb[35:59, 35:59] = 0.02
    masks = tracker.update(rgb, 1 / 30, shot_id=0)
    assert masks.skin.max() == 0
    assert masks.skin_confidence == 0 or masks.warnings


def test_manual_video_edits_remain_in_explicit_frame_coordinates():
    edits = MaskEdits(
        strokes=(BrushStroke("skin", "add", ((0.5, 0.5),), 0.08),),
        waterline=((0, 0.4), (0.5, 0.5), (1, 0.4)),
    )
    tracker = TemporalMaskTracker(edits, use_model=False, proxy_edge=128)
    for index in range(3):
        masks = tracker.update(blue((120, 180)), index / 30)
        assert masks.skin[60, 90] > 0.9
        assert masks.underwater[10, 90] == 0
        assert masks.water[10].max() == 0
        assert masks.person_confidence >= 0.9
        assert any("координатах" in message for message in masks.warnings)


def test_tracking_memory_is_independent_of_frame_count_and_source_resolution():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(refresh_seconds=10, proxy_edge=128, mask_builder=builder)
    large = blue((360, 640))
    tracker.update(large, 0, shot_id=0)
    retained = tracker.retained_bytes
    for index in range(1, 30):
        masks = tracker.update(large, index / 30, shot_id=0)
    assert masks.skin.shape == (360, 640)
    assert tracker.retained_bytes == retained
    assert tracker.retained_bytes < 128 * 128 * 34
    assert builder.calls == 1


def test_fresh_mask_with_low_confidence_cannot_keep_old_skin_alive():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(refresh_seconds=10, proxy_edge=192, mask_builder=builder)
    rgb = textured_subject()
    tracker.update(rgb, 0)
    fresh = replace(builder(rgb), skin_confidence=0)
    masks = tracker.update(rgb, 1 / 30, fresh_masks=fresh)
    assert masks.skin.max() == 0


def semantic_patch(rgb):
    shape = rgb.shape[:2]
    person = np.zeros(shape, np.float32)
    person[8:-8, 8:-8] = 1
    skin = np.zeros(shape, np.float32)
    skin[30:55, 30:55] = 0.95
    return MaskSet(
        person,
        skin,
        1 - person,
        np.ones(shape, np.float32),
        0.94,
        0.4,  # The skin classifier is independent of the old silhouette score.
        skin_source="selfie-multiclass-v1",
        skin_status="automatic",
        skin_model_sha256="a" * 64,
        skin_diagnostics={"score_kind": "uncalibrated-model-probability"},
        skin_exclusion=np.zeros(shape, np.float32),
    )


@pytest.mark.parametrize("color", [[0.08, 0.32, 0.49], [0.08, 0.26, 0.16], [0.02, 0.035, 0.06]])
def test_semantic_skin_is_not_vetoed_by_blue_green_or_dark_pixels(color):
    rgb = blue((96, 160))
    rgb[30:55, 30:55] = color
    original = rgb.copy()
    masks = semantic_patch(rgb)
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    result = tracker.update(rgb, 0, fresh_masks=masks)
    assert result.skin[42, 42] > 0.9
    assert result.skin_status == "automatic"
    assert result.skin_model_sha256 == "a" * 64
    assert result.skin_diagnostics == masks.skin_diagnostics
    np.testing.assert_array_equal(rgb, original)


def test_semantic_exclusion_removes_hair_or_gear_immediately_without_fading():
    rgb = textured_subject()
    masks = semantic_patch(rgb)
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    tracker.update(rgb, 0, fresh_masks=masks)
    excluded = masks.skin_exclusion.copy()
    excluded[36:42, 36:42] = 0.95
    result = tracker.update(rgb, 1 / 30, fresh_masks=replace(masks, skin_exclusion=excluded))
    assert result.skin[38, 38] == 0
    assert result.skin[50, 50] > 0.9
    assert result.skin_exclusion[38, 38] > 0.9


@pytest.mark.parametrize("mode", ["empty", "needs-review"])
def test_fresh_abstention_or_missing_person_clears_old_skin_immediately(mode):
    rgb = textured_subject()
    masks = semantic_patch(rgb)
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    tracker.update(rgb, 0, fresh_masks=masks)
    fresh = (
        replace(masks, skin=np.zeros_like(masks.skin), person=np.zeros_like(masks.person))
        if mode == "empty"
        else replace(masks, skin_status="needs-review")
    )
    result = tracker.update(rgb, 1 / 30, fresh_masks=fresh)
    assert result.skin.max() == 0
    again = tracker.update(rgb, 2 / 30)
    assert again.skin.max() == 0


def test_small_occlusion_cannot_hide_inside_a_large_person(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowFarneback",
        lambda current, *_args: np.zeros((*current.shape, 2), np.float32),
    )
    rgb = textured_subject()
    masks = semantic_patch(rgb)
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    tracker.update(rgb, 0, fresh_masks=masks)
    covered = rgb.copy()
    covered[40:44, 40:44] = 0.02  # Less than 3% of skin and 0.2% of person.
    result = tracker.update(covered, 1 / 30)
    assert result.skin[41, 41] == 0
    assert result.skin[50, 50] > 0.9
    assert tracker.skin_rejections == 1
    assert tracker.flow_failures == 0


def test_color_occlusion_with_same_luminance_is_rejected(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowFarneback",
        lambda current, *_args: np.zeros((*current.shape, 2), np.float32),
    )
    rgb = blue((96, 160))
    rgb[30:55, 30:55] = [0.2, 0.45, 0.4]
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    tracker.update(rgb, 0, fresh_masks=semantic_patch(rgb))
    covered = rgb.copy()
    covered[40:44, 40:44] = [0.55, 0.34595, 0.4]
    luma = np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert abs(float((covered[41, 41] - rgb[41, 41]) @ luma)) < 1e-4
    result = tracker.update(covered, 1 / 30)
    assert result.skin[41, 41] == 0
    assert result.skin[50, 50] > 0.9


def test_flow_failure_discards_history_until_a_new_semantic_observation(monkeypatch):
    rgb = textured_subject()
    tracker = TemporalMaskTracker(use_model=False, proxy_edge=192)
    tracker.update(rgb, 0, fresh_masks=semantic_patch(rgb))

    def failure(*_args):
        raise cv2.error("failed flow")

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", failure)
    result = tracker.update(rgb, 1 / 30)
    assert result.skin.max() == 0
    assert result.skin_status == "needs-review"
    assert tracker.to_metadata()["skin_status"] == "needs-review"
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowFarneback",
        lambda current, *_args: np.zeros((*current.shape, 2), np.float32),
    )
    result = tracker.update(rgb, 2 / 30)
    assert result.skin.max() == 0
    result = tracker.update(rgb, 3 / 30, fresh_masks=semantic_patch(rgb))
    assert result.skin[42, 42] > 0


def test_default_refresh_uses_half_second_in_source_pts():
    builder = FakeSemanticBuilder()
    tracker = TemporalMaskTracker(mask_builder=builder)
    for timestamp in (100, 100.1, 100.49, 100.5, 100.99, 101):
        tracker.update(textured_subject(), timestamp, shot_id=0)
    assert builder.calls == 3
    metadata = tracker.to_metadata()
    assert metadata["refresh_seconds"] == 0.5
    assert metadata["skin_source"] == "selfie-multiclass-v1"
    assert metadata["skin_model_sha256"] == "a" * 64
    assert sum(metadata["skin_status_frames"].values()) == 6
