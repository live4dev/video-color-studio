import json
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from freedive_color.core.masks import MaskSet
from tools import verify_skin_video as verifier


def mask(alpha):
    alpha = np.asarray(alpha, dtype=np.float32)
    ones = np.ones_like(alpha)
    return MaskSet(ones, alpha, ones - alpha, ones, 0.9, 0.99)


def test_sparse_regions_measure_interior_disks_and_report_positive_and_negative_failures():
    alpha = np.zeros((40, 60), np.float32)
    alpha[8:16, 8:16] = 0.8
    alpha[25:32, 25:32] = 0.2
    stats = verifier.evaluate_regions(
        mask(alpha),
        {
            "skin_arm": [12, 12, 2],
            "skin_leg": [40, 12, 2],
            "water": [40, 30, 2],
            "hair": [28, 28, 2],
        },
    )
    assert stats["skin_arm"]["passed"]
    assert stats["skin_arm"]["mean_alpha"] == pytest.approx(0.8)
    assert stats["skin_arm"]["pixel_count"] == 13
    assert not stats["skin_leg"]["passed"]
    assert stats["water"]["passed"]
    assert not stats["hair"]["passed"]
    with pytest.raises(ValueError, match="outside"):
        verifier.evaluate_regions(mask(alpha), {"skin_arm": [80, 12, 2]})


@pytest.mark.parametrize(
    "points",
    [
        {},
        {"05": {}},
        {"05": {"skin_arm": [2, 2, 0]}},
        {"05": {"skin_arm": [2, 2, True]}},
        {"05": {"skin_arm": [2, 2, float("nan")]}},
        {"05": {"skin_arm": [2, 2]}},
    ],
)
def test_invalid_sparse_samples_are_rejected(tmp_path, points):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"points": points}), encoding="utf-8")
    with pytest.raises(ValueError):
        verifier.read_samples(path)


def test_target_overrides_are_explicit_and_finite(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"points": {"face": {"skin_face": [2, 2, 1]}}}))
    samples = verifier.read_samples(path, {"face": 0.05})
    assert samples["face"]["requested_seconds"] == 0.05
    assert verifier.parse_times(["face=0.05"]) == {"face": 0.05}
    with pytest.raises(ValueError, match="existing sample"):
        verifier.read_samples(path, {"other": 1})
    for invalid in (["face=nan"], ["face=-1"], ["face=2", "face=3"], ["face"]):
        with pytest.raises(ValueError):
            verifier.parse_times(invalid)


def test_decimal_time_rounding_does_not_skip_the_exact_source_frame():
    exact = Fraction(541, 30)
    assert verifier.reached_sample(exact, Fraction(0), 18.033333333333335)
    assert not verifier.reached_sample(exact - Fraction(1, 600), Fraction(0), float(exact))
    assert verifier.reached_sample(exact + 2, Fraction(2), float(exact))


@pytest.mark.parametrize("static_only", (False, True))
def test_video_audit_uses_absolute_pts_origin_and_runs_tracker_past_last_sample(
    tmp_path, monkeypatch, static_only
):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"immutable source")
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps({"points": {"0": {"skin_face": [2, 2, 1]}, "later": {"hair": [2, 2, 1]}}})
    )
    frames = [SimpleNamespace(pts=pts, time_base=Fraction(1, 30)) for pts in (30, 31, 34, 37)]

    class Container:
        streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, _stream):
            yield from frames

    class Tracker:
        count = 0

        def __init__(self):
            assert not static_only, "Static-only QA must not initialize video tracking"

        def update(self, _rgb, timestamp):
            self.count += 1
            return mask(np.full((8, 8), 0.8 if timestamp == 1 else 0, np.float32))

        def to_metadata(self):
            return {"frames": self.count}

    monkeypatch.setattr(verifier, "open_local_media", lambda _path: Container())
    monkeypatch.setattr(
        verifier,
        "probe_video",
        lambda _path: SimpleNamespace(video_start=Fraction(1), width=8, height=8),
    )
    monkeypatch.setattr(verifier, "TemporalMaskTracker", Tracker)
    converted = []

    def decode(frame, _info):
        converted.append(frame.pts)
        return np.full((8, 8, 3), 0.2, np.float32)

    monkeypatch.setattr(verifier, "decode_video_frame", decode)
    monkeypatch.setattr(verifier, "build_masks", lambda _rgb: mask(np.zeros((8, 8), np.float32)))
    report = verifier.evaluate_video(
        source, samples, times={"later": 0.05}, static_only=static_only
    )
    assert report["frame_count"] == 4
    assert len(converted) == (2 if static_only else 4)
    if static_only:
        assert report["tracker"] == {"enabled": False}
        assert report["mode"] == "static-only"
        assert "tracked" not in report["frames"]["0"]
        assert report["performance"]["tracking_seconds"] == 0
    else:
        assert report["tracker"]["frames"] == 4
        assert report["mode"] == "static-and-tracked"
    assert report["source_unchanged"]
    assert source.read_bytes() == b"immutable source"
    assert report["frames"]["0"]["source_frame_index"] == 0
    assert report["frames"]["later"]["source_frame_index"] == 2
    assert report["frames"]["later"]["source_pts_seconds"] == "17/15"
    assert report["failed_regions"] == [{"sample": "0", "mode": "static", "region": "skin_face"}]
    assert not report["acceptance_passed"]
    json.dumps(report, allow_nan=False)  # The public result contains no ndarray/media buffers.


def test_independent_export_audit_reports_exact_rational_clocks_and_packet_durations(
    tmp_path, monkeypatch
):
    from tools import render_video_evidence

    before = {
        "width": 8,
        "height": 8,
        "time_base": "1/30000",
        "video_pts": ["1", "31/30"],
        "video_stream_start": "1",
        "video_stream_duration": "1/15",
        "frame_count": 2,
        "audio": [
            {
                "packet_hashes": ["packet-a", "packet-b"],
                "pts": ["1", "101/100"],
                "dts": ["1", "101/100"],
                "duration": ["1/100", "1/100"],
                "language": "eng",
                "stream_start": "1",
                "stream_duration": "1/50",
                "time_base": "1/48000",
            }
        ],
    }
    after = deepcopy(before)
    after["video_pts"][1] = "62/60"
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    monkeypatch.setattr(
        render_video_evidence, "inspect_video", lambda path: before if path == source else after
    )
    exact = verifier.verify_export_streams(source, output)
    assert exact["all_video_pts_exact"]
    assert exact["audio"][0]["all_pts_exact"]
    assert exact["audio"][0]["all_dts_exact"]
    assert exact["audio"][0]["all_duration_exact"]
    # The inherited stream verifier allows mux time-base rounding. This additional
    # flag distinguishes that tolerance from literal preservation of every source PTS.
    after["video_pts"][1] = "1033334/1000000"
    assert not verifier.verify_export_streams(source, output)["all_video_pts_exact"]
