"""Measure local skin masks against sparse labelled video regions, without exposing pixels.

The samples JSON contains ``points: {label: {region_name: [x, y, radius]}}`` in
display-oriented source pixels. Numeric labels are requested seconds relative to
the video's first PTS; --time LABEL=SECONDS overrides them. Names beginning with
``skin_`` are positive skin regions, all other names are negative regions.

This is a sparse regression check, not a pixel-perfect ground truth segmentation.
No images, audio or media buffers are returned by this tool. Optional alpha PNGs
remain local for the user to inspect. Existing inputs, reports and PNGs are protected.
"""

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from freedive_color.core.masks import MaskSet, build_masks
from freedive_color.core.project import sha256_file
from freedive_color.core.temporal import TemporalMaskTracker
from freedive_color.core.video_pipeline import decode_video_frame, probe_video
from freedive_color.integrations.ffmpeg import open_local_media


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def read_samples(path: Path, times: dict[str, float] | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points") if isinstance(data, dict) else None
    if not isinstance(points, dict) or not points:
        raise ValueError("Samples must contain a nonempty points object")
    times = times or {}
    if set(times) - set(points):
        raise ValueError("Time overrides must refer to existing sample labels")
    samples = {}
    for label, regions in points.items():
        if not isinstance(label, str) or not label or not isinstance(regions, dict) or not regions:
            raise ValueError("Each sample needs a string label and nonempty regions")
        seconds = float(times.get(label, label))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Sample times must be finite nonnegative seconds")
        for name, disk in regions.items():
            if not isinstance(name, str) or not name or not isinstance(disk, (tuple, list)):
                raise ValueError("Each region needs a name and x/y/radius")
            if len(disk) != 3 or any(
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                for value in disk
            ):
                raise ValueError("Region coordinates must be finite x/y/radius numbers")
            x, y, radius = disk
            if min(x, y) < 0 or radius <= 0:
                raise ValueError("Region coordinates must be nonnegative and radius positive")
        samples[label] = {"requested_seconds": seconds, "regions": regions}
    return samples


def evaluate_regions(
    masks: MaskSet,
    regions: dict,
    *,
    positive_min: float = 0.5,
    negative_max: float = 0.05,
) -> dict:
    height, width = masks.skin.shape
    result = {}
    for name, (x, y, radius) in regions.items():
        if x >= width or y >= height:
            raise ValueError(f"Region {name} is outside the display-oriented source frame")
        left, right = max(0, math.floor(x - radius)), min(width, math.ceil(x + radius) + 1)
        top, bottom = max(0, math.floor(y - radius)), min(height, math.ceil(y + radius) + 1)
        yy, xx = np.ogrid[top:bottom, left:right]
        selected = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        if not np.any(selected):
            raise ValueError(f"Region {name} contains no sample pixels")
        alpha = masks.skin[top:bottom, left:right][selected]
        positive = name.startswith("skin_")
        mean = float(np.mean(alpha))
        result[name] = {
            "expected": "skin" if positive else "excluded",
            "pixel_count": int(alpha.size),
            "mean_alpha": mean,
            "minimum_alpha": float(np.min(alpha)),
            "maximum_alpha": float(np.max(alpha)),
            "fraction_alpha_gt_0_5": float(np.mean(alpha > 0.5)),
            "mean_person_alpha": float(np.mean(masks.person[top:bottom, left:right][selected])),
            "passed": mean >= positive_min if positive else mean <= negative_max,
        }
    return result


def mask_measurement(masks: MaskSet, regions: dict, positive_min: float, negative_max: float):
    stats = evaluate_regions(masks, regions, positive_min=positive_min, negative_max=negative_max)
    return {
        "metadata": masks.to_metadata(),
        "regions": stats,
        "failed_regions": [name for name, values in stats.items() if not values["passed"]],
    }


def reached_sample(timestamp: Fraction, origin: Fraction, seconds: float) -> bool:
    # Decimal CLI floats can lie a few femtoseconds above an exact source PTS.
    # One nanosecond of tolerance avoids skipping that frame, while preserving
    # substantially finer precision than the video's media clock.
    return timestamp + Fraction(1, 1_000_000_000) >= origin + Fraction(str(seconds))


def verify_export_streams(source: Path, output: Path) -> dict:
    # Independent packet/clock audit, separate from the exporting engine's own report.
    from tools.render_video_evidence import inspect_video, verify_full_export

    before, after = inspect_video(source), inspect_video(output)
    result = verify_full_export(before, after)
    result["source_frame_count"] = before["frame_count"]
    result["output_frame_count"] = after["frame_count"]
    result["all_video_pts_exact"] = all(
        Fraction(a) == Fraction(b)
        for a, b in zip(before["video_pts"], after["video_pts"], strict=True)
    )
    for index, (a, b) in enumerate(zip(before["audio"], after["audio"], strict=True)):
        for key in ("pts", "dts", "duration"):
            result["audio"][index][f"all_{key}_exact"] = all(
                x == y if x is None or y is None else Fraction(x) == Fraction(y)
                for x, y in zip(a[key], b[key], strict=True)
            )
    return result


def evaluate_video(
    source: Path,
    samples_path: Path,
    *,
    times: dict[str, float] | None = None,
    positive_min: float = 0.5,
    negative_max: float = 0.05,
    alpha_directory: Path | None = None,
    verify_export: Path | None = None,
    static_only: bool = False,
    progress=lambda _event: None,
) -> dict:
    if not 0 <= positive_min <= 1 or not 0 <= negative_max <= 1:
        raise ValueError("Acceptance thresholds must be in [0,1]")
    source = source.expanduser().resolve()
    samples = read_samples(samples_path, times)
    if alpha_directory is not None:
        alpha_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_sha = sha256_file(source)
    info = probe_video(source)
    tracker = None if static_only else TemporalMaskTracker()
    remaining = dict(samples)
    frames = {}
    tracking_seconds = 0.0
    static_seconds = 0.0
    frame_count = 0
    with open_local_media(source) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError("A source frame has no presentation timestamp")
            timestamp = Fraction(frame.pts) * frame.time_base
            frame_count += 1
            ready = [
                label
                for label, sample in remaining.items()
                if reached_sample(timestamp, info.video_start, sample["requested_seconds"])
            ]
            if static_only and not ready:
                # Count the complete source timeline, but avoid expensive SDR/RGB
                # conversion for frames that the static-only audit does not measure.
                continue
            rgb = decode_video_frame(frame, info)
            tracked = None
            if tracker is not None:
                track_started = time.perf_counter()
                tracked = tracker.update(rgb, float(timestamp))
                tracking_seconds += time.perf_counter() - track_started
            if not ready:
                continue
            static_started = time.perf_counter()
            static = build_masks(rgb)
            static_seconds += time.perf_counter() - static_started
            for label in ready:
                sample = remaining.pop(label)
                measured = {
                    "requested_seconds": sample["requested_seconds"],
                    "source_frame_index": frame_index,
                    "source_pts_seconds": str(timestamp),
                    "relative_seconds": float(timestamp - info.video_start),
                    "static": mask_measurement(
                        static, sample["regions"], positive_min, negative_max
                    ),
                }
                if tracked is not None:
                    measured["tracked"] = mask_measurement(
                        tracked, sample["regions"], positive_min, negative_max
                    )
                if alpha_directory is not None:
                    paths = {}
                    # Labels are user-supplied; use a numeric index as a safe filename.
                    number = len(frames)
                    for mode, masks in (("static", static), ("tracked", tracked)):
                        if masks is None:
                            continue
                        path = alpha_directory / f"sample-{number:03d}-{mode}-skin-alpha.png"
                        with path.open("xb") as output:
                            Image.fromarray(np.rint(masks.skin * 255).astype(np.uint8)).save(
                                output, format="PNG"
                            )
                        paths[mode] = str(path.resolve())
                    measured["local_alpha_paths"] = paths
                frames[label] = measured
                progress({"sample": label, "frame_index": frame_index})
    source_after = sha256_file(source)
    if source_after != source_sha:
        raise AssertionError("Source media changed during the read-only mask audit")
    failures = [
        {"sample": label, "mode": mode, "region": name}
        for label, sample in frames.items()
        for mode in (("static",) if static_only else ("static", "tracked"))
        for name in sample[mode]["failed_regions"]
    ]
    report = {
        "schema": 1,
        "scope": "Sparse labelled-region regression; does not establish full segmentation accuracy",
        "private_local_qa": True,
        "mode": "static-only" if static_only else "static-and-tracked",
        "source": str(source),
        "source_sha256": source_sha,
        "source_sha256_after": source_after,
        "sample_manifest_sha256": sha256_file(samples_path),
        "source_unchanged": True,
        "width": info.width,
        "height": info.height,
        "frame_count": frame_count,
        "thresholds": {
            "positive_mean_alpha_min": positive_min,
            "negative_mean_alpha_max": negative_max,
        },
        "frames": frames,
        "missing_samples": list(remaining),
        "failed_regions": failures,
        "acceptance_passed": not failures and not remaining,
        "tracker": tracker.to_metadata() if tracker is not None else {"enabled": False},
        "performance": {
            "total_seconds": time.perf_counter() - started,
            "tracking_seconds": tracking_seconds,
            "static_seconds": static_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    if verify_export is not None:
        report["independent_export_verification"] = verify_export_streams(source, verify_export)
    return report


def parse_times(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        label, separator, seconds = value.partition("=")
        if not separator or not label or label in result:
            raise ValueError("Use each --time once as LABEL=SECONDS")
        parsed = float(seconds)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("Sample times must be finite nonnegative seconds")
        result[label] = parsed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time", action="append", default=[], metavar="LABEL=SECONDS")
    parser.add_argument("--positive-min", type=float, default=0.5)
    parser.add_argument("--negative-max", type=float, default=0.05)
    parser.add_argument("--alpha-directory", type=Path)
    parser.add_argument("--verify-export", type=Path)
    parser.add_argument(
        "--static-only", action="store_true", help="Evaluate only the selected static frames"
    )
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (
        args.input.resolve(),
        args.samples.resolve(),
    ):
        parser.error("Choose a new report path; existing files and source inputs are protected")
    report = evaluate_video(
        args.input,
        args.samples,
        times=parse_times(args.time),
        positive_min=args.positive_min,
        negative_max=args.negative_max,
        alpha_directory=args.alpha_directory,
        verify_export=args.verify_export,
        static_only=args.static_only,
        progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output.resolve()),
                "acceptance_passed": report["acceptance_passed"],
                "failed_region_count": len(report["failed_regions"]),
                "missing_samples": report["missing_samples"],
            },
            ensure_ascii=False,
        )
    )
    if args.require_pass and not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
