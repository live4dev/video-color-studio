"""Compare one explicitly selected local skin candidate through the shared engines.

All media/sample paths are supplied by the developer. Only scalar quality reports
and paths are written; no raster masks, thumbnails, audio or media are returned.
This process-local override never changes the desktop default or MCP configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from freedive_color.core import masks
from tools.skin_candidates import CANDIDATE_NAMES, DEFAULT_MODEL_DIR, make_predictor
from tools.verify_skin_photo import evaluate_photo
from tools.verify_skin_video import evaluate_video, peak_rss_bytes


def _write_report(path: Path, report: dict) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _quality_summary(photo: dict, video: dict) -> dict:
    measurements = [photo]
    for frame in video["frames"].values():
        measurements.append(frame["static"])
        if "tracked" in frame:
            measurements.append(frame["tracked"])
    regions = [region for measured in measurements for region in measured["regions"].values()]
    failures = sum(not region["passed"] for region in regions)
    return {
        "measured_checks": len(regions),
        "passed_checks": len(regions) - failures,
        "failed_checks": failures,
        "missing_video_samples": video["missing_samples"],
        "worst_alpha_error": max(
            (
                1 - region["mean_alpha"] if region["expected"] == "skin" else region["mean_alpha"]
                for region in regions
            ),
            default=None,
        ),
        "acceptance_passed": photo["acceptance_passed"] and video["acceptance_passed"],
    }


def evaluate_candidate(
    candidate: str,
    photo: Path,
    photo_samples: Path,
    video: Path,
    video_samples: Path,
    output_directory: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    static_only: bool = False,
) -> dict:
    """Run fixed photo and video regressions without selecting a product backend."""
    if candidate not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown candidate: {candidate}")
    output_directory = output_directory.expanduser().resolve()
    # Reserve the complete run before model initialization or source analysis. Even an
    # empty previous directory is evidence and must never be reused or overwritten.
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    photo_path = output_directory / "photo.json"
    video_path = output_directory / "video.json"
    summary_path = output_directory / "summary.json"
    predictor = make_predictor(candidate, model_dir)
    # build_masks resolves this symbol in its module globals. Both scalar verifiers
    # and TemporalMaskTracker therefore exercise the actual common processing code.
    # patch restores the original callable even if evaluation raises or is cancelled.
    with patch.object(masks, "infer_skin", predictor):
        photo_report = evaluate_photo(photo, photo_samples)
        _write_report(photo_path, photo_report)
        video_report = evaluate_video(video, video_samples, static_only=static_only)
        _write_report(video_path, video_report)
    summary = {
        "schema": 1,
        "private_local_qa": True,
        "scope": "Sparse labelled-region comparison; not full segmentation ground truth",
        "candidate": candidate,
        "execution_status": "completed",
        "mode": "static-only" if static_only else "static-and-tracked",
        "model_provenance": photo_report["metadata"].get("skin_models", []),
        "skin_algorithm_version": photo_report["metadata"].get("skin_algorithm_version"),
        **_quality_summary(photo_report, video_report),
        "reports": {
            "photo": str(photo_path),
            "video": str(video_path),
            "summary": str(summary_path),
        },
        "performance": {
            "total_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "product_default_changed": False,
        "media_contents_returned": False,
    }
    _write_report(summary_path, summary)
    return summary


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATE_NAMES, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--photo-samples", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-samples", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(arguments)
    if args.output_directory.expanduser().exists():
        parser.error("Use a fresh output directory; existing evidence is never overwritten")
    summary = evaluate_candidate(
        args.candidate,
        args.photo,
        args.photo_samples,
        args.video,
        args.video_samples,
        args.output_directory,
        model_dir=args.model_dir,
        static_only=args.static_only,
    )
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    return 2 if args.require_pass and not summary["acceptance_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
