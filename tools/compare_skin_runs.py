"""Fail-closed selection from fixed photo/video scalar regression reports.

This developer tool never changes a runtime backend or installation. Private input
reports, coordinates and measurements must remain outside the public repository.
"""

import argparse
import json
import math
from pathlib import Path


def _measurements(photo, video):
    if video.get("mode") != "static-and-tracked" or video.get("missing_samples"):
        raise ValueError("Selection requires every fixed static and tracked video sample")
    rows = {}
    for name, row in photo["regions"].items():
        rows[f"photo/{name}"] = row
    for sample, frame in video["frames"].items():
        for mode in ("static", "tracked"):
            for name, row in frame[mode]["regions"].items():
                rows[f"video/{sample}/{mode}/{name}"] = row
    for report in (photo, video):
        if (
            not isinstance(report.get("sample_manifest_sha256"), str)
            or not report["sample_manifest_sha256"]
        ):
            raise ValueError("Fixed sample manifest hash is required")
        if (
            report.get("source_unchanged") is not True
            or report.get("source_sha256_after") != report["source_sha256"]
        ):
            raise ValueError("A regression source was modified")
        if report["thresholds"] != {
            "positive_mean_alpha_min": 0.5,
            "negative_mean_alpha_max": 0.05,
        }:
            raise ValueError("Fixed acceptance thresholds were changed")
    for row in rows.values():
        value = row["mean_alpha"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Alpha must be a finite scalar in [0,1]")
        if row["expected"] not in {"skin", "excluded"}:
            raise ValueError("Unknown region annotation")
    return rows


def _passed(row):
    return row["mean_alpha"] >= 0.5 if row["expected"] == "skin" else row["mean_alpha"] <= 0.05


def select_variant(baseline_photo: dict, baseline_video: dict, runs: dict) -> dict:
    baseline = _measurements(baseline_photo, baseline_video)
    baseline_passed = {key for key, row in baseline.items() if _passed(row)}
    signatures = {key: (row["expected"], row["pixel_count"]) for key, row in baseline.items()}
    scores = {}
    if not runs:
        raise ValueError("At least one candidate run is required")
    for name, (photo, video) in runs.items():
        rows = _measurements(photo, video)
        if {key: (row["expected"], row["pixel_count"]) for key, row in rows.items()} != signatures:
            raise ValueError("Candidate regions do not match the immutable baseline")
        if (
            photo["source_sha256"] != baseline_photo["source_sha256"]
            or video["source_sha256"] != baseline_video["source_sha256"]
        ):
            raise ValueError("Candidates did not use identical source media")
        for reference, candidate in ((baseline_photo, photo), (baseline_video, video)):
            if reference["sample_manifest_sha256"] != candidate["sample_manifest_sha256"]:
                raise ValueError("Candidate sample coordinates changed")
        for sample, frame in video["frames"].items():
            for key in ("source_frame_index", "source_pts_seconds", "requested_seconds"):
                if frame.get(key) != baseline_video["frames"][sample].get(key):
                    raise ValueError("Candidate control-frame timestamps changed")
        failures = [key for key, row in rows.items() if not _passed(row)]
        seconds = sum(report["performance"]["total_seconds"] for report in (photo, video))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid measured runtime")
        scores[name] = {
            "eligible": not failures,
            "check_count": len(rows),
            "failed_checks": failures,
            "negative_failures": sum(rows[key]["expected"] == "excluded" for key in failures),
            "regressed_checks": sorted(baseline_passed.intersection(failures)),
            "worst_error": max(
                abs(row["mean_alpha"] - (1 if row["expected"] == "skin" else 0))
                for row in rows.values()
            ),
            "total_seconds": seconds,
        }
    eligible = [name for name, score in scores.items() if score["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda name: (scores[name]["worst_error"], scores[name]["total_seconds"], name),
        )
        if eligible
        else None
    )
    experimental = min(
        scores,
        key=lambda name: (
            scores[name]["negative_failures"],
            len(scores[name]["failed_checks"]),
            scores[name]["worst_error"],
            scores[name]["total_seconds"],
            name,
        ),
    )
    return {
        "schema": 1,
        "private_local_qa": True,
        "scope": "Fixed sparse regression only; does not establish general underwater accuracy",
        "acceptance_passed": selected is not None,
        "selected_variant": selected,
        "best_experimental_variant": experimental,
        "installation_update_allowed": selected is not None,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-photo", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument(
        "--run", nargs=3, metavar=("NAME", "PHOTO", "VIDEO"), action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for name, photo, video in args.run:
        if name in runs:
            parser.error("Candidate names must be unique")
        runs[name] = (json.loads(Path(photo).read_text()), json.loads(Path(video).read_text()))
    result = select_variant(
        json.loads(args.baseline_photo.read_text()),
        json.loads(args.baseline_video.read_text()),
        runs,
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "report": str(args.output.resolve()),
                "selected_variant": result["selected_variant"],
                "installation_update_allowed": result["installation_update_allowed"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
