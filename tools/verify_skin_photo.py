"""Private, scalar-only sparse skin regression for a display-oriented photograph.

No user media or coordinates are built in. Samples use the video verifier's format,
with one ``points.photo`` object. Optional source_sha256 and source_size pins prevent
silently measuring a different photograph or a resized screenshot.
"""

import argparse
import json
import time
from pathlib import Path

from freedive_color.core.masks import build_masks
from freedive_color.core.media_io import load_photo
from freedive_color.core.project import sha256_file
from tools.verify_skin_video import mask_measurement, peak_rss_bytes, read_samples


def evaluate_photo(source: Path, samples_path: Path) -> dict:
    source = source.expanduser().resolve()
    samples_data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = read_samples(samples_path, {"photo": 0})
    if set(samples) != {"photo"}:
        raise ValueError("Photo samples must contain exactly points.photo")
    started = time.perf_counter()
    before = sha256_file(source)
    if samples_data.get("source_sha256", before) != before:
        raise ValueError("Photograph SHA does not match the fixed samples")
    photo = load_photo(source)
    shape = [photo.width, photo.height]
    if samples_data.get("source_size", shape) != shape:
        raise ValueError("Photograph geometry does not match the fixed samples")
    masks = build_masks(photo.rgb)
    measurement = mask_measurement(masks, samples["photo"]["regions"], 0.5, 0.05)
    after = sha256_file(source)
    if before != after:
        raise AssertionError("Source photograph changed during mask evaluation")
    return {
        "schema": 1,
        "private_local_qa": True,
        "scope": "Sparse interior-region regression, not full segmentation ground truth",
        "source": str(source),
        "source_sha256": before,
        "source_sha256_after": after,
        "sample_manifest_sha256": sha256_file(samples_path),
        "source_unchanged": True,
        "width": photo.width,
        "height": photo.height,
        "thresholds": {"positive_mean_alpha_min": 0.5, "negative_mean_alpha_max": 0.05},
        **measurement,
        "acceptance_passed": not measurement["failed_regions"],
        "performance": {
            "total_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is never overwritten")
    report = evaluate_photo(args.input, args.samples)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "report": str(args.output.resolve()),
                "acceptance_passed": report["acceptance_passed"],
                "failed_regions": report["failed_regions"],
            },
            ensure_ascii=True,
        )
    )
    return 1 if args.require_pass and not report["acceptance_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
