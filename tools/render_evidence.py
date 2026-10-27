"""Reproducible real-photo evidence, separate from clean single-photo application exports."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from freedive_color.core.image_pipeline import export_photo
from freedive_color.core.media_io import encode_photo, load_photo
from freedive_color.core.presets import PRESETS
from freedive_color.core.project import sha256_file


def render(source: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    original = load_photo(source)
    digest = sha256_file(source)
    encode_photo(directory / "before.png", original.rgb, original.alpha, original.exif_bytes, "PNG")
    reports = []
    for name, settings in PRESETS.items():
        reports.append(export_photo(source, directory / f"{name}.png", settings))
    # This labeled comparison is QA evidence only, never an application export format.
    before = Image.open(directory / "before.png").convert("RGB")
    after = Image.open(directory / "natural.png").convert("RGB")
    before.thumbnail((600, 720), Image.Resampling.LANCZOS)
    after = after.resize(before.size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (before.width * 2 + 18, before.height + 42), "#20262b")
    panel.paste(before, (0, 42))
    panel.paste(after, (before.width + 18, 42))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=17)
    draw.text((14, 11), "Before / original", fill="#dbe3e7", font=font)
    draw.text((before.width + 32, 11), "After / Natural", fill="#dbe3e7", font=font)
    panel.save(directory / "comparison.jpg", quality=95)
    record = {
        "source_unchanged": sha256_file(source) == digest,
        "source_sha256": digest,
        "presets": reports,
        "source_credit": "Andy Moreno, Freediver Moalboal, public domain (Wikimedia Commons)",
        "mean_absolute_change_natural": float(
            np.mean(np.abs(load_photo(directory / "natural.png").rgb - original.rgb))
        ),
    }
    (directory / "evidence.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_unchanged": record["source_unchanged"],
                "comparison": str((directory / "comparison.jpg").resolve()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/freediver_moalboal.jpg"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    render(args.input, args.output)
