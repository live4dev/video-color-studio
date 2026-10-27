"""Milestone B evidence: clean photos, separately named QA overlays and a 20-file series.

The real image is a licensed public-domain fixture. Series inputs are explicitly controlled
exposure variants of that same photograph, not a claim about 20 independent dive scenes.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from freedive_color.core.analyze import linear_to_srgb, srgb_to_linear
from freedive_color.core.batch import export_batch
from freedive_color.core.color_pipeline import LUMA
from freedive_color.core.image_pipeline import export_photo
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import BrushStroke, MaskEdits, build_masks
from freedive_color.core.media_io import encode_photo, load_photo
from freedive_color.core.presets import get_preset
from freedive_color.core.project import sha256_file
from freedive_color.core.reference import create_reference
from freedive_color.core.session import ExportOptions, Session, SessionItem, save_session

ROOT = Path(__file__).resolve().parents[1]


def _compare(before: np.ndarray, after: np.ndarray, destination: Path, title: str) -> None:
    def small(pixels):
        image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
        image.thumbnail((640, 720), Image.Resampling.LANCZOS)
        return image

    left, right = small(before), small(after)
    right = right.resize(left.size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (left.width * 2 + 16, left.height + 42), "#20262b")
    canvas.paste(left, (0, 42))
    canvas.paste(right, (left.width + 16, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((12, 11), "Original", font=font, fill="#dbe3e7")
    draw.text((left.width + 28, 11), title, font=font, fill="#dbe3e7")
    canvas.save(destination, quality=95)


def _overlay(rgb: np.ndarray, masks) -> np.ndarray:
    result = rgb.copy()
    for name, color in (("water", [0.1, 0.85, 1]), ("skin", [1, 0.55, 0.25])):
        alpha = getattr(masks, name)[..., None] * 0.45
        result = result * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 1).astype(np.float32)


def render(source: Path, directory: Path) -> dict:
    directory = directory.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(
            "Choose a new output directory: evidence never overwrites an existing run."
        )
    directory.mkdir(parents=True, exist_ok=True)
    original = load_photo(source)
    source_digest = sha256_file(source)
    settings = get_preset("natural")
    local = LocalSettings(
        skin_warmth=0.3, skin_strength=0.55, water_hue=0.18, water_saturation=0.12
    )
    automatic = build_masks(original.rgb)
    encode_photo(
        directory / "real_before.png", original.rgb, original.alpha, original.exif_bytes, "PNG"
    )
    auto_report = export_photo(
        source, directory / "real_auto.png", settings, local_settings=local.to_dict()
    )
    auto_result = load_photo(directory / "real_auto.png")
    encode_photo(
        directory / "qa_auto_masks.png", _overlay(original.rgb, automatic), None, b"", "PNG"
    )
    _compare(
        original.rgb,
        auto_result.rgb,
        directory / "qa_real_auto_comparison.jpg",
        "Natural + local masks",
    )
    # Explicit refinements: a conservative whole-person region protects the blue suit/fins,
    # and a small hand stroke avoids the face mask. The automatic output remains separate
    # evidence, because confident segmentation can still miss blue underwater apparel.
    edits = MaskEdits(
        person_box=(0.31, 0.10, 0.84, 0.81),
        strokes=(
            BrushStroke(
                target="skin",
                operation="add",
                points=((0.362, 0.758), (0.377, 0.731), (0.389, 0.709)),
                radius=0.012,
            ),
        ),
    )
    manual_masks = build_masks(original.rgb, edits)
    manual_report = export_photo(
        source,
        directory / "real_manual.png",
        settings,
        local_settings=local.to_dict(),
        mask_edits=edits.to_dict(),
    )
    encode_photo(
        directory / "qa_manual_masks.png", _overlay(original.rgb, manual_masks), None, b"", "PNG"
    )
    manual_result = load_photo(directory / "real_manual.png")
    _compare(
        original.rgb,
        manual_result.rgb,
        directory / "qa_real_manual_comparison.jpg",
        "Local + person region / hand mask",
    )

    height, width = 360, 640
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    board = np.empty((height, width, 3), np.float32)
    board[..., 0] = 0.06 + y * 0.05
    board[..., 1] = 0.45 - y * 0.18
    board[..., 2] = 0.72 - y * 0.2
    board[:110] = [0.5, 0.7, 0.9]
    board[210:270, 80:160] = [0.65, 0.48, 0.36]
    board[210:270, 180:260] = [0.25, 0.18, 0.13]
    board[210:270, 280:360] = [0.02, 0.02, 0.02]
    board[210:270, 380:460] = [0.8, 0.05, 0.55]
    board_path = directory / "split_target_before.png"
    encode_photo(board_path, board, None, b"", "PNG")
    boundary = MaskEdits(waterline=((0, 0.35), (0.35, 0.32), (0.7, 0.36), (1, 0.34)))
    split_local = replace(local, skin_strength=0, above_exposure=0)
    split_report = export_photo(
        board_path,
        directory / "split_target_after.png",
        get_preset("vivid"),
        local_settings=split_local.to_dict(),
        mask_edits=boundary.to_dict(),
    )
    board_original = load_photo(board_path).rgb
    board_after = load_photo(directory / "split_target_after.png").rgb
    air_max_change = float(np.max(np.abs(board_after[:70] - board_original[:70])))
    if air_max_change > 1.1 / 255:
        raise AssertionError("The air region changed despite zero above-water exposure")
    _compare(
        board_original,
        board_after,
        directory / "qa_split_comparison.jpg",
        "Separate air / water grade",
    )

    # Twenty deterministic variants preserve the real fixture's objects and geometry. Resizing
    # here constructs small QA inputs; the application exports each input at that exact size.
    series_inputs = directory / "series_inputs"
    series_inputs.mkdir()
    image = Image.fromarray(np.rint(original.rgb * 255).astype(np.uint8))
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    reduced = np.asarray(image).astype(np.float32) / 255
    reference = create_reference(
        auto_result.rgb,
        automatic,
        source_name="Natural local reference",
        source_sha256=sha256_file(directory / "real_auto.png"),
    )
    items = []
    hashes = {}
    input_luminance = []
    for index, exposure in enumerate(np.linspace(-0.45, 0.45, 20)):
        variant = np.clip(linear_to_srgb(srgb_to_linear(reduced) * 2**exposure), 0, 1)
        path = series_inputs / f"dive_{index + 1:02d}.png"
        encode_photo(path, variant, None, b"", "PNG")
        digest = sha256_file(path)
        hashes[str(path)] = digest
        input_luminance.append(float(np.mean(load_photo(path).rgb @ LUMA)))
        items.append(
            SessionItem(
                path=path,
                settings=settings,
                local_settings=local.to_dict(),
                mask_edits=edits.to_dict(),
                source_sha256=digest,
            )
        )
    outputs = directory / "series_outputs"
    batch = export_batch(items, outputs, ExportOptions(suffix=".png"), reference=reference)
    image_paths = sorted(outputs.glob("*.png"))
    unchanged = sha256_file(source) == source_digest and all(
        sha256_file(Path(path)) == digest for path, digest in hashes.items()
    )
    if batch.completed != 20 or batch.failed or len(image_paths) != 20 or not unchanged:
        raise AssertionError("The twenty-file series failed or a source changed")
    dimensions = [load_photo(path).rgb.shape[:2] for path in image_paths]
    if any(size != reduced.shape[:2] for size in dimensions):
        raise AssertionError("Series export changed an input's dimensions")
    output_luminance = [float(np.mean(load_photo(path).rgb @ LUMA)) for path in image_paths]
    save_session(
        directory / "series.fcsproj",
        Session(
            items=tuple(items),
            reference_path=directory / "real_auto.png",
            reference_data=reference.to_dict(),
            output_directory=outputs,
            selected_index=0,
        ),
    )
    record = {
        "milestone": "B",
        "source_unchanged": unchanged,
        "source": {
            "path": str(source.resolve()),
            "sha256": source_digest,
            "dimensions": [original.width, original.height],
            "credit": "Andy Moreno, Freediver Moalboal, public-domain dedication",
            "url": "https://commons.wikimedia.org/wiki/File:Freediver_Moalboal.jpg",
        },
        "automatic_masks": automatic.to_metadata(),
        "manual_masks": manual_masks.to_metadata(),
        "real_auto_export": auto_report,
        "real_manual_export": manual_report,
        "split_level": {"air_max_absolute_change": air_max_change, "export": split_report},
        "batch": batch.to_dict(),
        "series_source_hashes": hashes,
        "series_luminance": {"inputs": input_luminance, "outputs": output_luminance},
        "limitations": [
            "Twenty series inputs are controlled exposure variants of one public-domain photo.",
            "Color targets and one photo do not establish segmentation quality on all skin tones.",
            "Auto segmentation can miss blue apparel; manual result uses a person region.",
            "QA overlays are separate; clean exports contain no labels or collages.",
        ],
    }
    (directory / "evidence.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "tests/fixtures/freediver_moalboal.jpg"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/milestone-b")
    args = parser.parse_args()
    report = render(args.input, args.output)
    print(
        json.dumps(
            {
                "source_unchanged": report["source_unchanged"],
                "completed": report["batch"]["completed"],
                "failed": report["batch"]["failed"],
                "evidence": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
