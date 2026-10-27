"""Deterministic synthetic color/geometry targets. No downloaded models or generated people."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:360, 0:640].astype(np.float32)
    water = np.stack(
        (
            0.05 + 0.05 * (1 - y / 360),
            0.3 + 0.3 * (1 - y / 360),
            0.4 + 0.4 * (1 - y / 360) + np.zeros_like(x),
        ),
        axis=2,
    )
    fixtures = {
        "water_gradient": water,
        "water_constant": np.full_like(water, [0.08, 0.4, 0.6]),
        "missing_red": water * [0, 1, 1],
        "green_cast": water * [0.55, 1.3, 0.8],
    }
    board = water.copy()
    colors = ([0.68, 0.5, 0.4], [0.25, 0.18, 0.13], [0.02, 0.02, 0.02], [0.8, 0.08, 0.55])
    for index, color in enumerate(colors):
        board[140:240, 50 + index * 145 : 150 + index * 145] = color
    fixtures["skin_and_equipment_swatches"] = board
    split = water.copy()
    split[:120] = [0.5, 0.7, 0.9]
    fixtures["split_color_board"] = split
    rays = water.copy()
    pattern = (np.sin(x / 18 + y / 35) > 0.97) & (y < 200)
    rays[pattern] *= 1.12
    fixtures["existing_light_pattern"] = rays
    for name, pixels in fixtures.items():
        Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(
            directory / f"{name}.png"
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "seed": "analytic gradients, no randomness",
                "license": "MIT, same as source code",
                "fixtures": list(fixtures),
                "limitation": "Color targets, not anatomical or semantic segmentation validation.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/fixtures"))
    generate(parser.parse_args().output)
