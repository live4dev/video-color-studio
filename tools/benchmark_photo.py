"""One large synthetic photo, recorded timings and peak RSS; run separately from quick tests."""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from freedive_color.core.image_pipeline import export_photo, make_preview
from freedive_color.core.media_io import encode_photo
from freedive_color.core.presets import get_preset


def main() -> None:
    directory = Path("artifacts/benchmark")
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "24mp.png"
    target = directory / "24mp_natural.png"
    pixels = np.empty((4000, 6000, 3), np.float32)
    gradient = np.linspace(0, 1, 4000, dtype=np.float32)[:, None]
    pixels[..., 0] = 0.05 + gradient * 0.15
    pixels[..., 1] = 0.2 + gradient * 0.4
    pixels[..., 2] = 0.3 + gradient * 0.5
    encode_photo(source, pixels, None, b"", "PNG")
    del pixels
    settings = get_preset("natural")
    start = time.perf_counter()
    preview = make_preview(source, settings)
    preview_seconds = time.perf_counter() - start
    report = export_photo(source, target, settings, analysis=preview.analysis)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "dimensions": [6000, 4000],
        "preview_seconds": preview_seconds,
        "export_seconds": report["metrics"]["seconds"],
        "peak_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
    }
    (directory / "benchmark.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
