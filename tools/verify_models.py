"""Fail desktop packaging if a pinned offline model or its provenance is missing."""

import hashlib
import json
from pathlib import Path

try:
    from convert_skin_model import MODEL_NAME, ONNX_SHA256, SOURCE_SHA256
except ImportError:
    from tools.convert_skin_model import MODEL_NAME, ONNX_SHA256, SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]


def verify_models(directory: Path | None = None) -> dict[str, str]:
    directory = directory or ROOT / "src/freedive_color/assets/models"
    expected = {
        "human_segmentation_pphumanseg_2023mar.onnx": (
            "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24"
        ),
        MODEL_NAME: ONNX_SHA256,
    }
    for filename, digest in expected.items():
        path = directory / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundled offline model missing or modified: {filename}")
    metadata = json.loads(
        (directory / "SELFIE_MULTICLASS_METADATA.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("model_id") != "selfie-multiclass-v1"
        or metadata.get("onnx_sha256") != ONNX_SHA256
        or metadata.get("source_sha256") != SOURCE_SHA256
        or metadata.get("license") != "Apache-2.0"
        or not (directory / "SELFIE_MULTICLASS_LICENSE.txt").is_file()
    ):
        raise ValueError("Bundled skin model provenance/license is missing or inconsistent")
    return expected


if __name__ == "__main__":
    print(json.dumps(verify_models(), indent=2))
