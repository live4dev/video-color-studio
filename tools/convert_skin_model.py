#!/usr/bin/env python3
"""Reproduce the bundled skin model from Google's versioned TFLite weights.

Developer-only tool; neither TensorFlow nor this downloader is used by the app.
Conversion environment (Python 3.11): tensorflow==2.15.1 tf2onnx==1.16.1
numpy==1.26.4 onnx==1.17.0 onnxruntime==1.23.2 protobuf==3.20.3.
Pass --download explicitly to acquire the source, or supply a local verified file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
)
SOURCE_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
ONNX_SHA256 = "TO_BE_VERIFIED"
MODEL_NAME = "selfie_multiclass_256x256.onnx"
MODEL_DIRECTORY = ROOT / "src/freedive_color/assets/models"
VERSIONS = {
    "tensorflow": "2.15.1",
    "tf2onnx": "1.16.1",
    "numpy": "1.26.4",
    "onnx": "1.17.0",
    "onnxruntime": "1.23.2",
    "protobuf": "3.20.3",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source: Path) -> None:
    if not source.is_file() or sha256(source) != SOURCE_SHA256:
        raise ValueError("Official SelfieMulticlass v1 source SHA-256 mismatch")


def acquire_source(source: Path) -> None:
    if source.exists():
        verify_source(source)
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skin-download-", dir=source.parent) as work:
        temporary = Path(work) / "model.tflite"
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            temporary.write_bytes(response.read(32 * 1024 * 1024))
        verify_source(temporary)
        temporary.replace(source)


def check_environment() -> None:
    from importlib.metadata import version

    if sys.version_info[:2] != (3, 11):
        raise ValueError("Use the isolated Python 3.11 conversion environment")
    actual = {name: version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError(f"Conversion dependencies differ from pinned versions: {actual}")


def parity(source: Path, model: Path) -> dict:
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    reference = tf.lite.Interpreter(model_path=str(source), num_threads=2)
    reference.allocate_tensors()
    input_info, output_info = reference.get_input_details()[0], reference.get_output_details()[0]
    rng = np.random.default_rng(31)
    samples = [
        np.zeros((1, 256, 256, 3), np.float32),
        np.ones((1, 256, 256, 3), np.float32),
        rng.random((1, 256, 256, 3), dtype=np.float32),
    ]
    bars = np.zeros((1, 256, 256, 3), np.float32)
    for index, rgb in enumerate(
        ((0.7, 0.5, 0.4), (0.1, 0.4, 0.5), (0.1, 0.1, 0.1), (0.8, 0.1, 0.1))
    ):
        bars[:, :, index * 64 : (index + 1) * 64] = rgb
    samples.append(bars)
    rows = []
    for index, sample in enumerate(samples):
        reference.set_tensor(input_info["index"], sample)
        reference.invoke()
        expected = reference.get_tensor(output_info["index"])
        actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
        difference = float(np.max(np.abs(expected - actual)))
        agreement = float(np.mean(expected.argmax(axis=-1) == actual.argmax(axis=-1)))
        if not np.isfinite(actual).all() or difference > 0.002 or agreement < 0.999:
            raise ValueError(f"TFLite/ONNX numerical parity failed on synthetic input {index}")
        rows.append(
            {"synthetic_input": index, "max_abs_error": difference, "argmax_agreement": agreement}
        )
    return {
        "reference": "official TFLite v1 via TensorFlow Lite CPU",
        "scope": "numerical conversion parity; not semantic or underwater accuracy",
        "samples": rows,
        "input": {
            "name": session.get_inputs()[0].name,
            "shape": session.get_inputs()[0].shape,
            "dtype": "float32",
            "normalization": "RGB [0,1] NHWC",
        },
        "output": {
            "name": session.get_outputs()[0].name,
            "shape": session.get_outputs()[0].shape,
            "kind": "logits",
            "activation": "softmax over final class axis",
            "classes": ["background", "hair", "body-skin", "face-skin", "clothes", "others"],
        },
    }


def convert(source: Path, output: Path) -> dict:
    verify_source(source)
    check_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skin-convert-", dir=output.parent) as work:
        temporary = Path(work) / MODEL_NAME
        environment = {
            **os.environ,
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_NUM_INTEROP_THREADS": "2",
            "TF_NUM_INTRAOP_THREADS": "2",
            "PYTHONHASHSEED": "0",
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--tflite",
                str(source),
                "--opset",
                "17",
                "--output",
                str(temporary),
            ],
            env=environment,
            check=True,
        )
        evidence = parity(source, temporary)
        digest = sha256(temporary)
        if ONNX_SHA256 != "TO_BE_VERIFIED" and digest != ONNX_SHA256:
            raise ValueError("Converted ONNX differs from the reproducible bundled model")
        if output.exists():
            if sha256(output) != digest:
                raise FileExistsError("Output exists with different content; choose a new path")
        else:
            temporary.replace(output)
    return {
        "model_id": "selfie-multiclass-v1",
        "name": "Google MediaPipe SelfieMulticlass 256x256 float32 v1",
        "file": MODEL_NAME,
        "bytes": output.stat().st_size,
        "onnx_sha256": digest,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "license": "Apache-2.0",
        "license_file": "SELFIE_MULTICLASS_LICENSE.txt",
        "model_card_url": "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf",
        "attribution": "Google LLC; original weights unchanged, TFLite converted to ONNX",
        "conversion": {
            "python": "3.11",
            "dependencies": VERSIONS,
            "opset": 17,
            "tool": "tools/convert_skin_model.py",
        },
        "runtime": (
            "ONNX Runtime 1.23.2 CPU; bundled weights; telemetry disabled; no runtime downloads"
        ),
        "limitations": (
            "General selfie/full-body segmentation, not trained specifically for underwater skin; "
            "probability scores are not measured accuracy."
        ),
        **evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MODEL_DIRECTORY / MODEL_NAME)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    if arguments.download:
        acquire_source(arguments.source)
    report = convert(arguments.source, arguments.output)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
