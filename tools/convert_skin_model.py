#!/usr/bin/env python3
"""Reproduce the bundled skin model from Google's versioned TFLite weights.

Developer-only tool; neither TensorFlow nor this downloader is used by the app.
Conversion environment (Python 3.11): tensorflow==2.15.1 tf2onnx==1.16.1
numpy==1.26.4 onnx==1.17.0 onnxruntime==1.23.2 protobuf==3.20.3.
Pass --download explicitly to acquire the source, or supply a local verified file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
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


def read_normalization(source: Path) -> dict:
    """Read the authoritative NormalizationOptions from the official TFLite bytes.

    TFLite and metadata use separate FlatBuffer roots. Reading only these documented
    schema fields avoids importing the MediaPipe framework or assuming a model card's
    normalization. The source checksum is verified before any metadata is interpreted.
    """
    from flatbuffers.table import Table

    verify_source(source)
    data = source.read_bytes()

    def integer(buffer, offset):
        return struct.unpack_from("<I", buffer, offset)[0]

    def children(table, field):
        offset = table.Offset(4 + 2 * field)
        if not offset:
            return []
        start = table.Vector(offset)
        return [
            Table(table.Bytes, start + index * 4 + integer(table.Bytes, start + index * 4))
            for index in range(table.VectorLen(offset))
        ]

    def text(table, field):
        offset = table.Offset(4 + 2 * field)
        return table.String(table.Pos + offset).decode("utf-8") if offset else None

    def floats(table, field):
        offset = table.Offset(4 + 2 * field)
        return (
            list(
                struct.unpack_from(
                    "<" + "f" * table.VectorLen(offset), table.Bytes, table.Vector(offset)
                )
            )
            if offset
            else []
        )

    model = Table(data, integer(data, 0))
    buffers = children(model, 4)
    metadata = next(row for row in children(model, 6) if text(row, 0) == "TFLITE_METADATA")
    index = integer(data, metadata.Pos + metadata.Offset(6))
    offset = buffers[index].Offset(4)
    begin, length = buffers[index].Vector(offset), buffers[index].VectorLen(offset)
    metadata_bytes = bytes(data[begin : begin + length])
    description = Table(metadata_bytes, integer(metadata_bytes, 0))
    tensor = children(children(description, 3)[0], 2)[0]
    for process in children(tensor, 4):
        # ProcessUnitOptions enum: 1 is NormalizationOptions.
        type_offset = process.Offset(4)
        if not type_offset or process.Bytes[process.Pos + type_offset] != 1:
            continue
        option_offset = process.Pos + process.Offset(6)
        options = Table(process.Bytes, option_offset + integer(process.Bytes, option_offset))
        result = {
            "input_name": text(tensor, 0),
            "mean": floats(options, 0),
            "std": floats(options, 1),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }
        if result["mean"] != [127.5] or result["std"] != [127.5]:
            raise ValueError(f"Official model has unexpected normalization: {result}")
        return result
    raise ValueError("Official model has no NormalizationOptions metadata")


def canonicalize_model(path: Path) -> None:
    """Give an equivalent inference DAG stable names, order and protobuf bytes.

    tf2onnx's optimiser generates different internal identifiers/order on repeated
    runs. Content-addressing deterministic operators and constants removes that
    incidental variation. Public tensor names, values and output shapes stay fixed;
    identical pure operations/constants are shared. Numerical parity is checked on
    the resulting bytes, not only on the raw converter output.
    """
    import onnx
    from onnx import helper

    model = onnx.load(path)
    graph = model.graph
    if graph.sparse_initializer or model.functions or model.training_info:
        raise ValueError("Unexpected non-dense inference model; canonicalisation is not defined")
    public_inputs = {value.name for value in graph.input}
    public_outputs = [value.name for value in graph.output]
    names = {key: key for key in public_inputs}
    initializers = {}
    for original in graph.initializer:
        tensor = copy.deepcopy(original)
        tensor.name = ""
        tensor.doc_string = ""
        digest = hashlib.sha256(tensor.SerializeToString(deterministic=True)).hexdigest()
        name = "constant_" + digest
        names[original.name] = name
        tensor.name = name
        initializers[name] = tensor

    def normalized_node(original):
        node = copy.deepcopy(original)
        node.name = ""
        node.doc_string = ""
        del node.input[:]
        node.input.extend(names.get(name, name) for name in original.input)
        del node.output[:]
        node.output.extend(f"output_{index}" for index in range(len(original.output)))
        attributes = sorted(node.attribute, key=lambda item: item.name)
        del node.attribute[:]
        for attribute in attributes:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Unexpected control-flow subgraph in fixed skin model")
            if attribute.type == onnx.AttributeProto.TENSOR:
                attribute.t.name = ""
                attribute.t.doc_string = ""
            if attribute.type == onnx.AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    tensor.name = ""
                    tensor.doc_string = ""
            node.attribute.append(attribute)
        return node

    pending = list(graph.node)
    nodes, seen = [], set()
    while pending:
        ready = [
            node for node in pending if all(not value or value in names for value in node.input)
        ]
        if not ready:
            raise ValueError("Inference graph is cyclic or has an unresolved tensor")
        ranked = []
        for original in ready:
            if original.op_type.startswith("Random") or original.op_type in (
                "Bernoulli",
                "Dropout",
            ):
                raise ValueError("Stochastic operators must not be coalesced")
            node = normalized_node(original)
            key = node.SerializeToString(deterministic=True)
            ranked.append((key, original, node))
        for key, original, node in sorted(ranked, key=lambda row: row[0]):
            if (
                node.op_type == "Identity"
                and not node.domain
                and not node.attribute
                and len(original.input) == len(original.output) == 1
            ):
                names[original.output[0]] = names[original.input[0]]
                continue
            digest = hashlib.sha256(key).hexdigest()
            outputs = [f"value_{digest}_{index}" for index in range(len(original.output))]
            for previous, stable in zip(original.output, outputs, strict=True):
                names[previous] = stable
            if digest not in seen:
                node.name = "node_" + digest
                del node.output[:]
                node.output.extend(outputs)
                nodes.append(node)
                seen.add(digest)
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]
    for index, name in enumerate(public_outputs):
        nodes.append(
            helper.make_node("Identity", [names[name]], [name], name=f"public_output_{index}")
        )

    information = {}
    for original in graph.value_info:
        if (
            original.name not in names
            or original.name in public_inputs
            or original.name in public_outputs
        ):
            continue
        value = copy.deepcopy(original)
        value.name = names[original.name]
        value.doc_string = ""
        for axis, dimension in enumerate(value.type.tensor_type.shape.dim):
            if dimension.HasField("dim_param"):
                dimension.dim_param = f"unknown_dimension_{axis}"
        previous = information.get(value.name)
        if previous is None or value.SerializeToString(
            deterministic=True
        ) < previous.SerializeToString(deterministic=True):
            information[value.name] = value
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers[name] for name in sorted(initializers))
    del graph.value_info[:]
    graph.value_info.extend(information[name] for name in sorted(information))
    graph.name = "selfie_multiclass_v1"
    graph.doc_string = ""
    model.doc_string = ""
    del model.metadata_props[:]
    imports = sorted(model.opset_import, key=lambda item: (item.domain, item.version))
    del model.opset_import[:]
    model.opset_import.extend(imports)
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString(deterministic=True))


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
    normalization = read_normalization(source)
    input_info, output_info = reference.get_input_details()[0], reference.get_output_details()[0]
    rng = np.random.default_rng(31)
    samples = [
        np.zeros((1, 256, 256, 3), np.float32),
        np.ones((1, 256, 256, 3), np.float32),
        -np.ones((1, 256, 256, 3), np.float32),
        rng.random((1, 256, 256, 3), dtype=np.float32) * 2 - 1,
    ]
    bars = np.zeros((1, 256, 256, 3), np.float32)
    for index, rgb in enumerate(
        ((0.7, 0.5, 0.4), (0.1, 0.4, 0.5), (0.1, 0.1, 0.1), (0.8, 0.1, 0.1))
    ):
        bars[:, :, index * 64 : (index + 1) * 64] = rgb
    samples.append(bars * 2 - 1)
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
            "normalization": "RGB [-1,1] NHWC; normalized_rgb = rgb_float_0_1 * 2 - 1",
            "normalization_mean": normalization["mean"],
            "normalization_std": normalization["std"],
            "range": [-1, 1],
        },
        "source_normalization_metadata": normalization,
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
        canonicalize_model(temporary)
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
            "canonicalization": "content-addressed-inference-dag-v1",
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
