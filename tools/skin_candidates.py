"""Explicit, developer-only offline skin candidates; never downloads or selects a model.

The desktop default remains core.skin.infer_skin. These wrappers are only used by
the bounded evaluation CLI after an audited local ONNX conversion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from freedive_color.core.analyze import validate_rgb
from freedive_color.core.errors import StudioError
from freedive_color.core.skin import RawPredictionCache, SelfieSkinSegmenter, SkinPrediction

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "build/skin-candidates/onnx"
CANDIDATE_NAMES = ("selfie-v2", "schp-atr", "fcn-skin")
UPSTREAM = {
    "schp-atr": {
        "model_id": "schp-atr-20190830",
        "repository": "https://github.com/GoGoDuck912/Self-Correction-Human-Parsing",
        "commit": "eb84c432cc697f494d99662a05f2335eb2f26095",
        "checkpoint_url": "https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP",
        "checkpoint_download_url": "https://drive.usercontent.google.com/download?"
        "id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP&confirm=t",
        "checkpoint_sha256": "e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a",
        "checkpoint_bytes": 267445237,
        "source_zip_sha256": "d56663279f812e787598f6943e1ef3b1f1e12bf0b76e4a74e5691e809a51c500",
        "channels": 18,
    },
    "fcn-skin": {
        "model_id": "fcn-resnet101-skin-30",
        "repository": "https://github.com/WillBrennan/SemanticSegmentation",
        "commit": "7646661aa0e05f5c5b087f2aeda3faff42c34e00",
        "checkpoint_url": "https://media.githubusercontent.com/media/WillBrennan/"
        "SemanticSegmentation/7646661aa0e05f5c5b087f2aeda3faff42c34e00/"
        "pretrained/model_segmentation_skin_30.pth",
        "checkpoint_sha256": "a2739556d28995ad1ceb42dca83f292a45f0ecbae3f602f0d1b70f742b7821eb",
        "checkpoint_bytes": 217753996,
        "source_zip_sha256": "8f7b4b20e36c926cb1847441628118eb377d385dc267aa8d6f4941880741b321",
        "channels": 1,
    },
}
# Populated only by an independently verified conversion, not a runtime download.
ONNX_SHA256: dict[str, str] = {
    "fcn-skin": "adb8680d68faea1fda4729bf1c4335e99c0442ed0b8336e3e5369375d59b0ed9",
    "schp-atr": "fc33b8a8709427cce17434f1c99137bc7595ac6b4e6e2184fa08e5dd02f8b868",
}
INPUT_EDGE = 512


@dataclass(frozen=True)
class InputGeometry:
    shape: tuple[int, int]
    content: tuple[int, int, int, int] | None = None
    inverse_affine: np.ndarray | None = None


def prepare_candidate_input(name: str, rgb: np.ndarray) -> tuple[np.ndarray, InputGeometry]:
    """Normalize all source pixels; padding/rotation only affects analysis tensors."""
    validate_rgb(rgb)
    height, width = rgb.shape[:2]
    if name == "schp-atr":
        # Exact zero-rotation square affine from upstream bbox=(0,0,w-1,h-1).
        extent = max(height - 1, width - 1, 1)
        scale = (INPUT_EDGE - 1) / extent
        matrix = np.array(
            [
                [scale, 0, (INPUT_EDGE - 1 - scale * (width - 1)) / 2],
                [0, scale, (INPUT_EDGE - 1 - scale * (height - 1)) / 2],
            ],
            np.float64,
        )
        square = cv2.warpAffine(
            rgb[..., ::-1],
            matrix,
            (INPUT_EDGE, INPUT_EDGE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        geometry = InputGeometry((height, width), inverse_affine=cv2.invertAffineTransform(matrix))
        mean, std = [0.406, 0.456, 0.485], [0.225, 0.224, 0.229]
    elif name == "fcn-skin":
        scale = INPUT_EDGE / max(height, width)
        size = max(1, round(width * scale)), max(1, round(height * scale))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        top, left = (INPUT_EDGE - size[1]) // 2, (INPUT_EDGE - size[0]) // 2
        square = np.zeros((INPUT_EDGE, INPUT_EDGE, 3), np.float32)
        square[top : top + size[1], left : left + size[0]] = small
        geometry = InputGeometry((height, width), (left, top, left + size[0], top + size[1]))
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown candidate: {name}")
    tensor = (square - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], np.float32), geometry


def restore_candidate_logits(logits: np.ndarray, geometry: InputGeometry) -> np.ndarray:
    """Inverse map logits before class decisions, preserving all source edges."""
    if logits.ndim != 4 or logits.shape[0] != 1 or not np.isfinite(logits).all():
        raise StudioError("Некорректные logits исследовательской модели кожи.")
    maps = logits[0].transpose(1, 2, 0).astype(np.float32)
    if maps.shape[:2] != (INPUT_EDGE, INPUT_EDGE):
        maps = cv2.resize(maps, (INPUT_EDGE, INPUT_EDGE), interpolation=cv2.INTER_LINEAR)
    if maps.ndim == 2:
        maps = maps[..., None]
    height, width = geometry.shape
    if geometry.inverse_affine is not None:
        maps = cv2.warpAffine(
            maps,
            geometry.inverse_affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        left, top, right, bottom = geometry.content
        maps = cv2.resize(
            maps[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR
        )
    return maps[..., None] if maps.ndim == 2 else maps


def map_candidate_logits(name: str, logits: np.ndarray) -> np.ndarray:
    """Map exact upstream classes into the shared six-class mask contract."""
    result = np.zeros((*logits.shape[:2], 6), np.float32)
    if name == "fcn-skin":
        if logits.shape[2] != 1:
            raise StudioError("Ожидалась одна FCN-карта кожи.")
        skin = 1 / (1 + np.exp(-np.clip(logits[..., 0], -80, 80)))
        result[..., 0], result[..., 2] = 1 - skin, skin
    elif name == "schp-atr":
        if logits.shape[2] != 18:
            raise StudioError("Ожидались 18 классов SCHP-ATR.")
        probability = np.exp(logits - logits.max(axis=2, keepdims=True))
        probability /= probability.sum(axis=2, keepdims=True)
        for target, sources in enumerate(
            ((0,), (2,), (12, 13, 14, 15), (11,), (4, 5, 6, 7, 8, 9, 10, 17), (1, 3, 16))
        ):
            result[..., target] = probability[..., sources].sum(axis=2)
    else:
        raise ValueError(f"Unknown candidate: {name}")
    return result


class CandidateSegmenter(SelfieSkinSegmenter):
    """Shared four-orientation/local-consensus fusion with explicit candidate weights."""

    def __init__(self, name: str, model_dir: Path):
        import onnxruntime as ort
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            Fail,
            InvalidArgument,
            InvalidGraph,
            InvalidProtobuf,
            NoSuchFile,
            NotImplemented,
            RuntimeException,
        )

        self.name = name
        self._runtime_errors = (
            RuntimeError,
            ValueError,
            OSError,
            Fail,
            InvalidArgument,
            InvalidGraph,
            InvalidProtobuf,
            NoSuchFile,
            NotImplemented,
            RuntimeException,
        )
        if ort.__version__ != "1.23.2":
            raise StudioError("Кандидаты проверяются через закреплённый CPU ONNX Runtime 1.23.2.")
        source = UPSTREAM[name]
        model_path = Path(model_dir) / f"{name}.onnx"
        manifest_path = model_path.with_suffix(".json")
        if not model_path.is_file() or not manifest_path.is_file():
            raise StudioError(
                "Исследовательская ONNX-модель отсутствует; выполните явную конвертацию."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = model_path.read_bytes()
        digest = hashlib.sha256(model).hexdigest()
        if digest != ONNX_SHA256.get(name) or manifest.get("onnx_sha256") != digest:
            raise StudioError("Контрольная сумма исследовательской модели кожи не совпала.")
        for key in ("commit", "checkpoint_sha256"):
            if manifest.get(key) != source[key]:
                raise StudioError("Не совпало происхождение исследовательской модели кожи.")
        if manifest.get("parity_passed") is not True:
            raise StudioError("Численная проверка конвертации не пройдена.")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.use_deterministic_compute = True
        self.session = ort.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise StudioError("Исследовательская модель должна использовать только CPU.")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != [1, 3, INPUT_EDGE, INPUT_EDGE]
            or len(outputs) != 1
            or outputs[0].shape != [1, source["channels"], INPUT_EDGE, INPUT_EDGE]
        ):
            raise StudioError("Не совпала сигнатура исследовательской модели кожи.")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.model_id, self.model_sha256 = source["model_id"], digest
        self.manifest = manifest
        self.lock, self._cache = Lock(), RawPredictionCache()

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        tensor, geometry = prepare_candidate_input(self.name, rgb)
        try:
            logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except self._runtime_errors as exc:
            raise StudioError("Ошибка CPU inference исследовательской модели кожи.") from exc
        return map_candidate_logits(self.name, restore_candidate_logits(logits, geometry))

    def infer(
        self, rgb: np.ndarray, person_probability: np.ndarray | None = None
    ) -> SkinPrediction:
        prediction = super().infer(rgb, person_probability)
        diagnostics = prediction.diagnostics.copy()
        diagnostics.update(
            candidate=self.name,
            input_edge=INPUT_EDGE,
            source_commit=self.manifest["commit"],
            checkpoint_sha256=self.manifest["checkpoint_sha256"],
            input_geometry="upstream-square-affine"
            if self.name == "schp-atr"
            else "aspect-preserving-black-letterbox",
            developer_only=True,
            conversion_parity_passed=True,
            missing_semantic_classes=(
                ["hair", "clothes", "accessories"] if self.name == "fcn-skin" else []
            ),
        )
        models = (
            {
                "id": self.model_id,
                "sha256": self.model_sha256,
                "checkpoint_sha256": self.manifest["checkpoint_sha256"],
                "source_commit": self.manifest["commit"],
            },
        )
        return replace(prediction, diagnostics=diagnostics, models=models)


def make_predictor(name: str, model_dir: Path | str = DEFAULT_MODEL_DIR):
    """Return a metadata-preserving callable; this function never performs network I/O."""
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown candidate {name}; expected {CANDIDATE_NAMES}")
    if name == "selfie-v2":
        return SelfieSkinSegmenter().infer
    return CandidateSegmenter(name, Path(model_dir)).infer
