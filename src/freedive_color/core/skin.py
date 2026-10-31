"""Offline semantic visible-skin segmentation; never edits or reconstructs image pixels.

Only the verified, bundled Google SelfieMulticlass weights are loaded. TensorFlow,
MediaPipe Tasks, downloaders and accelerator providers are not used at runtime.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .analyze import validate_rgb
from .errors import StudioError

SKIN_MODEL_ID = "selfie-multiclass-v1"
SKIN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "assets/models/selfie_multiclass_256x256.onnx"
)
# Set by the reproducible conversion, not by a mutable runtime download manifest.
SKIN_MODEL_SHA256 = "cd2dff0e466a1ce493f2f9d8a9452e2dfbe500695ae4cf77f8d3a57c089c6856"


@dataclass(frozen=True)
class SkinPrediction:
    skin: np.ndarray
    person: np.ndarray
    exclusion: np.ndarray
    model_sha256: str = SKIN_MODEL_SHA256
    diagnostics: dict = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        for array in (self.skin, self.person, self.exclusion):
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or not np.isfinite(array).all()
                or np.any(array < 0)
                or np.any(array > 1)
            ):
                raise StudioError("Некорректная семантическая карта кожи.")


def _regions(hint: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounded person crops with context, including detached hands/limbs nearby."""
    height, width = hint.shape
    binary = (hint > 0.25).astype(np.uint8)
    if not binary.any():
        return []
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(binary)
    minimum = max(8, hint.size * 0.0005)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if area < minimum:
            continue
        margin_x, margin_y = max(8, round(w * 0.2)), max(8, round(h * 0.2))
        boxes.append(
            (
                max(0, x - margin_x),
                max(0, y - margin_y),
                min(width, x + w + margin_x),
                min(height, y + h + margin_y),
            )
        )
    # Merge overlapping expanded regions so disconnected limbs retain body context.
    merged = []
    while boxes:
        box = boxes.pop()
        changed = True
        while changed:
            changed = False
            remaining = []
            for other in boxes:
                if (
                    box[0] < other[2]
                    and box[2] > other[0]
                    and box[1] < other[3]
                    and box[3] > other[1]
                ):
                    box = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                else:
                    remaining.append(other)
            boxes = remaining
        merged.append(box)
    merged.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return [box for box in merged[:4] if (box[2] - box[0]) * (box[3] - box[1]) < hint.size * 0.85]


class SelfieSkinSegmenter:
    """CPU inference over all four orientations of full-frame and contextual person crops."""

    def __init__(self, model_path: Path = SKIN_MODEL_PATH):
        if not model_path.is_file():
            raise StudioError("Локальная модель кожи отсутствует. Используйте кисть маски.")
        model = model_path.read_bytes()
        if hashlib.sha256(model).hexdigest() != SKIN_MODEL_SHA256:
            raise StudioError("Контрольная сумма модели кожи не совпала. Используйте кисть.")
        try:
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
        except (ImportError, OSError) as exc:
            raise StudioError("CPU-модель кожи недоступна. Используйте кисть маски.") from exc
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
        try:
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.use_deterministic_compute = True
            self.session = ort.InferenceSession(
                model, sess_options=options, providers=["CPUExecutionProvider"]
            )
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (
                len(inputs) != 1
                or inputs[0].shape != [1, 256, 256, 3]
                or inputs[0].type != "tensor(float)"
                or len(outputs) != 1
                or outputs[0].shape != [1, 256, 256, 6]
            ):
                raise ValueError("Unexpected skin model signature")
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
        except self._runtime_errors as exc:
            raise StudioError("CPU-модель кожи недоступна. Используйте кисть маски.") from exc
        self.lock = Lock()

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        # Verified TFLite metadata: mean=127.5, std=127.5 for uint8 RGB, i.e. [-1, 1].
        # Preserve body proportions on narrow whole-body crops. Padding is analysis-only
        # and is removed from the probability map, never from or added to source media.
        height, width = rgb.shape[:2]
        scale = 256 / max(height, width)
        size = max(1, round(width * scale)), max(1, round(height * scale))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        top, left = (256 - size[1]) // 2, (256 - size[0]) // 2
        square = np.zeros((256, 256, 3), np.float32)
        square[top : top + size[1], left : left + size[0]] = small
        normalized = square * 2 - 1
        try:
            result = self.session.run(
                [self.output_name],
                {self.input_name: np.ascontiguousarray(normalized[None], np.float32)},
            )[0][0]
        except self._runtime_errors as exc:
            raise StudioError("Не удалось выполнить локальную модель кожи.") from exc
        if result.shape != (256, 256, 6) or not np.isfinite(result).all():
            raise StudioError("Модель кожи вернула некорректные значения.")
        # The converted graph emits logits. Also accept normalized probabilities in tests
        # and equivalent conversion graphs without applying softmax twice.
        if (
            result.min() < 0
            or result.max() > 1
            or not np.allclose(result.sum(axis=2), 1, atol=1e-4)
        ):
            result = np.exp(result - result.max(axis=2, keepdims=True))
            result /= result.sum(axis=2, keepdims=True)
        result = result[top : top + size[1], left : left + size[0]]
        return cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

    @staticmethod
    def _agreement(person: np.ndarray, hint: np.ndarray) -> float:
        if np.any(hint > 0.25):
            return 2 * float(np.sum(person * hint)) / max(float(np.sum(person + hint)), 1e-5)
        selected = person > 0.8
        coverage = float(selected.mean())
        score = float(person[selected].mean()) if selected.any() else 0.0
        return score * min(1, coverage / 0.01) * min(1, (1 - coverage) / 0.15)

    def _oriented(self, rgb: np.ndarray, hint: np.ndarray) -> tuple[np.ndarray, int, float]:
        best, best_score, best_rotation = None, -1.0, 0
        for rotation in (0, 1, 2, 3):
            probabilities = np.rot90(self._predict(np.rot90(rgb, rotation)), -rotation)
            probabilities = cv2.resize(
                probabilities, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            person = 1 - probabilities[..., 0]
            score = self._agreement(person, hint)
            if score > best_score:
                best, best_score, best_rotation = probabilities, score, rotation
        return best, best_rotation, best_score

    def _refine_regions(self, rgb, hint, probabilities, boxes, minimum_agreement):
        shape, size = rgb.shape[:2], (hint.shape[1], hint.shape[0])
        orientations, scores, accepted = [], [], 0
        for x0, y0, x1, y1 in boxes:
            # Locate on the cheap proxy, but obtain zoom detail from original pixels.
            sx0, sx1 = round(x0 * shape[1] / size[0]), round(x1 * shape[1] / size[0])
            sy0, sy1 = round(y0 * shape[0] / size[1]), round(y1 * shape[0] / size[1])
            detail = rgb[sy0:sy1, sx0:sx1]
            scale = min(1, 512 / max(detail.shape[:2]))
            detail_size = (
                max(1, round(detail.shape[1] * scale)),
                max(1, round(detail.shape[0] * scale)),
            )
            detail = cv2.resize(detail, detail_size, interpolation=cv2.INTER_AREA)
            detail_hint = cv2.resize(
                hint[y0:y1, x0:x1], detail_size, interpolation=cv2.INTER_LINEAR
            )
            crop, rotation, score = self._oriented(detail, detail_hint)
            crop = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            orientations.append(rotation)
            scores.append(score)
            original = probabilities[y0:y1, x0:x1]
            baseline = self._agreement(1 - original[..., 0], hint[y0:y1, x0:x1])
            if score < max(minimum_agreement, baseline - 0.05):
                continue
            accepted += 1
            height, width = crop.shape[:2]
            yy, xx = np.mgrid[:height, :width]
            border = np.minimum.reduce((xx + 1, width - xx, yy + 1, height - yy))
            weight = np.clip(border / max(2, min(width, height) * 0.1), 0, 1)[..., None]
            probabilities[y0:y1, x0:x1] = original * (1 - weight) + crop * weight
        return accepted, orientations, scores

    def infer(
        self, rgb: np.ndarray, person_probability: np.ndarray | None = None
    ) -> SkinPrediction:
        validate_rgb(rgb)
        shape = rgb.shape[:2]
        if person_probability is not None and (
            person_probability.shape != shape or not np.isfinite(person_probability).all()
        ):
            raise StudioError("Некорректная подсказка области человека для модели кожи.")
        scale = min(1, 512 / max(shape))
        size = max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale))
        proxy = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        hint = (
            cv2.resize(person_probability.astype(np.float32), size, interpolation=cv2.INTER_AREA)
            if person_probability is not None
            else np.zeros(proxy.shape[:2], np.float32)
        )
        hint = np.clip(hint, 0, 1)
        boxes = _regions(hint)
        with self.lock:
            probabilities, rotation, score = self._oriented(proxy, hint)
            orientations = [rotation]
            scores = [score]
            accepted_crops, turns, agreements = self._refine_regions(
                rgb, hint, probabilities, boxes, 0.15
            )
            orientations.extend(turns)
            scores.extend(agreements)
        skin = probabilities[..., 2] + probabilities[..., 3]
        competitors = probabilities[..., 1] + probabilities[..., 4] + probabilities[..., 5]
        # Preserve competing class evidence even when skin narrowly wins. This allows
        # conservative rejection of ambiguous diving-mask/hair boundaries without RGB vetoes.
        exclusion = competitors.astype(np.float32)
        arrays = [
            np.clip(cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR), 0, 1)
            for array in (skin, 1 - probabilities[..., 0], exclusion)
        ]
        return SkinPrediction(
            *arrays,
            model_sha256=SKIN_MODEL_SHA256,
            diagnostics={
                "model_id": SKIN_MODEL_ID,
                "provider": "CPUExecutionProvider",
                "telemetry": False,
                "analysis_edge": max(proxy.shape[:2]),
                "input_geometry": "aspect-preserving-black-letterbox",
                "inference_calls": 4 * (1 + len(boxes)),
                "crop_count": len(boxes),
                "accepted_crops": accepted_crops,
                "selected_quarter_turns": orientations,
                "person_agreement": [round(value, 6) for value in scores],
                "score_meaning": "uncalibrated semantic probability; not measured accuracy",
            },
        )


_skin_segmenter: SelfieSkinSegmenter | None = None
_skin_lock = Lock()


def infer_skin(rgb: np.ndarray, person_probability: np.ndarray | None = None) -> SkinPrediction:
    global _skin_segmenter
    with _skin_lock:
        if _skin_segmenter is None:
            _skin_segmenter = SelfieSkinSegmenter()
    return _skin_segmenter.infer(rgb, person_probability)
