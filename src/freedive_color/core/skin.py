"""Offline semantic visible-skin segmentation; never edits or reconstructs image pixels.

Only the verified, bundled Google SelfieMulticlass weights are loaded. TensorFlow,
MediaPipe Tasks, downloaders and accelerator providers are not used at runtime.
"""

import copy
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .analyze import validate_rgb
from .errors import StudioError

SKIN_MODEL_ID = "selfie-multiclass-v1"
SKIN_ALGORITHM_VERSION = "semantic-consensus-components-v2"
SKIN_CLASS_NAMES = ("background", "hair", "body_skin", "face_skin", "clothes", "accessories")
RAW_CACHE_BYTES = 64 * 1024 * 1024
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
    model_id: str = SKIN_MODEL_ID
    algorithm_version: str = SKIN_ALGORITHM_VERSION
    models: tuple[dict, ...] = ()
    # Raw maps remain on the <=512px analysis grid; only the three final maps expand.
    class_maps: dict[str, np.ndarray] = field(default_factory=dict)
    views: tuple["SkinView", ...] = ()
    consensus: np.ndarray | None = None

    def validate(self, shape: tuple[int, int]) -> None:
        arrays = (self.skin, self.person, self.exclusion)
        if self.consensus is not None:
            arrays += (self.consensus,)
        for array in arrays:
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or not np.isfinite(array).all()
                or np.any(array < 0)
                or np.any(array > 1)
            ):
                raise StudioError("Некорректная семантическая карта кожи.")


@dataclass(frozen=True)
class SkinView:
    """One inverse-mapped geometric observation on a bounded analysis grid."""

    probabilities: np.ndarray
    region: tuple[int, int, int, int]
    quarter_turns: int
    scale: float = 1.0
    source: str = "full-frame"

    def provenance(self) -> dict:
        return {
            "region": [int(value) for value in self.region],
            "quarter_turns": int(self.quarter_turns),
            "scale": float(self.scale),
            "source": self.source,
        }


def _prediction_bytes(prediction: SkinPrediction) -> int:
    seen = set()
    total = 0
    for array in (
        prediction.skin,
        prediction.person,
        prediction.exclusion,
        prediction.consensus,
        *prediction.class_maps.values(),
        *(view.probabilities for view in prediction.views),
    ):
        if isinstance(array, np.ndarray) and id(array) not in seen:
            total += array.nbytes
            seen.add(id(array))
    return total


class RawPredictionCache:
    """Byte-bounded LRU of raw proxy results, isolated from caller mutations."""

    def __init__(self, max_bytes: int = RAW_CACHE_BYTES):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self._entries = OrderedDict()

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return copy.deepcopy(entry[0])

    def put(self, key, prediction):
        size = _prediction_bytes(prediction)
        previous = self._entries.pop(key, None)
        if previous:
            self.bytes -= previous[1]
        if size > self.max_bytes:
            return
        while self._entries and self.bytes + size > self.max_bytes:
            _old_key, (_old, old_size) = self._entries.popitem(last=False)
            self.bytes -= old_size
        self._entries[key] = (copy.deepcopy(prediction), size)
        self.bytes += size

    def clear(self):
        self._entries.clear()
        self.bytes = 0


def fuse_skin_views(
    views: list[SkinView],
    shape: tuple[int, int],
    *,
    model_id: str = SKIN_MODEL_ID,
    model_sha256: str = SKIN_MODEL_SHA256,
    algorithm_version: str = SKIN_ALGORITHM_VERSION,
) -> SkinPrediction:
    """Fuse local semantic agreement; silhouette overlap never elects a winner.

    A pixel needs two distinct geometry votes. Strong contrary foreground votes
    veto the ambiguous location, so a crop cannot overwrite agreeing full
    views with its own isolated skin error. Background-only views
    cannot veto a well-supported limb found outside the coarse silhouette.
    """
    height, width = shape
    summed = np.zeros((*shape, 6), np.float32)
    weights = np.zeros(shape, np.float32)
    skin_summed = np.zeros_like(summed)
    positive = np.zeros(shape, np.uint8)
    negative = np.zeros(shape, np.uint8)
    geometry = set()
    for view in views:
        x0, y0, x1, y1 = view.region
        p = view.probabilities
        if (
            not 0 <= x0 < x1 <= width
            or not 0 <= y0 < y1 <= height
            or p.shape != (y1 - y0, x1 - x0, 6)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or np.any(p > 1)
            or view.quarter_turns not in (0, 1, 2, 3)
        ):
            raise StudioError("Некорректное геометрическое предсказание кожи.")
        key = (view.region, view.quarter_turns, view.scale)
        if key in geometry:
            continue
        geometry.add(key)
        skin = p[..., 2] + p[..., 3]
        exclusion = p[..., 1] + p[..., 4] + p[..., 5]
        person = 1 - p[..., 0]
        valid = np.ones(p.shape[:2], bool)
        if view.source != "full-frame":
            yy, xx = np.mgrid[: p.shape[0], : p.shape[1]]
            border = np.minimum.reduce((xx + 1, p.shape[1] - xx, yy + 1, p.shape[0] - yy))
            valid &= border >= max(1, min(p.shape[:2]) * 0.05)
        strong = (skin >= 0.65) & (skin - exclusion >= 0.55) & (person >= 0.7) & valid
        contrary = (exclusion >= 0.65) & (exclusion - skin >= 0.55) & (person >= 0.7) & valid
        area = np.s_[y0:y1, x0:x1]
        summed[area] += p * valid[..., None]
        weights[area] += valid
        skin_summed[area] += p * strong[..., None]
        positive[area] += strong
        negative[area] += contrary
    average = summed / np.maximum(weights[..., None], 1)
    consensus = (positive >= 2) & (negative == 0)
    supported = skin_summed / np.maximum(positive[..., None], 1)
    average[consensus] = supported[consensus]
    skin = average[..., 2] + average[..., 3]
    exclusion = average[..., 1] + average[..., 4] + average[..., 5]
    conflict = (positive > 0) & (negative > 0)
    exclusion[conflict] = 1
    return SkinPrediction(
        skin=skin.astype(np.float32),
        person=(1 - average[..., 0]).astype(np.float32),
        exclusion=exclusion.astype(np.float32),
        model_sha256=model_sha256,
        diagnostics={
            "model_id": model_id,
            "algorithm_version": algorithm_version,
            "view_count": len(geometry),
            "minimum_geometric_support": 2,
            "consensus_pixels": int(np.count_nonzero(consensus)),
            "conflicting_pixels": int(np.count_nonzero(conflict)),
            "unconfirmed_pixels": int(np.count_nonzero((positive > 0) & ~consensus)),
            "views": [view.provenance() for view in views],
            "score_meaning": "uncalibrated semantic probability; not measured accuracy",
        },
        model_id=model_id,
        algorithm_version=algorithm_version,
        models=({"id": model_id, "sha256": model_sha256},),
        class_maps={name: average[..., i].copy() for i, name in enumerate(SKIN_CLASS_NAMES)},
        views=tuple(views),
        consensus=consensus.astype(np.float32),
    )


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

    model_id = SKIN_MODEL_ID
    model_sha256 = SKIN_MODEL_SHA256
    algorithm_version = SKIN_ALGORITHM_VERSION

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
        self._cache = RawPredictionCache()

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

    def _view_predictions(self, rgb, hint):
        shape = rgb.shape[:2]
        scale = min(1, 512 / max(shape))
        size = max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale))
        proxy = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        hint = cv2.resize(hint, size, interpolation=cv2.INTER_AREA)
        boxes = _regions(hint)
        views = []
        regions = [(0, 0, size[0], size[1]), *boxes]
        for index, (x0, y0, x1, y1) in enumerate(regions):
            if index == 0:
                detail = proxy
                detail_scale = scale
            else:
                sx0, sx1 = round(x0 * shape[1] / size[0]), round(x1 * shape[1] / size[0])
                sy0, sy1 = round(y0 * shape[0] / size[1]), round(y1 * shape[0] / size[1])
                detail = rgb[sy0:sy1, sx0:sx1]
                detail_scale = min(1, 512 / max(detail.shape[:2]))
                detail = cv2.resize(
                    detail,
                    (
                        max(1, round(detail.shape[1] * detail_scale)),
                        max(1, round(detail.shape[0] * detail_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            for rotation in (0, 1, 2, 3):
                p = np.rot90(self._predict(np.rot90(detail, rotation)), -rotation)
                p = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                views.append(
                    SkinView(
                        p,
                        (x0, y0, x1, y1),
                        rotation,
                        detail_scale,
                        "full-frame" if index == 0 else "person-crop",
                    )
                )
        return views, proxy.shape[:2], len(boxes)

    def infer(
        self, rgb: np.ndarray, person_probability: np.ndarray | None = None
    ) -> SkinPrediction:
        validate_rgb(rgb)
        shape = rgb.shape[:2]
        if person_probability is not None and (
            person_probability.shape != shape or not np.isfinite(person_probability).all()
        ):
            raise StudioError("Некорректная подсказка области человека для модели кожи.")
        hint = (
            np.clip(person_probability.astype(np.float32), 0, 1)
            if person_probability is not None
            else np.zeros(shape, np.float32)
        )
        digest = hashlib.sha256()
        digest.update(
            str((rgb.shape, self.model_id, self.model_sha256, self.algorithm_version)).encode()
        )
        digest.update(np.ascontiguousarray(rgb, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(hint).tobytes())
        key = digest.digest()
        with self.lock:
            if not hasattr(self, "_cache"):
                self._cache = RawPredictionCache()
            raw = self._cache.get(key)
            hit = raw is not None
            if raw is None:
                views, proxy_shape, crop_count = self._view_predictions(rgb, hint)
                raw = fuse_skin_views(
                    views,
                    proxy_shape,
                    model_id=self.model_id,
                    model_sha256=self.model_sha256,
                    algorithm_version=self.algorithm_version,
                )
                raw.diagnostics.update(
                    provider="CPUExecutionProvider",
                    telemetry=False,
                    analysis_edge=max(proxy_shape),
                    input_geometry="aspect-preserving-black-letterbox",
                    inference_calls=len(views),
                    crop_count=crop_count,
                )
                self._cache.put(key, raw)
            raw.diagnostics.update(
                cache_hit=hit,
                cache_bytes=self._cache.bytes,
                cache_limit_bytes=self._cache.max_bytes,
            )
        size = shape[1], shape[0]
        arrays = [
            np.clip(cv2.resize(a, size, interpolation=cv2.INTER_LINEAR), 0, 1)
            for a in (raw.skin, raw.person, raw.exclusion)
        ]
        return SkinPrediction(
            *arrays,
            model_sha256=raw.model_sha256,
            diagnostics=raw.diagnostics,
            model_id=raw.model_id,
            algorithm_version=raw.algorithm_version,
            models=raw.models,
            class_maps=raw.class_maps,
            views=raw.views,
            consensus=cv2.resize(raw.consensus, size, interpolation=cv2.INTER_NEAREST),
        )


_skin_segmenter: SelfieSkinSegmenter | None = None
_skin_lock = Lock()


def infer_skin(rgb: np.ndarray, person_probability: np.ndarray | None = None) -> SkinPrediction:
    global _skin_segmenter
    with _skin_lock:
        if _skin_segmenter is None:
            _skin_segmenter = SelfieSkinSegmenter()
    return _skin_segmenter.infer(rgb, person_probability)
