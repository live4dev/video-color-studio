"""Conservative, non-geometric person/skin/water masks and serializable manual edits.

The optional PPHumanSeg model is bundled at build time and runs on OpenCV's CPU backend.
No model, data or network client is downloaded/started by this module.
"""

import hashlib
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .analyze import validate_rgb
from .errors import StudioError

MODEL_NAME = "human_segmentation_pphumanseg_2023mar.onnx"
MODEL_SHA256 = "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24"
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / MODEL_NAME


def _point(point) -> tuple[float, float]:
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise StudioError("Точка маски должна содержать координаты x и y.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not isfinite(value)
        or not 0 <= value <= 1
        for value in point
    ):
        raise StudioError("Координаты маски должны быть конечными числами от 0 до 1.")
    return float(point[0]), float(point[1])


@dataclass(frozen=True)
class BrushStroke:
    target: str = "skin"
    operation: str = "add"
    points: tuple[tuple[float, float], ...] = ()
    radius: float = 0.02

    def __post_init__(self):
        if self.target not in ("skin", "water", "person") or self.operation not in (
            "add",
            "subtract",
        ):
            raise StudioError("Неизвестная кисть маски или действие.")
        if not isinstance(self.points, (tuple, list)) or not 1 <= len(self.points) <= 10_000:
            raise StudioError("Штрих кисти должен содержать от 1 до 10000 точек.")
        object.__setattr__(self, "points", tuple(_point(point) for point in self.points))
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, (float, int))
            or not isfinite(self.radius)
            or not 0.0005 <= self.radius <= 0.25
        ):
            raise StudioError("Радиус кисти должен быть от 0.0005 до 0.25 размера снимка.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BrushStroke":
        try:
            if not isinstance(data, dict) or set(data) - {
                "target",
                "operation",
                "points",
                "radius",
            }:
                raise ValueError("Unknown fields")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать штрих маски.") from exc


@dataclass(frozen=True)
class MaskEdits:
    strokes: tuple[BrushStroke, ...] = ()
    waterline: tuple[tuple[float, float], ...] = ()
    person_box: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.strokes, (tuple, list)) or len(self.strokes) > 2000:
            raise StudioError("Слишком много штрихов маски (максимум 2000).")
        if not all(isinstance(stroke, BrushStroke) for stroke in self.strokes):
            raise StudioError("Некорректный штрих маски.")
        object.__setattr__(self, "strokes", tuple(self.strokes))
        if not isinstance(self.waterline, (tuple, list)) or len(self.waterline) > 256:
            raise StudioError("Граница воды должна содержать не более 256 точек.")
        if len(self.waterline) == 1:
            raise StudioError("Для границы воды нужны хотя бы две точки.")
        line = tuple(_point(point) for point in self.waterline)
        if line and any(right[0] <= left[0] for left, right in zip(line, line[1:], strict=False)):
            raise StudioError("Точки границы воды должны идти слева направо.")
        object.__setattr__(self, "waterline", line)
        if self.person_box is not None:
            if not isinstance(self.person_box, (tuple, list)) or len(self.person_box) != 4:
                raise StudioError("Рамка человека должна содержать x0, y0, x1, y1.")
            x0, y0 = _point(self.person_box[:2])
            x1, y1 = _point(self.person_box[2:])
            if x0 >= x1 or y0 >= y1:
                raise StudioError("Рамка человека должна иметь ненулевую ширину и высоту.")
            object.__setattr__(self, "person_box", (x0, y0, x1, y1))

    def to_dict(self) -> dict:
        return {
            "strokes": [stroke.to_dict() for stroke in self.strokes],
            "waterline": [list(point) for point in self.waterline],
            "person_box": list(self.person_box) if self.person_box is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaskEdits":
        try:
            if not isinstance(data, dict) or set(data) - {"strokes", "waterline", "person_box"}:
                raise ValueError("Unknown fields")
            values = dict(data)
            values["strokes"] = tuple(
                BrushStroke.from_dict(stroke) for stroke in data.get("strokes", ())
            )
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать ручную маску.") from exc


@dataclass
class MaskSet:
    person: np.ndarray
    skin: np.ndarray
    water: np.ndarray
    underwater: np.ndarray
    skin_confidence: float
    person_confidence: float
    warnings: tuple[str, ...] = ()
    method: str = "manual-only"

    def to_metadata(self) -> dict:
        return {
            "method": self.method,
            "skin_confidence": self.skin_confidence,
            "person_confidence": self.person_confidence,
            "warnings": list(self.warnings),
            "coverage": {
                key: float(np.mean(getattr(self, key)))
                for key in ("person", "skin", "water", "underwater")
            },
        }

    def to_dict(self) -> dict:
        """Statistics only: full masks are regenerated from source pixels and normalized edits."""
        return self.to_metadata()


class PPHumanSegmenter:
    """Apache-2.0 OpenCV Zoo PPHumanSeg; inference is explicitly CPU-only."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.is_file():
            raise StudioError("Локальная модель человека отсутствует. Используйте рамку или кисть.")
        model_bytes = model_path.read_bytes()
        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != MODEL_SHA256:
            raise StudioError("Контрольная сумма локальной модели не совпала. Используйте кисть.")
        try:
            # Python handles Unicode filenames consistently on Windows; OpenCV receives bytes.
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer(model_bytes, dtype=np.uint8))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise StudioError(
                "CPU-декодер модели недоступен. Используйте рамку или кисть."
            ) from exc
        self.lock = Lock()

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        # PPHumanSeg expects RGB float32 in [-1, 1], NCHW at 192×192.
        validate_rgb(rgb)
        best_probability = None
        best_score = -1.0
        # Underwater people can be upside down. Only rotate analysis, never image pixels.
        with self.lock:
            for rotation in (0, 2, 1, 3):
                rotated = np.rot90(rgb, rotation)
                small = cv2.resize(rotated, (192, 192), interpolation=cv2.INTER_AREA)
                blob = cv2.dnn.blobFromImage((small - 0.5) * 2)
                self.net.setInput(blob)
                output = self.net.forward()[0]
                if output.shape[0] != 2:
                    raise StudioError("Неожиданный формат выхода модели человека.")
                if (
                    output.min() < 0
                    or output.max() > 1
                    or not np.allclose(output.sum(axis=0), 1, atol=0.02)
                ):
                    exp = np.exp(output - output.max(axis=0, keepdims=True))
                    output = exp / exp.sum(axis=0, keepdims=True)
                probability = np.rot90(output[1], -rotation)
                foreground = probability > 0.75
                coverage = float(foreground.mean())
                score = float(np.mean(probability[foreground])) if foreground.any() else 0
                score *= min(1, coverage / 0.025) * min(1, (1 - coverage) / 0.3)
                if score > best_score:
                    best_score, best_probability = score, probability
                if rotation == 0 and score > 0.94:
                    break
            # A color-guided refinement protects a dark/blue wetsuit that the general-purpose
            # model may miss under water. The graph operates on a fixed small proxy, and only
            # changes the mask. It never removes or reconstructs source objects.
            probability = self._refine(rgb, best_probability.astype(np.float32))
        return cv2.resize(probability, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _refine(rgb: np.ndarray, probability: np.ndarray) -> np.ndarray:
        scale = min(1, 384 / max(rgb.shape[:2]))
        size = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        probability = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
        certain = probability > 0.85
        if not 0.0005 < certain.mean() < 0.75:
            return probability
        radius = max(2, round(min(probability.shape) * 0.075))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        region = cv2.dilate((probability > 0.2).astype(np.uint8), kernel) > 0
        labels = np.full(probability.shape, cv2.GC_BGD, np.uint8)
        labels[region] = cv2.GC_PR_BGD
        labels[probability > 0.35] = cv2.GC_PR_FGD
        labels[certain] = cv2.GC_FGD
        try:
            # This method runs under the segmenter lock. Fixed initialization makes exported
            # masks reproducible and avoids stochastic differences between proxy/full renders.
            cv2.setRNGSeed(31)
            cv2.grabCut(
                np.rint(small * 255).astype(np.uint8),
                labels,
                None,
                np.zeros((1, 65)),
                np.zeros((1, 65)),
                3,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return probability
        foreground = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
        if foreground.mean() > 0.75:
            return probability
        # Newly grown regions do not gain a synthetic confidence of 1 from the graph cut.
        return np.maximum(probability, foreground.astype(np.float32) * 0.88)


_segmenter: PPHumanSegmenter | None = None
_segmenter_lock = Lock()


def infer_person(rgb: np.ndarray) -> np.ndarray:
    global _segmenter
    with _segmenter_lock:
        if _segmenter is None:
            _segmenter = PPHumanSegmenter()
    return _segmenter.infer(rgb)


def _feather(mask: np.ndarray, guidance: np.ndarray, fraction: float) -> np.ndarray:
    """Small guided filter: smooth alpha while respecting existing luminance edges."""
    radius = max(1, min(48, round(min(mask.shape) * fraction)))
    kernel = (radius * 2 + 1, radius * 2 + 1)
    mean_i = cv2.boxFilter(guidance, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guidance * guidance, -1, kernel, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guidance * mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
    a = (corr_ip - mean_i * mean_p) / (np.maximum(corr_i - mean_i * mean_i, 0) + 0.0005)
    b = mean_p - a * mean_i
    result = cv2.boxFilter(a, -1, kernel) * guidance + cv2.boxFilter(b, -1, kernel)
    return np.clip(result, 0, 1).astype(np.float32)


def rasterize_stroke(stroke: BrushStroke, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.float32)
    points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in stroke.points]
    radius = max(1, round(stroke.radius * min(shape)))
    cv2.polylines(mask, [np.array(points, np.int32)], False, 1.0, radius * 2, cv2.LINE_AA)
    for point in (points[0], points[-1]):
        cv2.circle(mask, point, radius, 1.0, -1, cv2.LINE_AA)
    return mask


def _apply_strokes(
    mask: np.ndarray, edits: MaskEdits, target: str, guidance: np.ndarray, feather: float
) -> np.ndarray:
    for stroke in edits.strokes:
        if stroke.target != target:
            continue
        alpha = _feather(rasterize_stroke(stroke, mask.shape), guidance, feather * 0.5)
        if stroke.operation == "add":
            mask = np.maximum(mask, alpha)
        else:
            mask *= 1 - alpha
    return np.clip(mask, 0, 1).astype(np.float32)


def _underwater_mask(shape: tuple[int, int], edits: MaskEdits, feather: float) -> np.ndarray:
    if not edits.waterline:
        return np.ones(shape, np.float32)
    height, width = shape
    xs, ys = zip(*edits.waterline, strict=True)
    boundary = np.interp(np.linspace(0, 1, width), xs, ys)
    signed = np.arange(height, dtype=np.float32)[:, None] / max(1, height - 1) - boundary[None, :]
    # Smooth alpha around the user's actual polyline; never straighten or move the wave.
    alpha = np.clip(signed / max(feather, 1 / max(1, height)) * 0.5 + 0.5, 0, 1)
    return (alpha * alpha * (3 - 2 * alpha)).astype(np.float32)


def build_masks(
    rgb: np.ndarray,
    edits: MaskEdits | None = None,
    person_probability: np.ndarray | None = None,
    *,
    feather: float = 0.006,
    use_model: bool = True,
) -> MaskSet:
    validate_rgb(rgb)
    edits = edits or MaskEdits()
    if not isinstance(edits, MaskEdits) or not isfinite(feather) or not 0.0001 <= feather <= 0.05:
        raise StudioError("Некорректные параметры маски.")
    shape = rgb.shape[:2]
    guidance = (rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    underwater = _underwater_mask(shape, edits, feather)
    warnings = []
    method = "manual-only"
    if person_probability is None and edits.person_box is None and use_model:
        try:
            person_probability = infer_person(rgb)
            method = "pphumanseg-opencv-cpu"
        except (StudioError, cv2.error) as exc:
            warnings.append(str(exc))
    elif person_probability is not None:
        method = "supplied-person-probability"
    person = np.zeros(shape, np.float32)
    person_confidence = 0.0
    if person_probability is not None:
        if (
            not isinstance(person_probability, np.ndarray)
            or person_probability.shape != shape
            or not np.isfinite(person_probability).all()
        ):
            raise StudioError("Размер или значения маски человека некорректны.")
        probability = np.clip(person_probability.astype(np.float32), 0, 1)
        certain = probability > 0.8
        coverage = float(certain.mean())
        if 0.0005 < coverage < 0.85:
            person_confidence = float(np.median(probability[certain]))
            person = _feather(np.clip((probability - 0.35) / 0.5, 0, 1), guidance, feather)
        else:
            warnings.append("Человек не выделен уверенно. Автоматическая коррекция кожи отключена.")
    if edits.person_box is not None:
        x0, y0, x1, y1 = edits.person_box
        box = np.zeros(shape, np.float32)
        box[
            round(y0 * shape[0]) : round(y1 * shape[0]), round(x0 * shape[1]) : round(x1 * shape[1])
        ] = 1
        person = _feather(box, guidance, feather)
        person_confidence = 0.9
        method = "manual-person-region"
    person = _apply_strokes(person, edits, "person", guidance, feather)
    if any(stroke.target == "person" and stroke.operation == "add" for stroke in edits.strokes):
        person_confidence = max(person_confidence, 0.9)
    if person_confidence < 0.8 and not warnings:
        warnings.append(
            "Нет уверенной маски человека. Укажите область человека или добавьте кожу кистью."
        )
    r, g, b = np.moveaxis(rgb, -1, 0)
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-5)
    # This is a skin-color candidate *inside* a semantic/manual person region, not a skin model.
    # It deliberately abstains on severe cyan casts and very saturated clothing.
    candidate = (
        (r > g * 0.87)
        & (r > b * 0.8)
        & (r - np.maximum(g, b) > np.maximum(0.012, maximum * 0.035))
        & (g > b * 0.65)
        & (g < r * 1.17)
        & (saturation < 0.66)
        & (saturation > 0.08)
        & (maximum > 0.075)
        & (minimum > 0.025)
        & (guidance < 0.96)
    ).astype(np.float32)
    skin = np.zeros(shape, np.float32)
    skin_confidence = 0.0
    accepted = candidate * (person > 0.7)
    radius = max(1, min(6, round(min(shape) * 0.002)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    accepted = cv2.morphologyEx(accepted, cv2.MORPH_OPEN, kernel)
    if person_confidence >= 0.8 and np.count_nonzero(accepted) >= max(
        8, rgb.shape[0] * rgb.shape[1] * 0.00005
    ):
        skin_confidence = min(0.9, person_confidence)
        skin = _feather(accepted, guidance, feather) * person * candidate * skin_confidence
    elif person_confidence >= 0.8:
        warnings.append(
            "Цвет кожи не определён уверенно. Для локальной коррекции добавьте маску кистью."
        )
    skin = _apply_strokes(skin, edits, "skin", guidance, feather)
    if any(stroke.target == "skin" and stroke.operation == "add" for stroke in edits.strokes):
        skin_confidence = 1.0
        # Explicitly marked skin is also known to belong to the person. It does not create
        # an inferred body silhouette outside the user's actual brush area.
        person = np.maximum(person, skin)
        person_confidence = max(person_confidence, 0.9)
        warnings = [
            message
            for message in warnings
            if not message.startswith(
                ("Нет уверенной маски человека", "Цвет кожи не определён", "Человек не выделен")
            )
        ]
        method += "+manual-skin"
    # Water hue heuristic is allowed only outside the person, and below a manual waterline.
    water_color = ((g > r * 1.08) & (b > r * 1.08) & (maximum > 0.05)).astype(np.float32)
    water = _feather(water_color, guidance, feather) * (1 - person) * underwater * (1 - skin)
    water = _apply_strokes(water, edits, "water", guidance, feather) * underwater * (1 - skin)
    return MaskSet(
        person,
        skin,
        water.astype(np.float32),
        underwater,
        skin_confidence,
        person_confidence,
        tuple(dict.fromkeys(warnings)),
        method,
    )
