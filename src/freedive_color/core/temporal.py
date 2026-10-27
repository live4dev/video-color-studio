"""Shot-level color plans and bounded-memory temporal tracking of alpha masks.

Only small analysis proxies and mask alphas are warped. Source video pixels are never
geometrically transformed here. Times are source PTS in seconds, including non-zero origins.
"""

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp, isfinite

import cv2
import numpy as np

from .analyze import SceneAnalysis, analyze_scene, validate_rgb
from .errors import StudioError
from .masks import MaskEdits, MaskSet, _apply_strokes, _feather, _underwater_mask, build_masks

ANALYSIS_FIELDS = (
    "red_gain",
    "green_gain",
    "red_confidence",
    "cyan_cast",
    "luminance_p05",
    "luminance_p50",
    "luminance_p95",
)


def _time(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise StudioError("Временная метка должна быть конечным числом секунд.")
    return float(value)


def _analysis(values: np.ndarray, warnings: tuple[str, ...] = ()) -> SceneAnalysis:
    return SceneAnalysis(
        **dict(zip(ANALYSIS_FIELDS, map(float, values), strict=True)), warnings=warnings
    )


@dataclass(frozen=True)
class FrameAnchor:
    time: float
    analysis: SceneAnalysis
    shot_id: int = 0
    frame_index: int = 0

    def __post_init__(self):
        _time(self.time)
        if (
            not isinstance(self.analysis, SceneAnalysis)
            or not isinstance(self.shot_id, int)
            or isinstance(self.shot_id, bool)
            or self.shot_id < 0
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise StudioError("Некорректная опорная точка видеоанализа.")

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "analysis": self.analysis.to_dict(),
            "shot_id": self.shot_id,
            "frame_index": self.frame_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FrameAnchor":
        try:
            if not isinstance(data, dict) or set(data) != {
                "time",
                "analysis",
                "shot_id",
                "frame_index",
            }:
                raise ValueError("Invalid anchor")
            return cls(
                data["time"],
                SceneAnalysis.from_dict(data["analysis"]),
                data["shot_id"],
                data["frame_index"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise StudioError("Не удалось прочитать опорную точку видеоанализа.") from exc


@dataclass(frozen=True)
class ShotPlan:
    anchors: tuple[FrameAnchor, ...]
    cuts: tuple[float, ...]
    duration: float
    sampled_frames: int = 0
    frame_count: int = 0
    start_time: float = 0
    algorithm: str = "shot-smooth-v1"
    _shot_anchors: tuple = field(init=False, repr=False, compare=False)
    _shot_times: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        duration, start = _time(self.duration), _time(self.start_time)
        if duration < 0 or self.algorithm != "shot-smooth-v1" or not self.anchors:
            raise StudioError("План видеоанализа пуст или несовместим с этой версией.")
        anchors = tuple(self.anchors)
        cuts = tuple(_time(value) for value in self.cuts)
        if (
            any(not isinstance(anchor, FrameAnchor) for anchor in anchors)
            or any(
                right.time <= left.time for left, right in zip(anchors, anchors[1:], strict=False)
            )
            or any(right <= left for left, right in zip(cuts, cuts[1:], strict=False))
            or any(not start < cut <= start + duration + 1e-4 for cut in cuts)
            or any(not start - 1e-4 <= anchor.time <= start + duration + 1e-4 for anchor in anchors)
        ):
            raise StudioError("Временные метки плана видеоанализа некорректны.")
        if any(anchor.shot_id != bisect_right(cuts, anchor.time) for anchor in anchors):
            raise StudioError("Опорная точка относится к другому фрагменту видео.")
        if set(anchor.shot_id for anchor in anchors) != set(range(len(cuts) + 1)):
            raise StudioError("В плане отсутствуют опорные точки одного из фрагментов.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.sampled_frames, self.frame_count)
        ):
            raise StudioError("Некорректное число кадров видеоанализа.")
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "cuts", cuts)
        grouped = [[] for _index in range(len(cuts) + 1)]
        for anchor in anchors:
            grouped[anchor.shot_id].append(anchor)
        object.__setattr__(self, "_shot_anchors", tuple(tuple(group) for group in grouped))
        object.__setattr__(
            self, "_shot_times", tuple(tuple(anchor.time for anchor in group) for group in grouped)
        )

    def shot_at(self, timestamp: float) -> int:
        return bisect_right(self.cuts, _time(timestamp))

    def analysis_at(self, timestamp: float) -> SceneAnalysis:
        timestamp = _time(timestamp)
        shot = self.shot_at(timestamp)
        anchors = self._shot_anchors[shot]
        times = self._shot_times[shot]
        right = bisect_right(times, timestamp)
        if right == 0:
            return anchors[0].analysis
        if right >= len(anchors):
            return anchors[-1].analysis
        before, after = anchors[right - 1], anchors[right]
        fraction = (timestamp - before.time) / (after.time - before.time)
        # Smoothstep has zero derivative at anchor joins, so coefficient velocity cannot jump.
        fraction = fraction * fraction * (3 - 2 * fraction)
        values = np.array([getattr(before.analysis, key) for key in ANALYSIS_FIELDS]) * (
            1 - fraction
        )
        values += np.array([getattr(after.analysis, key) for key in ANALYSIS_FIELDS]) * fraction
        warnings = tuple(dict.fromkeys(before.analysis.warnings + after.analysis.warnings))
        return _analysis(values, warnings)

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "duration": self.duration,
            "start_time": self.start_time,
            "sampled_frames": self.sampled_frames,
            "frame_count": self.frame_count,
            "cuts": list(self.cuts),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShotPlan":
        try:
            if not isinstance(data, dict) or set(data) - {
                "anchors",
                "cuts",
                "duration",
                "sampled_frames",
                "frame_count",
                "start_time",
                "algorithm",
            }:
                raise ValueError("Invalid plan")
            values = dict(data)
            values["anchors"] = tuple(FrameAnchor.from_dict(item) for item in values["anchors"])
            values["cuts"] = tuple(values.get("cuts", ()))
            return cls(**values)
        except (ValueError, KeyError, TypeError) as exc:
            raise StudioError("Не удалось прочитать план видеоанализа.") from exc


class ShotDetector:
    """Cheap proxy histogram/correspondence detector, not an independent frame grade."""

    def __init__(self, threshold: float = 0.42):
        if not isfinite(threshold) or not 0.1 <= threshold <= 1:
            raise StudioError("Порог смены сцены должен быть от 0.1 до 1.")
        self.threshold = threshold
        self.previous = None
        self.last_score = 0.0

    @staticmethod
    def _features(rgb: np.ndarray):
        small = cv2.resize(rgb, (64, 48), interpolation=cv2.INTER_AREA)
        total = np.maximum(small.sum(axis=2), 0.03)
        chroma = (small[..., :2] / total[..., None]).astype(np.float32)
        histogram = cv2.calcHist([chroma], [0, 1], None, [12, 12], [0, 1, 0, 1])
        histogram = cv2.GaussianBlur(histogram, (3, 3), 0.65)
        histogram /= max(float(histogram.sum()), 1)
        luminance = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lhist, _edges = np.histogram(luminance, bins=24, range=(0, 1))
        lhist = cv2.GaussianBlur(lhist.astype(np.float32)[None, :], (5, 1), 0.9)[0]
        lhist /= max(float(lhist.sum()), 1)
        median = np.median(chroma.reshape(-1, 2), axis=0)
        return histogram, lhist, luminance, median

    def update(self, rgb: np.ndarray, timestamp: float | None = None) -> bool:
        validate_rgb(rgb)
        if timestamp is not None:
            _time(timestamp)
        current = self._features(rgb)
        if self.previous is None:
            self.previous = current
            return False
        old = self.previous
        color_distance = float(np.abs(current[0] - old[0]).sum() * 0.5)
        luminance_distance = float(np.abs(current[1] - old[1]).sum() * 0.5)
        spatial_change = float(np.mean(np.abs(current[2] - old[2])))
        chroma_change = float(np.linalg.norm(current[3] - old[3]))
        self.last_score = min(
            1.0, color_distance * 0.65 + luminance_distance * 0.2 + spatial_change * 0.5
        )
        cut = (
            self.last_score > self.threshold
            and (chroma_change > 0.045 or (color_distance > 0.55 and spatial_change > 0.07))
        ) or (spatial_change > 0.4 and luminance_distance > 0.3)
        self.previous = current
        return bool(cut)


class TemporalAnalyzer:
    """Streaming analysis: a handful of scalar samples per window, never RGB history."""

    def __init__(
        self,
        sample_interval: float = 0.5,
        anchor_interval: float = 2.0,
        detector: ShotDetector | None = None,
    ):
        if (
            not isfinite(sample_interval)
            or not 0.1 <= sample_interval <= 10
            or not isfinite(anchor_interval)
            or not sample_interval <= anchor_interval <= 30
        ):
            raise StudioError("Некорректный интервал анализа видео.")
        self.sample_interval, self.anchor_interval = sample_interval, anchor_interval
        self.detector = detector or ShotDetector()
        self._pending: list[FrameAnchor] = []
        self._anchors: list[FrameAnchor] = []
        self._cuts: list[float] = []
        self._shot = 0
        self._first_time = None
        self._last_time = None
        self._last_sample = None
        self._window_start = None
        self._sampled_frames = 0
        self._frame_count = 0
        self._plan = None

    def _flush(self):
        if not self._pending:
            return
        values = np.array(
            [[getattr(anchor.analysis, key) for key in ANALYSIS_FIELDS] for anchor in self._pending]
        )
        warnings = tuple(
            dict.fromkeys(
                message for anchor in self._pending for message in anchor.analysis.warnings
            )
        )
        middle = self._pending[len(self._pending) // 2]
        self._anchors.append(
            FrameAnchor(
                float(np.mean([anchor.time for anchor in self._pending])),
                _analysis(np.median(values, axis=0), warnings),
                self._shot,
                middle.frame_index,
            )
        )
        self._pending.clear()
        self._window_start = None

    def add(
        self,
        rgb: np.ndarray,
        timestamp_seconds: float,
        frame_index: int | None = None,
        force_cut: bool = False,
    ) -> bool:
        if self._plan is not None:
            raise StudioError("Анализ видео уже завершён.")
        timestamp = _time(timestamp_seconds)
        if self._last_time is not None and timestamp < self._last_time:
            raise StudioError("Временные метки видео идут в обратном порядке.")
        index = self._frame_count if frame_index is None else frame_index
        if not isinstance(index, int) or index < 0:
            raise StudioError("Некорректный номер кадра.")
        cut = self.detector.update(rgb, timestamp) or force_cut
        if self._first_time is None:
            self._first_time = timestamp
            cut = False
        if cut and timestamp > self._first_time and (not self._cuts or timestamp > self._cuts[-1]):
            self._flush()
            self._cuts.append(timestamp)
            self._shot += 1
            self._last_sample = None
        else:
            cut = False
        self._last_time = timestamp
        self._frame_count += 1
        if (
            self._last_sample is None
            or timestamp - self._last_sample >= self.sample_interval - 1e-8
        ):
            if (
                self._window_start is not None
                and timestamp - self._window_start >= self.anchor_interval
            ):
                self._flush()
            self._pending.append(FrameAnchor(timestamp, analyze_scene(rgb), self._shot, index))
            self._window_start = self._window_start if self._window_start is not None else timestamp
            self._last_sample = timestamp
            self._sampled_frames += 1
        return cut

    def finish(self, duration: float | None = None) -> ShotPlan:
        if self._plan is not None:
            return self._plan
        if self._first_time is None:
            raise StudioError("Видеопоток не содержит кадров для анализа.")
        self._flush()
        length = self._last_time - self._first_time if duration is None else _time(duration)
        length = max(length, self._last_time - self._first_time)
        stabilized = []
        for shot in range(self._shot + 1):
            anchors = [anchor for anchor in self._anchors if anchor.shot_id == shot]
            values = np.array(
                [[getattr(anchor.analysis, key) for key in ANALYSIS_FIELDS] for anchor in anchors]
            )
            # Symmetric filtering uses the analysis pass's lookahead. No causal lag is copied
            # into scene exposure, and high-frequency per-frame noise cannot drive the grade.
            for direction in (range(1, len(anchors)), range(len(anchors) - 2, -1, -1)):
                for index in direction:
                    neighbor = index - 1 if direction.step > 0 else index + 1
                    dt = abs(anchors[index].time - anchors[neighbor].time)
                    blend = 1 - exp(-dt / 3)
                    values[index, :4] = (
                        values[neighbor, :4] * (1 - blend) + values[index, :4] * blend
                    )
            for index in range(1, len(anchors)):
                dt = anchors[index].time - anchors[index - 1].time
                rates = np.array([0.04, 0.01, 0.25, 0.2]) * dt
                values[index, :4] = np.clip(
                    values[index, :4], values[index - 1, :4] - rates, values[index - 1, :4] + rates
                )
            for index, anchor in enumerate(anchors):
                stabilized.append(
                    FrameAnchor(
                        anchor.time,
                        _analysis(values[index], anchor.analysis.warnings),
                        shot,
                        anchor.frame_index,
                    )
                )
        self._plan = ShotPlan(
            tuple(stabilized),
            tuple(self._cuts),
            length,
            self._sampled_frames,
            self._frame_count,
            self._first_time,
        )
        return self._plan


def build_shot_plan(samples, duration: float | None = None, **kwargs) -> ShotPlan:
    analyzer = TemporalAnalyzer(**kwargs)
    for index, (timestamp, rgb) in enumerate(samples):
        analyzer.add(rgb, timestamp, index)
    return analyzer.finish(duration)


def _resize_masks(masks: MaskSet, size: tuple[int, int]) -> MaskSet:
    arrays = []
    for key in ("person", "skin", "water", "underwater"):
        value = getattr(masks, key)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise StudioError("Некорректная временная маска видео.")
        arrays.append(
            np.clip(
                cv2.resize(value.astype(np.float32), size, interpolation=cv2.INTER_LINEAR), 0, 1
            )
        )
    return MaskSet(
        *arrays, masks.skin_confidence, masks.person_confidence, masks.warnings, masks.method
    )


def _blend_masks(before: MaskSet, after: MaskSet, amount: float) -> MaskSet:
    arrays = [
        np.clip(getattr(before, key) * (1 - amount) + getattr(after, key) * amount, 0, 1)
        for key in ("person", "skin", "water", "underwater")
    ]
    return MaskSet(
        *arrays,
        after.skin_confidence,
        after.person_confidence,
        after.warnings,
        "farneback-alpha-v1",
    )


class TemporalMaskTracker:
    """Periodic semantic refresh with dense backward-flow propagation of alpha only.

    Normalized user brush/waterline edits are intentionally reapplied in image coordinates,
    not presented as tracked object strokes. At most two proxy MaskSets and one gray frame persist.
    """

    def __init__(
        self,
        edits: MaskEdits | None = None,
        feather: float = 0.006,
        refresh_seconds: float = 1.0,
        proxy_edge: int = 384,
        mask_builder: Callable | None = None,
        use_model: bool = True,
    ):
        if (
            not isfinite(refresh_seconds)
            or not 0.25 <= refresh_seconds <= 10
            or not isinstance(proxy_edge, int)
            or not 64 <= proxy_edge <= 1024
            or not isfinite(feather)
            or not 0.0001 <= feather <= 0.05
        ):
            raise StudioError("Некорректные параметры отслеживания масок видео.")
        self.edits = edits or MaskEdits()
        self.feather, self.refresh_seconds, self.proxy_edge = feather, refresh_seconds, proxy_edge
        self.mask_builder = mask_builder or build_masks
        self.use_model = use_model
        self._gray = None
        self._state = None
        self._target = None
        self._last_time = None
        self._last_refresh = None
        self._shot_id = None
        self._source_shape = None
        self.detector = ShotDetector()
        self.model_calls = 0
        self.frames = 0
        self.resets = 0
        self.flow_failures = 0

    def _proxy(self, rgb):
        scale = min(1, self.proxy_edge / max(rgb.shape[:2]))
        size = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    def _refresh(self, proxy, timestamp, fresh_masks=None):
        if fresh_masks is None:
            # A manual box is a static requested analysis region. Brushes and waterline are
            # applied later at the source resolution rather than accumulated in tracked alpha.
            region = MaskEdits(person_box=self.edits.person_box)
            masks = self.mask_builder(proxy, region, feather=self.feather, use_model=self.use_model)
            self.model_calls += 1
        else:
            masks = _resize_masks(fresh_masks, (proxy.shape[1], proxy.shape[0]))
        self._last_refresh = timestamp
        return masks

    @staticmethod
    def _warp(masks, map_x, map_y):
        arrays = [
            cv2.remap(
                getattr(masks, key),
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for key in ("person", "skin", "water", "underwater")
        ]
        return MaskSet(
            *arrays,
            masks.skin_confidence,
            masks.person_confidence,
            masks.warnings,
            "farneback-alpha-v1",
        )

    def _manual_and_safety(self, rgb, masks: MaskSet, reliable: bool) -> MaskSet:
        person, skin, water = masks.person.copy(), masks.skin.copy(), masks.water.copy()
        guidance = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        warnings = list(masks.warnings)
        maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
        saturated = (maximum - minimum) / np.maximum(maximum, 1e-5) > 0.75
        definite_non_skin = (maximum < 0.075) | saturated | (rgb[..., 2] > rgb[..., 0] * 1.85)
        skin[definite_non_skin] = 0
        skin_confidence, person_confidence = masks.skin_confidence, masks.person_confidence
        if skin_confidence < 0.8:
            skin *= 0
        if not reliable:
            skin *= 0
            skin_confidence = 0
            warnings.append(
                "Отслеживание кожи неуверенно: автоматическая коррекция временно отключена."
            )
        if self.edits.person_box is not None:
            x0, y0, x1, y1 = self.edits.person_box
            person = np.zeros_like(person)
            person[
                round(y0 * rgb.shape[0]) : round(y1 * rgb.shape[0]),
                round(x0 * rgb.shape[1]) : round(x1 * rgb.shape[1]),
            ] = 1
            person = _feather(person, guidance, self.feather)
            skin *= person
            person_confidence = 0.9
        person = _apply_strokes(person, self.edits, "person", guidance, self.feather)
        skin = _apply_strokes(skin, self.edits, "skin", guidance, self.feather)
        if any(
            stroke.target == "skin" and stroke.operation == "add" for stroke in self.edits.strokes
        ):
            skin_confidence = 1
            person = np.maximum(person, skin)
            person_confidence = max(person_confidence, 0.9)
        underwater = _underwater_mask(rgb.shape[:2], self.edits, self.feather)
        water *= (1 - person) * (1 - skin)
        water = _apply_strokes(water, self.edits, "water", guidance, self.feather)
        water *= underwater * (1 - skin)
        if self.edits.strokes or self.edits.waterline or self.edits.person_box:
            warnings.append(
                "Ручные видеомаски фиксированы в координатах кадра; проверьте движение объектов."
            )
        return MaskSet(
            person,
            skin,
            water,
            underwater,
            skin_confidence,
            person_confidence,
            tuple(dict.fromkeys(warnings)),
            "farneback-alpha-v1",
        )

    def update(
        self,
        rgb: np.ndarray,
        timestamp: float,
        *,
        shot_id: int | None = None,
        force_reset: bool = False,
        fresh_masks: MaskSet | None = None,
    ) -> MaskSet:
        validate_rgb(rgb)
        timestamp = _time(timestamp)
        proxy = self._proxy(rgb)
        gray = np.rint(
            np.clip(proxy @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0, 1) * 255
        ).astype(np.uint8)
        cut = self.detector.update(proxy, timestamp)
        reset = (
            force_reset
            or self._state is None
            or self._source_shape != rgb.shape[:2]
            or (self._last_time is not None and timestamp < self._last_time)
            or (shot_id is not None and self._shot_id is not None and shot_id != self._shot_id)
            or cut
        )
        reliable = True
        if reset:
            self._target = self._refresh(proxy, timestamp, fresh_masks)
            self._state = _resize_masks(self._target, (proxy.shape[1], proxy.shape[0]))
            self.resets += 1
        else:
            dt = max(1e-4, timestamp - self._last_time)
            reliable = True
            try:
                # Backward flow maps a CURRENT output pixel to the PREVIOUS mask's source.
                flow = cv2.calcOpticalFlowFarneback(
                    gray, self._gray, None, 0.5, 3, 21, 3, 5, 1.1, 0
                )
                yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]].astype(np.float32)
                map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
                self._state = self._warp(self._state, map_x, map_y)
                self._target = self._warp(self._target, map_x, map_y)
                previous_gray = cv2.remap(
                    self._gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                residual = np.abs(gray.astype(np.float32) - previous_gray) / 255
                region = self._state.person > 0.2
                error = float(np.percentile(residual[region] if np.any(region) else residual, 80))
                reliable = (
                    error < 0.13
                    and float(np.percentile(np.linalg.norm(flow, axis=2), 95))
                    < min(gray.shape) * 0.25
                )
            except cv2.error:
                reliable = False
            if not reliable:
                self.flow_failures += 1
            periodic = timestamp - self._last_refresh >= self.refresh_seconds - 1e-6
            recovery = not reliable and timestamp - self._last_refresh >= 0.25
            if periodic or recovery or fresh_masks is not None:
                self._target = self._refresh(proxy, timestamp, fresh_masks)
            self._state = _blend_masks(self._state, self._target, 1 - exp(-dt / 0.18))
        self._gray = gray.copy()
        self._last_time, self._shot_id, self._source_shape = timestamp, shot_id, rgb.shape[:2]
        self.frames += 1
        full = _resize_masks(self._state, (rgb.shape[1], rgb.shape[0]))
        return self._manual_and_safety(rgb, full, reliable)

    @property
    def retained_bytes(self) -> int:
        total = self._gray.nbytes if self._gray is not None else 0
        for masks in (self._state, self._target):
            if masks is not None:
                total += sum(
                    getattr(masks, key).nbytes for key in ("person", "skin", "water", "underwater")
                )
        return total

    def to_metadata(self) -> dict:
        return {
            "method": "farneback-alpha-v1",
            "frames": self.frames,
            "semantic_refreshes": self.model_calls,
            "resets": self.resets,
            "flow_failures": self.flow_failures,
            "proxy_edge": self.proxy_edge,
            "retained_bytes": self.retained_bytes,
            "manual_edits": "static-normalized-frame-coordinates",
        }
