"""Bounded masked palette matching: only chroma statistics, never exposure or structure."""

from dataclasses import asdict, dataclass
from math import isfinite

import cv2
import numpy as np

from .analyze import validate_rgb
from .color_pipeline import LUMA, _gamut_limit
from .errors import StudioError
from .masks import MaskSet, build_masks


def _ab(value) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(number)
            or not -128 <= number <= 128
            for number in value
        )
    ):
        raise StudioError("Некорректная палитра эталона.")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class PaletteReference:
    water_ab: tuple[float, float] | None = None
    skin_ab: tuple[float, float] | None = None
    source_name: str = "Эталон"
    source_sha256: str = ""
    warnings: tuple[str, ...] = ()
    algorithm: str = "masked-chroma-v1"

    def __post_init__(self):
        object.__setattr__(self, "water_ab", _ab(self.water_ab))
        object.__setattr__(self, "skin_ab", _ab(self.skin_ab))
        if (
            self.algorithm != "masked-chroma-v1"
            or not isinstance(self.source_name, str)
            or len(self.source_name) > 4096
            or not isinstance(self.source_sha256, str)
        ):
            raise StudioError("Эталон не поддерживается этой версией приложения.")
        if self.source_sha256 and (
            len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256.lower())
        ):
            raise StudioError("Некорректная контрольная сумма эталона.")
        if not isinstance(self.warnings, (tuple, list)) or not all(
            isinstance(message, str) for message in self.warnings
        ):
            raise StudioError("Некорректные сведения об эталоне.")
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaletteReference":
        try:
            if not isinstance(data, dict):
                raise ValueError("Expected dict")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать палитру эталона.") from exc


@dataclass(frozen=True)
class ReferenceTransform:
    """Frozen, unscaled chroma deltas for one shot; no exposure or image content."""

    water_delta_ab: tuple[float, float] | None = None
    skin_delta_ab: tuple[float, float] | None = None
    source_name: str = "Эталон"
    source_sha256: str = ""
    warnings: tuple[str, ...] = ()
    algorithm: str = "frozen-chroma-v1"

    def __post_init__(self):
        for key, limit in (("water_delta_ab", 6), ("skin_delta_ab", 4)):
            value = _ab(getattr(self, key))
            if value is not None and any(abs(number) > limit for number in value):
                raise StudioError("Поправка эталона превышает допустимый предел цвета.")
            object.__setattr__(self, key, value)
        metadata = PaletteReference(
            source_name=self.source_name, source_sha256=self.source_sha256, warnings=self.warnings
        )
        object.__setattr__(self, "warnings", metadata.warnings)
        if self.algorithm != "frozen-chroma-v1":
            raise StudioError("Версия фиксированной палитры эталона не поддерживается.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceTransform":
        try:
            if not isinstance(data, dict):
                raise ValueError("Expected dict")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать фиксированную палитру эталона.") from exc


def _sample_lab(rgb: np.ndarray, masks: MaskSet) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scale = min(1, 640 / max(rgb.shape[:2]))
    width, height = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
    sample = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    small_masks = {
        key: cv2.resize(getattr(masks, key), (width, height), interpolation=cv2.INTER_AREA)
        for key in ("skin", "water")
    }
    return cv2.cvtColor(sample, cv2.COLOR_RGB2Lab), small_masks


def _median_ab(lab: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    selected = lab[(mask > 0.55) & (lab[..., 0] > 5) & (lab[..., 0] < 95)]
    if len(selected) < 12:
        return None
    # A median does not copy a bright caustic or scene exposure into the whole series.
    return tuple(float(value) for value in np.median(selected[:, 1:3], axis=0))


def create_reference(
    rgb: np.ndarray,
    masks: MaskSet | None = None,
    *,
    source_name: str = "Эталон",
    source_sha256: str = "",
) -> PaletteReference:
    validate_rgb(rgb)
    masks = masks or build_masks(rgb)
    lab, small_masks = _sample_lab(rgb, masks)
    water = _median_ab(lab, small_masks["water"])
    skin = _median_ab(lab, small_masks["skin"])
    warnings = []
    if water is None:
        warnings.append("В эталоне нет уверенной области воды: её палитра не переносится.")
    if skin is None:
        warnings.append("В эталоне нет уверенной области кожи: её палитра не переносится.")
    return PaletteReference(water, skin, source_name, source_sha256, tuple(warnings))


def freeze_reference(
    rgb: np.ndarray, masks: MaskSet, reference: PaletteReference
) -> ReferenceTransform:
    """Measure one representative shot frame once; strength is deliberately not applied here."""
    validate_rgb(rgb)
    if not isinstance(reference, PaletteReference):
        raise StudioError("Для фиксации палитры нужен исходный эталон.")
    sample_lab, sample_masks = _sample_lab(rgb, masks)
    deltas = {}
    warnings = list(reference.warnings)
    for key, target, limit in (
        ("water", reference.water_ab, 6.0),
        ("skin", reference.skin_ab, 4.0),
    ):
        current = _median_ab(sample_lab, sample_masks[key])
        delta = None
        if current is not None and target is not None:
            delta = tuple(
                float(value) for value in np.clip(np.array(target) - current, -limit, limit)
            )
        elif target is not None:
            warnings.append(
                "Во фрагменте нет уверенной области "
                + ("воды" if key == "water" else "кожи")
                + ": палитра эталона не применяется."
            )
        deltas[key + "_delta_ab"] = delta
    return ReferenceTransform(
        **deltas,
        source_name=reference.source_name,
        source_sha256=reference.source_sha256,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def match_reference(
    rgb: np.ndarray,
    masks: MaskSet,
    reference: PaletteReference | ReferenceTransform,
    strength: float = 0.45,
) -> np.ndarray:
    validate_rgb(rgb)
    if (
        not isinstance(reference, (PaletteReference, ReferenceTransform))
        or not isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise StudioError("Некорректная сила палитры эталона.")
    if strength == 0:
        return rgb.copy()
    transform = (
        freeze_reference(rgb, masks, reference)
        if isinstance(reference, PaletteReference)
        else reference
    )
    if transform.water_delta_ab is transform.skin_delta_ab is None:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    combined = np.zeros_like(rgb)
    alpha_sum = np.zeros(rgb.shape[:2], np.float32)
    luminance = rgb @ LUMA
    for key, delta in (
        ("water", transform.water_delta_ab),
        ("skin", transform.skin_delta_ab),
    ):
        if delta is None:
            continue
        delta = np.array(delta, np.float32)
        amount = getattr(masks, key) * np.float32(strength)
        shifted = lab.copy()
        shifted[..., 1:3] += delta
        candidate = cv2.cvtColor(shifted, cv2.COLOR_Lab2RGB)
        # Lab round-trips/gamut clipping can perturb L slightly. Restore source luminance.
        candidate += (luminance - candidate @ LUMA)[..., None]
        candidate = _gamut_limit(candidate, luminance)
        # A pathological reference cannot turn a different water/skin palette into an
        # aggressive recolor. Scaling a zero-luminance delta also preserves local exposure.
        difference = candidate - rgb
        peak = np.max(np.abs(difference), axis=2)
        candidate = rgb + difference * np.minimum(1, 0.1 / np.maximum(peak, 1e-6))[..., None]
        combined += candidate * amount[..., None]
        alpha_sum += amount
    alpha_sum = np.clip(alpha_sum, 0, 1)
    return np.clip(rgb * (1 - alpha_sum[..., None]) + combined, 0, 1).astype(np.float32)
