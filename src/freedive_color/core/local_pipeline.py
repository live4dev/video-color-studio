"""Localized grading blends color-only edits into the original scene geometry."""

from dataclasses import asdict, dataclass, fields
from math import isfinite

import cv2
import numpy as np

from .analyze import linear_to_srgb, srgb_to_linear, validate_rgb
from .color_pipeline import LUMA, _gamut_limit
from .errors import StudioError
from .masks import MaskEdits, MaskSet, build_masks
from .reference import PaletteReference, match_reference

LOCAL_RANGES = {
    "skin_warmth": (-1, 1),
    "skin_strength": (0, 1),
    "water_hue": (-1, 1),
    "water_saturation": (-1, 1),
    "water_strength": (0, 1),
    "above_exposure": (-3, 3),
    "feather": (0.0001, 0.05),
    "reference_strength": (0, 1),
}


@dataclass(frozen=True)
class LocalSettings:
    enabled: bool = True
    skin_warmth: float = 0.12
    skin_strength: float = 0.35
    water_hue: float = 0
    water_saturation: float = 0
    water_strength: float = 1
    above_exposure: float = 0
    feather: float = 0.006
    reference_strength: float = 0.45

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise StudioError("Переключатель локальной коррекции должен быть логическим значением.")
        for key, (low, high) in LOCAL_RANGES.items():
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not low <= value <= high
            ):
                raise StudioError(f"Параметр {key} должен быть числом от {low} до {high}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LocalSettings":
        try:
            if not isinstance(data, dict) or set(data) - {field.name for field in fields(cls)}:
                raise ValueError("Unknown fields")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать параметры локальной коррекции.") from exc


@dataclass
class LocalResult:
    rgb: np.ndarray
    masks: MaskSet
    warnings: tuple[str, ...] = ()


def _luminance_preserving(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    luminance = source @ LUMA
    candidate += (luminance - candidate @ LUMA)[..., None]
    return _gamut_limit(candidate, luminance)


def apply_local_grade(
    original_rgb: np.ndarray,
    globally_graded_rgb: np.ndarray,
    settings: LocalSettings | None = None,
    edits: MaskEdits | None = None,
    masks: MaskSet | None = None,
    reference: PaletteReference | None = None,
) -> LocalResult:
    validate_rgb(original_rgb)
    validate_rgb(globally_graded_rgb)
    if original_rgb.shape != globally_graded_rgb.shape:
        raise StudioError("Размеры оригинала и результата локальной обработки не совпадают.")
    settings = settings or LocalSettings()
    edits = edits or MaskEdits()
    masks = masks or build_masks(original_rgb, edits, feather=settings.feather)
    if any(
        getattr(masks, key).shape != original_rgb.shape[:2]
        or not np.isfinite(getattr(masks, key)).all()
        for key in ("person", "skin", "water", "underwater")
    ):
        raise StudioError("Маски не соответствуют снимку.")
    if not settings.enabled:
        return LocalResult(globally_graded_rgb.copy(), masks, masks.warnings)
    # Only the underwater side receives the underwater global grade. Above water gets its
    # own explicit exposure control, with a soft transition along the original wave/polyline.
    if edits.waterline:
        above = original_rgb
        if settings.above_exposure:
            above = np.clip(
                linear_to_srgb(
                    srgb_to_linear(original_rgb) * np.float32(2**settings.above_exposure)
                ),
                0,
                1,
            )
        out = (
            above * (1 - masks.underwater[..., None])
            + globally_graded_rgb * masks.underwater[..., None]
        )
    else:
        out = globally_graded_rgb.copy()
    if settings.skin_strength and settings.skin_warmth and np.any(masks.skin):
        # Existing signal is multiplied, never reconstructed or smoothed. Cool reflections
        # and luminance texture remain; mask uncertainty already attenuates automatic alpha.
        gains = np.array(
            [
                1 + settings.skin_warmth * 0.18,
                1 - settings.skin_warmth * 0.025,
                1 - settings.skin_warmth * 0.055,
            ],
            np.float32,
        )
        skin = np.clip(linear_to_srgb(srgb_to_linear(out) * gains), 0, 1)
        skin = _luminance_preserving(out, skin)
        amount = (masks.skin * settings.skin_strength)[..., None]
        out = out * (1 - amount) + skin * amount
    if (
        settings.water_strength
        and (settings.water_hue or settings.water_saturation)
        and np.any(masks.water)
    ):
        lab = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2Lab)
        angle = np.float32(settings.water_hue * np.pi / 12)  # capped at ±15 degrees
        a, b = lab[..., 1].copy(), lab[..., 2].copy()
        chroma = np.float32(1 + settings.water_saturation * 0.5)
        lab[..., 1] = (a * np.cos(angle) - b * np.sin(angle)) * chroma
        lab[..., 2] = (a * np.sin(angle) + b * np.cos(angle)) * chroma
        water = _luminance_preserving(out, cv2.cvtColor(lab, cv2.COLOR_Lab2RGB))
        amount = (masks.water * settings.water_strength)[..., None]
        out = out * (1 - amount) + water * amount
    if reference is not None:
        out = match_reference(out.astype(np.float32), masks, reference, settings.reference_strength)
    warnings = masks.warnings + (reference.warnings if reference is not None else ())
    return LocalResult(np.clip(out, 0, 1).astype(np.float32), masks, tuple(dict.fromkeys(warnings)))
