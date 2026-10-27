"""Conservative scene statistics, deliberately not a gray-world white balance."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .errors import StudioError


def validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise StudioError("Ожидается непустое RGB-изображение.")
    if rgb.dtype != np.float32 or not np.isfinite(rgb).all():
        raise StudioError("Некорректные пиксели: требуется конечный float32 RGB.")
    if rgb.min() < 0 or rgb.max() > 1:
        raise StudioError("HDR/линейные float-изображения пока не поддерживаются. Нужен SDR sRGB.")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055).astype(
        np.float32
    )


@dataclass(frozen=True)
class SceneAnalysis:
    red_gain: float
    green_gain: float
    red_confidence: float
    cyan_cast: float
    luminance_p05: float
    luminance_p50: float
    luminance_p95: float
    warnings: tuple[str, ...] = ()
    algorithm: str = "conservative-sdr-v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneAnalysis":
        try:
            values = dict(data)
            values["warnings"] = tuple(values.get("warnings", ()))
            result = cls(**values)
            numbers = [
                result.red_gain,
                result.green_gain,
                result.red_confidence,
                result.cyan_cast,
                result.luminance_p05,
                result.luminance_p50,
                result.luminance_p95,
            ]
            if (
                result.algorithm != "conservative-sdr-v1"
                or not all(np.isfinite(number) for number in numbers)
                or not 1 <= result.red_gain <= 1.75
                or not 0.9 <= result.green_gain <= 1
                or not all(0 <= number <= 1 for number in numbers[2:])
                or not all(isinstance(warning, str) for warning in result.warnings)
            ):
                raise ValueError("Invalid analysis")
            return result
        except (TypeError, ValueError, KeyError) as exc:
            raise StudioError("Сохранённый анализ несовместим с этой версией движка.") from exc


def analyze_scene(rgb: np.ndarray) -> SceneAnalysis:
    validate_rgb(rgb)
    height, width = rgb.shape[:2]
    scale = min(1, 640 / max(height, width))
    sample = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = sample @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    useful = sample[(y > 0.04) & (y < 0.95)]
    if len(useful) < 32:
        useful = sample.reshape(-1, 3)
    median = np.median(useful, axis=0)
    p90 = np.percentile(useful, 90, axis=0)
    red, green, blue = (float(value) for value in median)
    cast = float(np.clip((min(green, blue) - red) / (min(green, blue) + 1e-5), 0, 1))
    # Signal reliability goes to zero when red is absent. No green-to-red substitution.
    confidence = float(
        np.clip(p90[0] / max(p90[1], 0.05) / 0.5, 0, 1) * np.clip((p90[0] - 0.015) / 0.12, 0, 1)
    )
    green_cast = float(np.clip((green - blue) / (green + blue + 1e-5), 0, 0.5))
    warnings = []
    if confidence < 0.35:
        warnings.append(
            "Красного сигнала мало: точный цвет восстановить нельзя; усиление ограничено."
        )
    if float(np.percentile(y, 95)) < 0.14:
        warnings.append("Кадр очень тёмный. Сильное осветление может проявить шум.")
    q05, q50, q95 = (float(value) for value in np.percentile(y, [5, 50, 95]))
    return SceneAnalysis(
        red_gain=1 + 0.75 * cast * confidence,
        green_gain=1 - 0.13 * green_cast,
        red_confidence=confidence,
        cyan_cast=cast,
        luminance_p05=q05,
        luminance_p50=q50,
        luminance_p95=q95,
        warnings=tuple(warnings),
    )
