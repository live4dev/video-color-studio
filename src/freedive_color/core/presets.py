"""Serializable, validated controls. No hidden per-preset processing branches."""

from dataclasses import asdict, dataclass, fields
from math import isfinite

from .errors import StudioError

RANGES: dict[str, tuple[float, float]] = {
    "auto_strength": (0, 1),
    "exposure": (-3, 3),
    "temperature": (-1, 1),
    "tint": (-1, 1),
    "red_recovery": (0, 1),
    "contrast": (-1, 1),
    "saturation": (-1, 1),
    "highlights": (-1, 1),
    "shadows": (-1, 1),
    "dehaze": (0, 0.35),
    "denoise": (0, 1),
    "detail": (0, 1),
}


@dataclass(frozen=True)
class GradeSettings:
    preset: str = "natural"
    auto_strength: float = 0.7
    exposure: float = 0
    temperature: float = 0.03
    tint: float = 0.02
    red_recovery: float = 0.45
    contrast: float = 0.1
    saturation: float = 0.03
    highlights: float = -0.08
    shadows: float = 0.04
    dehaze: float = 0.04
    denoise: float = 0
    detail: float = 0.08

    def __post_init__(self) -> None:
        if self.preset not in ("natural", "cinematic", "vivid", "custom"):
            raise StudioError("Неизвестный стиль. Выберите Natural, Cinematic или Vivid.")
        for key, (low, high) in RANGES.items():
            value = getattr(self, key)
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not isfinite(value)
                or not low <= value <= high
            ):
                raise StudioError(f"Параметр {key} должен быть числом от {low} до {high}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GradeSettings":
        if not isinstance(data, dict) or set(data) - {field.name for field in fields(cls)}:
            raise StudioError("Некорректный файл настроек: неизвестные поля.")
        try:
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise StudioError("Не удалось прочитать параметры цветокоррекции.") from exc


PRESETS = {
    "natural": GradeSettings(),
    "cinematic": GradeSettings(
        preset="cinematic",
        contrast=0.24,
        saturation=-0.04,
        highlights=-0.28,
        shadows=0.03,
        dehaze=0.06,
        temperature=0.04,
        detail=0.06,
    ),
    "vivid": GradeSettings(
        preset="vivid",
        auto_strength=0.75,
        red_recovery=0.5,
        contrast=0.16,
        saturation=0.24,
        highlights=-0.12,
        shadows=0.06,
        dehaze=0.07,
        detail=0.12,
    ),
}


def get_preset(name: str) -> GradeSettings:
    try:
        return PRESETS[name.lower()]
    except (KeyError, AttributeError) as exc:
        raise StudioError(f"Неизвестный стиль: {name}.") from exc
