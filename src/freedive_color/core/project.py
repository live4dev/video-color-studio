"""Versioned local JSON settings, source checksums and reproducibility records."""

import hashlib
import json
from pathlib import Path

from .analyze import SceneAnalysis
from .errors import StudioError
from .presets import GradeSettings

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    try:
        if Path(path).stat().st_size > 2_000_000:
            raise StudioError("Файл настроек слишком большой (больше 2 МБ).")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise StudioError("Версия файла настроек не поддерживается.")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise StudioError("Не удалось прочитать JSON настроек. Проверьте файл.") from exc


def load_settings(path: str | Path) -> GradeSettings:
    data = read_json(path)
    if "settings" not in data:
        raise StudioError("В JSON отсутствуют параметры обработки.")
    return GradeSettings.from_dict(data["settings"])


def load_recipe(path: str | Path, source: Path) -> tuple[GradeSettings, SceneAnalysis]:
    data = read_json(path)
    try:
        if data["source"]["sha256"] != sha256_file(source):
            raise StudioError("Рецепт создан для другого исходника. Контрольная сумма не совпала.")
        return GradeSettings.from_dict(data["settings"]), SceneAnalysis.from_dict(data["analysis"])
    except (KeyError, TypeError) as exc:
        raise StudioError("В рецепте отсутствует исходник, анализ или параметры.") from exc


def json_bytes(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_settings(path: str | Path, settings: GradeSettings, *, local_settings=None) -> None:
    """Explicit save dialog: still never overwrite an existing file."""
    try:
        with Path(path).open("xb") as handle:
            handle.write(
                json_bytes(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "preset",
                        "settings": settings.to_dict(),
                        **(
                            {"local_settings": local_settings.to_dict()}
                            if local_settings is not None
                            else {}
                        ),
                    }
                )
            )
    except FileExistsError as exc:
        raise StudioError("Файл настроек уже существует. Выберите новое имя.") from exc
    except OSError as exc:
        raise StudioError("Не удалось сохранить настройки. Проверьте доступ к папке.") from exc
