"""Versioned, local project sessions; explicit state, never executable project files."""

import errno
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from .analyze import SceneAnalysis
from .errors import StudioError
from .presets import GradeSettings
from .project import json_bytes

SESSION_SCHEMA_VERSION = 1
MAX_PROJECT_BYTES = 16 * 1024 * 1024
MAX_ITEMS = 10_000


def _json_dict(value: Any, label: str) -> dict:
    """Reject Python objects, non-finite numbers and excessively deep/large JSON trees."""
    if not isinstance(value, dict):
        raise StudioError(f"{label}: ожидается JSON-объект.")
    count = 0

    def check(item: Any, depth: int = 0) -> None:
        nonlocal count
        count += 1
        if depth > 16 or count > 300_000:
            raise StudioError(f"{label}: слишком большой или вложенный объект.")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                check(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child, depth + 1)
            return
        raise StudioError(f"{label}: допустимы только конечные JSON-значения.")

    check(value)
    # Isolate mutable caller dictionaries, preserving only their JSON representation.
    return json.loads(json.dumps(value, allow_nan=False))


def _local_path(value: str | Path, base: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or "\0" in str(value):
        raise StudioError("Некорректный локальный путь в проекте.")
    text = str(value)
    if text.startswith(("//", "\\\\")) or "://" in text:
        raise StudioError("Сетевые пути в проекте не поддерживаются. Выберите локальные файлы.")
    if PureWindowsPath(text).is_absolute() and os.name != "nt":
        raise StudioError(
            "Проект содержит абсолютный путь Windows. Переместите медиа рядом с проектом."
        )
    if isinstance(value, str) and not PureWindowsPath(text).is_absolute():
        text = text.replace("\\", "/")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


def _serialized_path(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Different Windows drives cannot be made relative.
        return str(path)


def _local_state(data: dict, kind: str) -> dict:
    clean = _json_dict(data, "Параметры локальной обработки" if kind == "local" else "Правки масок")
    if not clean:
        return clean
    if kind == "local":
        from .local_pipeline import LocalSettings

        return LocalSettings.from_dict(clean).to_dict()
    from .masks import MaskEdits

    return MaskEdits.from_dict(clean).to_dict()


@dataclass(frozen=True)
class ExportOptions:
    suffix: str = ".png"
    bit_depth: int = 8
    quality: int = 95
    keep_gps: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.suffix, str) or self.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        ):
            raise StudioError("Формат серии должен быть JPEG, PNG или TIFF.")
        object.__setattr__(self, "suffix", self.suffix.lower())
        if type(self.bit_depth) is not int or self.bit_depth not in (8, 16):
            raise StudioError("Разрядность серии должна быть 8 или 16 бит.")
        if self.suffix in (".jpg", ".jpeg") and self.bit_depth != 8:
            raise StudioError("JPEG поддерживает только 8 бит. Выберите PNG/TIFF для 16 бит.")
        if type(self.quality) is not int or not 1 <= self.quality <= 100:
            raise StudioError("Качество JPEG должно быть от 1 до 100.")
        if type(self.keep_gps) is not bool:
            raise StudioError("Сохранение GPS должно быть логическим значением.")

    def to_dict(self) -> dict:
        return {
            "suffix": self.suffix,
            "bit_depth": self.bit_depth,
            "quality": self.quality,
            "keep_gps": self.keep_gps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportOptions":
        if not isinstance(data, dict) or set(data) - {"suffix", "bit_depth", "quality", "keep_gps"}:
            raise StudioError("Неизвестные параметры экспорта проекта.")
        try:
            return cls(**data)
        except TypeError as exc:
            raise StudioError("Некорректные параметры экспорта проекта.") from exc


@dataclass(frozen=True)
class SessionItem:
    path: Path
    settings: GradeSettings = field(default_factory=GradeSettings)
    included: bool = True
    local_settings: dict = field(default_factory=dict)
    mask_edits: dict = field(default_factory=dict)
    analysis: dict | None = None
    media_info: dict = field(default_factory=dict)
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _local_path(self.path))
        if not isinstance(self.settings, GradeSettings):
            raise StudioError("Некорректные параметры снимка в проекте.")
        if type(self.included) is not bool:
            raise StudioError("Включение файла в серию должно быть логическим значением.")
        object.__setattr__(self, "local_settings", _local_state(self.local_settings, "local"))
        object.__setattr__(self, "mask_edits", _local_state(self.mask_edits, "masks"))
        object.__setattr__(self, "media_info", _json_dict(self.media_info, "Сведения о медиа"))
        if self.analysis is not None:
            analysis = SceneAnalysis.from_dict(_json_dict(self.analysis, "Анализ снимка"))
            object.__setattr__(self, "analysis", analysis.to_dict())
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            raise StudioError("Некорректная контрольная сумма исходника в проекте.")


@dataclass(frozen=True)
class Session:
    items: tuple[SessionItem, ...] = ()
    reference_path: Path | None = None
    reference_data: dict | None = None
    temp_directory: Path | None = None
    output_directory: Path | None = None
    export_options: ExportOptions = field(default_factory=ExportOptions)
    selected_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.items, (tuple, list)) or len(self.items) > MAX_ITEMS:
            raise StudioError("Проект содержит слишком много файлов.")
        if any(not isinstance(item, SessionItem) for item in self.items):
            raise StudioError("Некорректный список файлов проекта.")
        object.__setattr__(self, "items", tuple(self.items))
        if len({item.path for item in self.items}) != len(self.items):
            raise StudioError("Один исходник добавлен в проект дважды.")
        for name in ("reference_path", "temp_directory", "output_directory"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _local_path(value))
        if self.reference_data is not None:
            from .reference import PaletteReference

            clean = _json_dict(self.reference_data, "Эталонная палитра")
            object.__setattr__(self, "reference_data", PaletteReference.from_dict(clean).to_dict())
        if not isinstance(self.export_options, ExportOptions):
            raise StudioError("Некорректные параметры экспорта проекта.")
        if self.selected_index is not None and (
            type(self.selected_index) is not int or not 0 <= self.selected_index < len(self.items)
        ):
            raise StudioError("Выбранного файла нет в проекте.")


def session_to_dict(session: Session, project_path: str | Path) -> dict:
    base = _local_path(project_path).parent
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "kind": "freedive_project",
        "items": [
            {
                "path": _serialized_path(item.path, base),
                "settings": item.settings.to_dict(),
                "included": item.included,
                "local_settings": item.local_settings,
                "mask_edits": item.mask_edits,
                "analysis": item.analysis,
                "media_info": item.media_info,
                "source_sha256": item.source_sha256,
            }
            for item in session.items
        ],
        "reference_path": _serialized_path(session.reference_path, base),
        "reference_data": session.reference_data,
        "temp_directory": _serialized_path(session.temp_directory, base),
        "output_directory": _serialized_path(session.output_directory, base),
        "export_options": session.export_options.to_dict(),
        "selected_index": session.selected_index,
    }


def _read_project(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_PROJECT_BYTES:
            raise StudioError("Проект больше 16 МБ. Уменьшите число файлов или правок масок.")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise StudioError("Не удалось прочитать проект. Проверьте JSON и доступ к файлу.") from exc
    if not isinstance(data, dict) or data.get("kind") != "freedive_project":
        raise StudioError("Это не проект Freedive Color Studio.")
    if (
        type(data.get("schema_version")) is not int
        or data["schema_version"] != SESSION_SCHEMA_VERSION
    ):
        raise StudioError("Версия проекта не поддерживается.")
    return data


def load_session(path: str | Path) -> Session:
    path = _local_path(path)
    data = _read_project(path)
    allowed = {
        "schema_version",
        "kind",
        "items",
        "reference_path",
        "reference_data",
        "temp_directory",
        "output_directory",
        "export_options",
        "selected_index",
    }
    if set(data) - allowed:
        raise StudioError("Проект содержит неизвестные поля.")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or len(raw_items) > MAX_ITEMS:
        raise StudioError("Некорректный список файлов проекта.")
    items = []
    item_fields = {
        "path",
        "settings",
        "included",
        "local_settings",
        "mask_edits",
        "analysis",
        "media_info",
        "source_sha256",
    }
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) - item_fields or "path" not in raw:
            raise StudioError("Некорректная запись файла в проекте.")
        items.append(
            SessionItem(
                path=_local_path(raw["path"], path.parent),
                settings=GradeSettings.from_dict(raw.get("settings", {})),
                included=raw.get("included", True),
                local_settings=raw.get("local_settings", {}),
                mask_edits=raw.get("mask_edits", {}),
                analysis=raw.get("analysis"),
                media_info=raw.get("media_info", {}),
                source_sha256=raw.get("source_sha256"),
            )
        )
    paths = {
        name: _local_path(data[name], path.parent) if data.get(name) is not None else None
        for name in ("reference_path", "temp_directory", "output_directory")
    }
    return Session(
        items=tuple(items),
        **paths,
        reference_data=data.get("reference_data"),
        export_options=ExportOptions.from_dict(data.get("export_options", {})),
        selected_index=data.get("selected_index"),
    )


def save_session(path: str | Path, session: Session, *, overwrite: bool = False) -> None:
    """Save As protects existing data; explicit Save may atomically replace only a project JSON."""
    path = _local_path(path)
    if path.suffix.lower() not in (".json", ".fcsproj"):
        raise StudioError("Для проекта выберите расширение .json или .fcsproj.")
    protected = {item.path for item in session.items}
    if session.reference_path is not None:
        protected.add(session.reference_path)
    if path in protected:
        raise StudioError("Исходный файл защищён. Выберите другое имя проекта.")
    if path.exists():
        if not overwrite:
            raise StudioError(
                "Проект уже существует. Выберите новое имя или сохраните текущий проект."
            )
        _read_project(path)
    payload = json_bytes(session_to_dict(session, path))
    if len(payload) > MAX_PROJECT_BYTES:
        raise StudioError("Проект больше 16 МБ. Уменьшите число файлов или правок масок.")
    staging = None
    exclusive_created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".freedive-project-", suffix=".tmp", dir=path.parent)
        staging = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(staging, path)
        else:
            try:
                os.link(staging, path)
            except OSError as exc:
                if exc.errno not in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                    raise
                with path.open("xb") as handle:
                    exclusive_created = True
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
    except OSError as exc:
        if exclusive_created:
            path.unlink(missing_ok=True)
        raise StudioError(
            "Не удалось сохранить проект. Проверьте права, место и имя файла."
        ) from exc
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
