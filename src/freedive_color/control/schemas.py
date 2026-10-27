"""Strict, media-free API 1 schemas shared by the controller and MCP adapter.

Importing this module does not import Qt, OpenCV, the segmentation model or a
network transport. Numeric bounds mirror the core and are checked by tests.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from freedive_color import __version__
from freedive_color.core.presets import PRESETS, RANGES

API_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576
MAX_GROUP_SIZE = 100
MAX_PAGE_SIZE = 100

# Keep the adapter lightweight. A regression test compares these values/defaults
# with LocalSettings without importing its image-processing dependencies here.
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
LOCAL_DEFAULTS = {
    "enabled": True,
    "skin_warmth": 0.12,
    "skin_strength": 0.35,
    "water_hue": 0,
    "water_saturation": 0,
    "water_strength": 1,
    "above_exposure": 0,
    "feather": 0.006,
    "reference_strength": 0.45,
}


class StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _patch_model(name: str, bounds: dict, *, enabled: bool = False) -> type[BaseModel]:
    # default_factory allows omission without advertising/accepting a JSON null.
    # model_fields_set determines which values are actually applied by a patch.
    fields = {
        key: (Annotated[float, Field(ge=low, le=high)], Field(default_factory=float))
        for key, (low, high) in bounds.items()
    }
    if enabled:
        fields["enabled"] = (bool, Field(default_factory=lambda: True))
    return create_model(name, __base__=StrictParams, **fields)


GradePatch = _patch_model("GradePatch", RANGES)
LocalPatch = _patch_model("LocalPatch", LOCAL_RANGES, enabled=True)
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
RequestId = Annotated[str, Field(min_length=1, max_length=128)]
Revision = Annotated[int, Field(ge=0)]
Seconds = Annotated[float, Field(ge=0)]
LocalPath = Annotated[str, Field(min_length=1, max_length=32_768)]
MediaIds = Annotated[list[Identifier], Field(min_length=1, max_length=MAX_GROUP_SIZE)]


class EmptyParams(StrictParams):
    pass


class SessionParams(StrictParams):
    session_id: Identifier


class ListMediaParams(SessionParams):
    cursor: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    limit: Annotated[int, Field(ge=1, le=MAX_PAGE_SIZE)] = 50


class ImportMediaParams(SessionParams):
    paths: Annotated[list[LocalPath], Field(min_length=1, max_length=MAX_GROUP_SIZE)]
    request_id: RequestId | None = None

    @model_validator(mode="after")
    def distinct_paths(self):
        if len(self.paths) != len(set(self.paths)):
            raise ValueError("Укажите каждый исходный путь только один раз.")
        return self


class SelectMediaParams(SessionParams):
    media_id: Identifier
    time_seconds: Seconds = 0


class SetGradeParams(SessionParams):
    media_ids: MediaIds
    expected_revisions: dict[Identifier, Revision]
    preset: Literal["natural", "cinematic", "vivid"] | None = None
    grade: GradePatch | None = None
    local: LocalPatch | None = None

    @model_validator(mode="after")
    def complete_atomic_patch(self):
        _distinct_media(self.media_ids)
        if set(self.expected_revisions) != set(self.media_ids):
            raise ValueError("expected_revisions должен содержать ровно все выбранные media_ids.")
        if (
            self.preset is None
            and not (self.grade and self.grade.model_fields_set)
            and not (self.local and self.local.model_fields_set)
        ):
            raise ValueError("Передайте preset либо хотя бы один параметр grade/local.")
        return self


class SetReferenceParams(SessionParams):
    source: Literal["photo", "frame", "clear"]
    path: LocalPath | None = None
    media_id: Identifier | None = None
    time_seconds: Seconds | None = None
    expected_revisions: dict[Identifier, Revision] | None = None
    request_id: RequestId | None = None

    @model_validator(mode="after")
    def one_reference_source(self):
        if self.source == "photo":
            if self.path is None or self.media_id is not None or self.time_seconds is not None:
                raise ValueError("Для source=photo передайте только локальный path.")
        elif self.source == "frame":
            if self.media_id is None or self.time_seconds is None or self.path is not None:
                raise ValueError("Для source=frame передайте media_id и time_seconds без path.")
        elif self.path is not None or self.media_id is not None or self.time_seconds is not None:
            raise ValueError("Для source=clear не передавайте path, media_id или time_seconds.")
        return self


class RenderPreviewParams(SessionParams):
    media_id: Identifier
    start_seconds: Seconds = 0
    duration_seconds: Annotated[float, Field(gt=0, le=15)] = 10
    expected_revision: Revision | None = None
    request_id: RequestId | None = None


class PhotoExportOptions(StrictParams):
    suffix: Literal[".png", ".jpg", ".jpeg", ".tif", ".tiff"] = ".png"
    bit_depth: Annotated[int, Field(json_schema_extra={"enum": [8, 16]})] = 8
    quality: Annotated[int, Field(ge=1, le=100)] = 95
    keep_gps: bool = False

    @field_validator("bit_depth")
    @classmethod
    def supported_bit_depth(cls, value):
        if value not in (8, 16):
            raise ValueError("bit_depth должен быть целым числом 8 или 16.")
        return value

    @model_validator(mode="after")
    def jpeg_precision(self):
        if self.suffix in (".jpg", ".jpeg") and self.bit_depth != 8:
            raise ValueError("JPEG поддерживает только bit_depth=8; для 16 бит выберите PNG/TIFF.")
        return self


class ExportMediaParams(SessionParams):
    media_ids: MediaIds
    directory: LocalPath
    expected_revisions: dict[Identifier, Revision] | None = None
    request_id: RequestId | None = None
    options: PhotoExportOptions | None = None
    video_suffix: Literal[".mp4", ".mov"] = ".mp4"

    @model_validator(mode="after")
    def distinct_export_targets(self):
        _distinct_media(self.media_ids)
        if self.expected_revisions is not None and set(self.expected_revisions) != set(
            self.media_ids
        ):
            raise ValueError("expected_revisions должен содержать ровно все выбранные media_ids.")
        return self


class JobParams(SessionParams):
    job_id: Identifier


class SaveProjectParams(SessionParams):
    path: LocalPath
    overwrite: bool = False


class LoadProjectParams(SessionParams):
    path: LocalPath
    request_id: RequestId | None = None


class ShowWindowParams(SessionParams):
    visible: bool


def _distinct_media(media_ids: list[str]) -> None:
    if len(media_ids) != len(set(media_ids)):
        raise ValueError("Каждый media_id должен присутствовать только один раз.")


PARAM_MODELS = {
    "connect_session": EmptyParams,
    "get_capabilities": EmptyParams,
    "get_state": SessionParams,
    "list_media": ListMediaParams,
    "import_media": ImportMediaParams,
    "select_media": SelectMediaParams,
    "set_grade": SetGradeParams,
    "set_reference": SetReferenceParams,
    "render_preview": RenderPreviewParams,
    "export_media": ExportMediaParams,
    "get_job": JobParams,
    "cancel_job": JobParams,
    "save_project": SaveProjectParams,
    "load_project": LoadProjectParams,
    "show_window": ShowWindowParams,
}

TOOL_DESCRIPTIONS = {
    "connect_session": "Подключить приложение и открыть окно; вернуть session_id.",
    "get_capabilities": "Прочитать операции, пресеты, схемы и ограничения без запуска окна.",
    "get_state": "Прочитать общее состояние проекта, выбранного медиа и очереди заданий.",
    "list_media": "Получить страницу media_id, ревизий, метаданных и предупреждений.",
    "import_media": "Поставить импорт явно указанных локальных файлов в очередь; вернуть job_id.",
    "select_media": "Выбрать конкретный media_id и время кадра в общем окне приложения.",
    "set_grade": "Атомарно обновить настройки media_ids при совпадении всех ревизий.",
    "set_reference": "Установить общий эталон из фотографии/обработанного кадра либо очистить его.",
    "render_preview": "Поставить короткое превью указанного media_id в очередь и вернуть job_id.",
    "export_media": "По явной команде экспортировать media_ids; вернуть job_id.",
    "get_job": "Прочитать статус, этап, прогресс, предупреждения и пути результата одного задания.",
    "cancel_job": "Отменить задание, сохранив готовые результаты.",
    "save_project": "Сохранить совместимый .fcsproj по указанному локальному пути.",
    "load_project": "Загрузить проект через очередь, если других незавершённых заданий нет.",
    "show_window": "Показать или скрыть окно приложения, сохранив проект и обработку в фоне.",
}


def tool_schemas() -> dict[str, dict]:
    return {name: model.model_json_schema() for name, model in PARAM_MODELS.items()}


def _validation_message(issue: dict) -> str:
    messages = {
        "extra_forbidden": "Неизвестное поле; удалите его из запроса.",
        "missing": "Обязательное поле отсутствует.",
        "float_type": "Требуется число, а не строка или логическое значение.",
        "int_type": "Требуется целое число без преобразования строк или дробей.",
        "bool_type": "Требуется логическое значение true или false.",
        "string_type": "Требуется строка.",
        "dict_type": "Требуется JSON-объект.",
        "list_type": "Требуется JSON-массив.",
        "finite_number": "Требуется конечное число; NaN и бесконечности запрещены.",
        "literal_error": "Значение отсутствует в списке поддерживаемых вариантов.",
        "greater_than": "Значение должно быть больше нижней границы схемы.",
        "greater_than_equal": "Значение меньше допустимой нижней границы.",
        "less_than_equal": "Значение превышает допустимую верхнюю границу.",
        "string_too_short": "Строка слишком короткая или пустая.",
        "string_too_long": "Строка превышает допустимую длину.",
        "string_pattern_mismatch": "Идентификатор имеет недопустимый формат.",
        "too_short": "В списке недостаточно элементов.",
        "too_long": "Список превышает допустимое число элементов.",
        "model_type": "Параметры должны быть JSON-объектом.",
    }
    if issue["type"] == "value_error":
        return issue["msg"].removeprefix("Value error, ")
    return messages.get(issue["type"], "Значение не соответствует схеме параметра.")


def validate_params(method: str, params: dict) -> dict:
    """Validate before side effects and return detached JSON-compatible parameters."""
    from .errors import ControlError

    model = PARAM_MODELS.get(method)
    if model is None:
        raise ControlError(
            "METHOD_NOT_FOUND",
            "Неизвестная операция управления.",
            "Получите get_capabilities и выберите поддерживаемую операцию.",
        )
    try:
        parsed = model.model_validate(params, strict=True)
        normalized = parsed.model_dump(mode="json")
        for name in ("grade", "local"):
            patch = getattr(parsed, name, None)
            if patch is not None:
                normalized[name] = patch.model_dump(mode="json", exclude_unset=True)
        # The controller distinguishes omitted optional fields from supplied data.
        normalized = {key: value for key, value in normalized.items() if value is not None}
        paths = list(normalized.get("paths", []))
        paths += [normalized[key] for key in ("path", "directory") if key in normalized]
        for value in paths:
            if (
                "\0" in value
                or "://" in value
                or value.startswith(("//", "\\\\"))
                or not Path(value).is_absolute()
            ):
                raise ValueError("Укажите абсолютный путь к локальному файлу или каталогу.")
        return normalized
    except (ValidationError, ValueError) as exc:
        details = (
            [
                {
                    "field": ".".join(str(part) for part in issue["loc"]),
                    "type": issue["type"],
                    "message": _validation_message(issue),
                }
                for issue in exc.errors(include_input=False, include_url=False)
            ]
            if isinstance(exc, ValidationError)
            else [{"field": "path", "type": "local_path", "message": str(exc)}]
        )
        raise ControlError(
            "INVALID_PARAMS",
            "Параметры операции не соответствуют поддерживаемой схеме.",
            "Проверьте названные поля и диапазоны через get_capabilities; затем повторите запрос.",
            {"fields": details},
        ) from exc


def get_capabilities() -> dict:
    """Static capabilities only: no app launch, media access or optional model load."""
    return {
        "api_version": API_VERSION,
        "application_version": __version__,
        "operations": list(PARAM_MODELS),
        "presets": {key: value.to_dict() for key, value in PRESETS.items()},
        "grade_ranges": {key: list(value) for key, value in RANGES.items()},
        "local_ranges": {key: list(value) for key, value in LOCAL_RANGES.items()},
        "local_defaults": deepcopy(LOCAL_DEFAULTS),
        "parameter_schemas": tool_schemas(),
        "limits": {
            "message_bytes": MAX_MESSAGE_BYTES,
            "group_size": MAX_GROUP_SIZE,
            "list_page_size": MAX_PAGE_SIZE,
            "preview_seconds": 15,
            "preview_max_edge": 1280,
            "photo_max_pixels": 40_000_000,
            "video_max_pixels": 4096 * 2304,
            "sequential_heavy_jobs": True,
        },
        "formats": {
            "photos": [".jpg", ".jpeg", ".png", ".tif", ".tiff"],
            "video": [".mp4", ".mov"],
            "photo_output_bit_depth": [8, 16],
            "video_output": "H.264 SDR Rec.709 с исходными аудиопакетами и временными метками",
            "unsupported": ["RAW", "HDR/PQ/HLG", "монтаж", "генеративные изменения"],
        },
        "privacy": {
            "offline": True,
            "media_contents_in_responses": False,
            "visual_review": "Просматривайте реальные кадры и превью в окне приложения.",
        },
        "job_states": ["queued", "running", "succeeded", "failed", "cancelled", "interrupted"],
    }
