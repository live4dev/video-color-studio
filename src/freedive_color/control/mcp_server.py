"""Official SDK 2.x stdio adapter. Only metadata and local paths cross MCP.

The application is owned by the desktop runtime, not by this subprocess.
Initialization, tools/list and static capabilities never start that runtime.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Callable

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations

from freedive_color import __version__

from .errors import ControlError
from .schemas import (
    MAX_MESSAGE_BYTES,
    TOOL_DESCRIPTIONS,
    get_capabilities,
    tool_schemas,
    validate_params,
)

SERVER_INSTRUCTIONS = """Freedive Color Studio — управление локальным проектом API 1.
1. Вызовите connect_session только при первом намеренном подключении к приложению.
   Инициализация и список инструментов не открывают окно; get_capabilities доступен без приложения.
2. Получите get_state и list_media, определите конкретные media_id и актуальные revision.
   Не используйте текущий выбор окна как неявную цель обработки. Если указание файла
   неоднозначно, уточните его у пользователя до изменения настроек.
3. Переведите словесное описание стиля в поддерживаемый preset и параметры grade/local
   из get_capabilities. Примените set_grade с expected_revisions для всех media_ids.
   Непереданные локальные параметры и ручные маски сохраняются. При REVISION_CONFLICT
   перечитайте состояние; не перезаписывайте более свежие ручные изменения вслепую.
4. Создайте короткое render_preview (до 15 секунд), отслеживайте job_id через get_job.
   Учитывайте замечания пользователя после просмотра превью в окне приложения.
5. Экспортируйте указанные media_ids по явной команде пользователя через export_media.
   Повторное подтверждение уже запрошенного экспорта не требуется. Длительные операции
   быстро возвращают job_id; не выдавайте queued/running за готовый результат.
   При повторе постановки используйте прежний request_id, чтобы не дублировать задание.
6. Оригиналы, геометрия и существующие результаты сохраняются. Обработка выполняется
   локально; монтаж, музыка, генеративное изменение сцены и облачные операции не поддерживаются.
7. Ответы содержат только настройки, метаданные, статистики, предупреждения и локальные пути.
   Не запрашивайте и не передавайте пиксели, миниатюры, звук или содержимое медиа в модель.
   Пользователь оценивает изображение в приложении. Статистики анализа не доказывают
   визуальное качество, точность масок или восстановление потерянного под водой цвета.
8. Закрытие адаптера не отменяет задания. show_window visible=false скрывает окно и
   сохраняет работу процесса; отмена выполняется явно через cancel_job.
"""

_READ_ONLY = {"get_capabilities", "get_state", "list_media", "get_job"}
_IDEMPOTENT = _READ_ONLY | {"connect_session", "select_media", "show_window", "cancel_job"}
_FORBIDDEN_FIELDS = {
    "pixels",
    "rgb",
    "rgba",
    "image_bytes",
    "audio_bytes",
    "thumbnail",
    "thumbnails",
    "thumbnail_data",
    "base64",
    "data_base64",
    "audio_samples",
    "waveform",
}
_SUCCESS_MESSAGES = {
    "connect_session": "Приложение подключено; окно проекта открыто.",
    "get_capabilities": "Возможности и строгие схемы параметров прочитаны без запуска приложения.",
    "get_state": "Состояние общего проекта прочитано.",
    "list_media": "Страница медиа и актуальные ревизии прочитаны.",
    "import_media": "Задание импорта принято; проверяйте его через get_job.",
    "select_media": "Указанное медиа выбрано в окне приложения.",
    "set_grade": "Настройки выбранных медиа обновлены.",
    "set_reference": "Общий эталон проекта обновлён.",
    "render_preview": "Задание превью принято; проверяйте его через get_job.",
    "export_media": "Задание экспорта принято; проверяйте его через get_job.",
    "get_job": "Состояние задания прочитано.",
    "cancel_job": "Запрос отмены задания обработан.",
    "save_project": "Проект сохранён.",
    "load_project": "Запрос загрузки проекта принят.",
    "show_window": "Видимость окна обновлена; состояние проекта сохранено.",
}


def _json_only(value, *, depth: int = 0) -> None:
    """Reject media buffers/content objects instead of silently encoding them."""
    if depth > 40:
        raise ValueError("Слишком глубокая структура ответа.")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Ответ содержит неконечное число.")
        return
    if type(value) is str:
        if value.lstrip().lower().startswith(("data:image/", "data:audio/", "data:video/")):
            raise ValueError("Содержимое медиа не передаётся через MCP.")
        return
    if type(value) is list:
        for item in value:
            _json_only(item, depth=depth + 1)
        return
    if type(value) is dict:
        if value.get("type") in ("image", "audio") and (
            "data" in value or "mimeType" in value or "mime_type" in value
        ):
            raise ValueError("Объекты содержимого медиа не передаются через MCP.")
        for key, item in value.items():
            if type(key) is not str or key.lower() in _FORBIDDEN_FIELDS:
                raise ValueError("Ответ содержит запрещённое поле медиа.")
            _json_only(item, depth=depth + 1)
        return
    raise ValueError("Ответ содержит бинарные данные или неподдерживаемый тип.")


def _safe_payload(payload: dict) -> dict:
    _json_only(payload)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_MESSAGE_BYTES - 4096:
        raise ValueError("Ответ превышает 1 МиБ; используйте постраничное чтение.")
    return payload


def _error_result(code: str, message: str, hint: str, details=None) -> CallToolResult:
    if (
        type(code) is not str
        or len(code) > 64
        or type(message) is not str
        or len(message) > 2048
        or type(hint) is not str
        or len(hint) > 2048
    ):
        code = "INVALID_RESPONSE"
        message = "Приложение вернуло некорректное описание ошибки."
        hint = "Проверьте состояние приложения и повторите запрос."
        details = {}
    error = {"code": code, "message": message, "hint": hint, "details": details or {}}
    payload = {"ok": False, "message": message, "error": error}
    try:
        _safe_payload(payload)
    except (ValueError, TypeError, OverflowError):
        error["details"] = {}
        try:
            _safe_payload(payload)
        except (ValueError, TypeError, OverflowError):
            return _error_result(
                "INVALID_RESPONSE",
                "Приложение вернуло запрещённое содержимое вместо описания ошибки.",
                "Просматривайте медиа только в окне приложения.",
            )
    return CallToolResult(
        content=[TextContent(text=f"{message} {hint}".strip())],
        structured_content=payload,
        is_error=True,
    )


def create_server(client_factory: Callable | None = None) -> Server:
    """Create protocol handlers without discovering/starting the desktop app."""
    schemas = tool_schemas()
    client = None

    async def list_tools(_context, params):
        if params is not None and params.cursor is not None:
            # There are only fifteen tools; no cursor is generated by this server.
            return ListToolsResult(tools=[])
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    description=TOOL_DESCRIPTIONS[name],
                    input_schema=schema,
                    annotations=ToolAnnotations(
                        read_only_hint=name in _READ_ONLY,
                        destructive_hint=False,
                        idempotent_hint=name in _IDEMPOTENT,
                        open_world_hint=False,
                    ),
                )
                for name, schema in schemas.items()
            ]
        )

    async def call_tool(_context, params):
        nonlocal client
        try:
            arguments = validate_params(params.name, params.arguments or {})
            if params.name == "get_capabilities":
                result = get_capabilities()
            else:
                if client is None:
                    if client_factory is None:
                        from .client import ApplicationClient

                        client = ApplicationClient()
                    else:
                        client = client_factory()
                result = await anyio.to_thread.run_sync(client.call, params.name, arguments)
            if type(result) is not dict:
                raise ValueError("Приложение вернуло ответ неподдерживаемого типа.")
            message = result.get("message", _SUCCESS_MESSAGES[params.name])
            if type(message) is not str or len(message) > 2048:
                raise ValueError("Пояснение ответа должно быть строкой.")
            payload = _safe_payload({**result, "ok": True, "message": message})
            return CallToolResult(
                content=[TextContent(text=message)], structured_content=payload, is_error=False
            )
        except ControlError as exc:
            return _error_result(exc.code, exc.message, exc.hint, exc.details)
        except Exception as exc:
            # RemoteControlError intentionally shares the structured error contract
            # without requiring Qt/client imports during MCP initialization.
            if all(hasattr(exc, name) for name in ("code", "message", "hint", "details")):
                return _error_result(exc.code, exc.message, exc.hint, exc.details)
            if isinstance(exc, (ValueError, TypeError, OverflowError)):
                return _error_result(
                    "INVALID_RESPONSE",
                    "Приложение вернуло неподдерживаемый или слишком большой ответ.",
                    "Повторите запрос меньшей страницы; содержимое медиа передавать запрещено.",
                )
            logging.getLogger(__name__).exception("Local application request failed")
            return _error_result(
                "INTERNAL_ERROR",
                "Не удалось выполнить локальную операцию управления.",
                "Проверьте состояние приложения и повторите запрос; оригиналы не изменялись.",
            )

    return Server(
        "Freedive Color Studio",
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        get_tool_input_schema=schemas.get,
    )


async def serve(client_factory: Callable | None = None) -> None:
    server = create_server(client_factory)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    # SDK 2.x stdio_server additionally diverts native/accidental stdout to stderr
    # after claiming its protocol descriptor. Never print diagnostics to stdout.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    anyio.run(serve)


if __name__ == "__main__":
    main()
