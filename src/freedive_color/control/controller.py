"""Authoritative project state and one cancellable queue, independent of widgets.

Only trusted in-process GUI callbacks receive pixel-bearing objects. Wire replies and
the restart journal are explicitly JSON metadata; no object introspection is used.
"""

import copy
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from freedive_color.core.analyze import SceneAnalysis, analyze_scene
from freedive_color.core.batch import BatchResult, plan_batch
from freedive_color.core.errors import Cancelled, StudioError
from freedive_color.core.image_pipeline import (
    CancellationToken,
    Preview,
    create_image_reference,
    export_photo,
    render_rgb,
)
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import MaskEdits, build_masks
from freedive_color.core.media_io import encode_photo, load_photo, signature
from freedive_color.core.presets import GradeSettings, get_preset
from freedive_color.core.project import json_bytes, sha256_file
from freedive_color.core.reference import PaletteReference, create_reference
from freedive_color.core.session import (
    ExportOptions,
    Session,
    SessionItem,
    _local_path,
    load_session,
    save_session,
)
from freedive_color.core.video_pipeline import (
    export_video,
    probe_video,
    read_video_frame,
)

from .errors import ControlError

logger = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


def _id(prefix):
    return prefix + uuid.uuid4().hex


def _clone(value):
    return copy.deepcopy(value)


def _processing(item):
    return (
        item.settings,
        item.local_settings or LocalSettings().to_dict(),
        item.mask_edits or MaskEdits().to_dict(),
    )


def _metadata(value):
    """A strict allowlist, deliberately rejecting arrays, bytes and arbitrary objects."""
    if isinstance(value, Preview):
        return {
            "source_path": str(value.photo.path),
            "width": value.photo.width,
            "height": value.photo.height,
            "warnings": list(value.local_warnings),
        }
    if isinstance(value, PaletteReference):
        return value.to_dict()
    if isinstance(value, BatchResult):
        return value.to_dict()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _metadata(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        # RGB/mask tuples from trusted UI renders carry no public result at all.
        if any(type(child).__module__.startswith("numpy") for child in value):
            return {"type": "render_result", "message": "Результат доступен в приложении."}
        return [_metadata(child) for child in value]
    return {"type": type(value).__name__, "message": "Результат доступен в приложении."}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _summary(value):
    payload = _encoded(value)
    return {
        "summary_only": True,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "json_bytes": len(payload),
    }


def _wire(value, key=""):
    """Bound large statistical/mask trees; the exact recipe stays on local disk.

    GUI reads get_job/read_record directly and therefore keeps the complete plan.
    No media bytes are serialized to construct these summaries.
    """
    if (
        key in {"plan", "shot_plan", "mask_edits"}
        and value is not None
        and len(_encoded(value)) > 24_000
    ):
        result = _summary(value)
        if isinstance(value, dict):
            for name in ("duration", "start_time", "sampled_frames", "frame_count"):
                if name in value:
                    result[name] = value[name]
            for name in ("anchors", "cuts", "strokes", "waterline"):
                if isinstance(value.get(name), list):
                    result[name + "_count"] = len(value[name])
        return result
    if isinstance(value, dict):
        result = {name: _wire(child, name) for name, child in value.items()}
        if len(_encoded(result)) > 400_000 and key not in {"", "job", "result"}:
            return _summary(result)
        return result
    if isinstance(value, list):
        return [_wire(child, key + "_item") for child in value]
    if isinstance(value, str) and len(value.encode()) > 64_000:
        return _summary(value)
    return value


def _error(exc):
    if isinstance(exc, ControlError):
        return exc
    if isinstance(exc, Cancelled):
        return ControlError("CANCELLED", "Задание отменено.", "Завершённые результаты сохранены.")
    if isinstance(exc, FileNotFoundError):
        return ControlError(
            "FILE_NOT_FOUND", "Локальный файл не найден.", "Повторите импорт файла."
        )
    if isinstance(exc, (StudioError, OSError)):
        message = str(exc)
        code = "PROCESSING_ERROR"
        if "уже существ" in message:
            code = "OUTPUT_BUSY"
        elif "изменил" in message or "сумма" in message:
            code = "SOURCE_CHANGED"
        elif "не поддерж" in message or "неподдерж" in message:
            code = "UNSUPPORTED_MEDIA"
        return ControlError(code, message, "Проверьте исходник, параметры и выходную папку.")
    logger.error("Control job failed (%s)", type(exc).__name__)
    return ControlError("INTERNAL_ERROR", "Не удалось выполнить задание.", "Повторите операцию.")


@dataclass(frozen=True)
class MediaRecord:
    media_id: str
    item: SessionItem
    revision: int = 0
    status: str = "ready"
    warnings: tuple[str, ...] = ()
    preview: dict | None = None

    def to_dict(self):
        return {
            "media_id": self.media_id,
            "path": str(self.item.path),
            "kind": "video" if self.item.path.suffix.lower() in VIDEO_SUFFIXES else "photo",
            "revision": self.revision,
            "status": self.status,
            "included": self.item.included,
            "settings": self.item.settings.to_dict(),
            "local_settings": _clone(self.item.local_settings),
            "mask_edits": _clone(self.item.mask_edits),
            "analysis": _clone(self.item.analysis),
            "metadata": _clone(self.item.media_info),
            "source_sha256": self.item.source_sha256,
            "warnings": list(self.warnings),
            "preview": _clone(self.preview),
        }


class _Signals(QObject):
    progress = Signal(str, int, str)
    done = Signal(str, object, object)


class _Task(QRunnable):
    def __init__(self, job_id, operation, token):
        super().__init__()
        self.job_id, self.operation, self.token = job_id, operation, token
        self.signals = _Signals()

    def run(self):
        try:
            self.token.check()
            result = self.operation(
                self.token,
                lambda value, message: self.signals.progress.emit(self.job_id, value, message),
            )
            self.signals.done.emit(self.job_id, result, None)
        except Exception as exc:
            self.signals.done.emit(self.job_id, None, _error(exc))


@dataclass
class _Work:
    task: _Task
    success: Callable | None = None
    error: Callable | None = None
    progress: Callable | None = None
    finished: Callable | None = None
    media_revisions: dict = field(default_factory=dict)


class ProjectController(QObject):
    state_changed = Signal(object)
    job_changed = Signal(str)

    def __init__(self, session=None, *, journal_path=None, temp_root=None, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.session_id = _id("s_")
        self.project_revision = 0
        self.project_path = None
        self._base = _clone(session or Session())
        self._records = {}
        self._positions = {}
        self._selected = None
        self._jobs = {}
        self._work = {}
        self._queue = deque()
        self._running = None
        self._requests = {}
        self._closing = False
        self._owned_temp = set()
        self._temp_root = _local_path(temp_root) if temp_root is not None else None
        self._journal = _local_path(journal_path) if journal_path is not None else None
        self._last_journal = 0
        self._reference_generation = 0
        self._initialize_records(self._base)
        self._load_journal()

    def _initialize_records(self, session):
        self._records = {}
        for item in session.items:
            media_id = _id("m_")
            self._records[media_id] = MediaRecord(media_id, _clone(item))
        ids = list(self._records)
        self._selected = ids[session.selected_index] if session.selected_index is not None else None
        self._positions = {}

    def _emit(self, kind, media_ids=(), **data):
        self.state_changed.emit(
            {"kind": kind, "session_id": self.session_id, "media_ids": list(media_ids), **data}
        )

    def records(self):
        return tuple(_clone(record) for record in self._records.values())

    @property
    def time_seconds(self):
        return self._positions.get(self._selected, 0.0)

    def update_view_position(self, media_id, seconds):
        """Trusted playback telemetry; does not rerender or emit a selection request."""
        if media_id != self._selected or media_id not in self._records:
            return
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
        ):
            raise ControlError("INVALID_PARAMS", "Позиция просмотра должна быть конечным числом.")
        duration = self._records[media_id].item.media_info.get("duration")
        self._positions[media_id] = (
            max(0.0, min(float(seconds), float(duration))) if duration else max(0.0, float(seconds))
        )

    def read_record(self, media_id):
        try:
            return _clone(self._records[media_id])
        except KeyError as exc:
            raise ControlError(
                "MEDIA_NOT_FOUND", "Файл с таким media_id не найден.", "Получите list_media заново."
            ) from exc

    def media_id_for_path(self, path):
        path = _local_path(path)
        return next(
            (key for key, record in self._records.items() if record.item.path == path), None
        )

    def get_session(self):
        ids = list(self._records)
        return replace(
            _clone(self._base),
            items=tuple(_clone(record.item) for record in self._records.values()),
            selected_index=ids.index(self._selected) if self._selected in ids else None,
        )

    def add_item(self, item, select=False):
        if not isinstance(item, SessionItem):
            raise ControlError("INVALID_PARAMS", "Ожидается запись SessionItem.")
        existing = self.media_id_for_path(item.path)
        if existing:
            return existing
        media_id = _id("m_")
        self._records[media_id] = MediaRecord(media_id, _clone(item))
        if select or self._selected is None:
            self._selected = media_id
        self.project_revision += 1
        self._emit("media_added", [media_id])
        return media_id

    def replace_item(self, media_id, item, expected_revision=None):
        old = self.read_record(media_id)
        if not isinstance(item, SessionItem) or item.path != old.item.path:
            raise ControlError("INVALID_PARAMS", "Исходник записи нельзя подменить.")
        self._check_revisions(
            [media_id], {media_id: expected_revision} if expected_revision is not None else None
        )
        changed = _processing(item) != _processing(old.item)
        if item != old.item:
            self._records[media_id] = replace(
                old,
                item=_clone(item),
                revision=old.revision + int(changed),
                preview=None if changed else old.preview,
            )
            self.project_revision += 1
            self._emit("grade_changed" if changed else "media_updated", [media_id])
        return self.read_record(media_id)

    def replace_session(self, session, project_path=None):
        if not self.is_idle():
            raise ControlError(
                "PROJECT_BUSY", "В очереди есть задания.", "Дождитесь завершения или отмените их."
            )
        self._replace_session(session, project_path)

    def save_project_from_ui(self, path, overwrite=False):
        """Trusted immediate UI Save; uses exactly the same project serializer."""
        path = _local_path(path)
        save_session(path, self.get_session(), overwrite=overwrite)
        self.project_path = path
        self._emit("project_saved", path=str(path))

    def _replace_session(self, session, project_path):
        self._base = _clone(session)
        self.project_path = _local_path(project_path) if project_path else None
        self.session_id = _id("s_")
        self.project_revision = 0
        self._initialize_records(session)
        self._emit("session_loaded", list(self._records))

    def update_session_options(self, **options):
        allowed = {
            "reference_path",
            "reference_data",
            "temp_directory",
            "output_directory",
            "export_options",
            "selected_index",
        }
        if set(options) - allowed:
            raise ControlError("INVALID_PARAMS", "Неизвестные параметры проекта.")
        if isinstance(options.get("export_options"), dict):
            options["export_options"] = ExportOptions.from_dict(options["export_options"])
        current = self.get_session()
        updated = replace(current, **_clone(options))
        if "selected_index" in options:
            ids = list(self._records)
            self._selected = (
                ids[updated.selected_index] if updated.selected_index is not None else None
            )
        changed_reference = (current.reference_path, current.reference_data) != (
            updated.reference_path,
            updated.reference_data,
        )
        self._base = updated
        if changed_reference:
            self._reference_generation += 1
            for key, record in self._records.items():
                self._records[key] = replace(record, revision=record.revision + 1, preview=None)
        if updated != current:
            self.project_revision += 1
            self._emit(
                "reference_changed" if changed_reference else "options_changed",
                list(self._records) if changed_reference else (),
            )

    def _check_session(self, session_id):
        if session_id != self.session_id:
            raise ControlError(
                "SESSION_NOT_FOUND",
                "Идентификатор проекта устарел.",
                "Вызовите connect_session и получите новые идентификаторы.",
            )

    def _check_revisions(self, media_ids, expected):
        records = [self.read_record(key) for key in media_ids]
        if expected is not None:
            if set(expected) != set(media_ids):
                raise ControlError("INVALID_PARAMS", "Укажите ревизию каждого выбранного файла.")
            conflicts = {
                r.media_id: {"expected": expected[r.media_id], "actual": r.revision}
                for r in records
                if expected[r.media_id] != r.revision
            }
            if conflicts:
                raise ControlError(
                    "REVISION_CONFLICT",
                    "Параметры были изменены в приложении или другом запросе.",
                    "Получите актуальные ревизии через list_media и повторите команду.",
                    conflicts,
                )
        return records

    def get_job(self, job_id):
        if job_id not in self._jobs:
            raise ControlError(
                "JOB_NOT_FOUND", "Задание не найдено.", "Получите состояние очереди."
            )
        return _clone(self._jobs[job_id])

    def is_idle(self):
        return self._running is None and not self._queue

    def shutdown(self):
        self._closing = True
        for job_id in list(self._work):
            self._cancel(job_id)
        if self.is_idle():
            self._clean_temp()

    def _clean_temp(self):
        for path in self._owned_temp:
            shutil.rmtree(path, ignore_errors=True)
        self._owned_temp.clear()

    def _job_temp(self):
        parent = self._base.temp_directory or self._temp_root
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="freedive-control-", dir=parent))
        self._owned_temp.add(path)
        return path

    def _snapshot(self, media_ids):
        return {
            "session_id": self.session_id,
            "media": [self.read_record(key).to_dict() for key in media_ids],
            "reference": _clone(self._base.reference_data),
            "reference_path": str(self._base.reference_path) if self._base.reference_path else None,
            "temp_directory": str(self._base.temp_directory) if self._base.temp_directory else None,
        }

    def submit_ui_job(
        self,
        kind,
        operation,
        *,
        media_ids=(),
        on_success=None,
        on_error=None,
        on_progress=None,
        on_finished=None,
        request_id=None,
        snapshot=None,
    ):
        return self._enqueue(
            "ui_" + kind,
            operation,
            media_ids,
            request_id=request_id,
            success=on_success,
            error=on_error,
            progress=on_progress,
            finished=on_finished,
            snapshot=snapshot,
        )

    def _enqueue(
        self,
        kind,
        operation,
        media_ids=(),
        *,
        request_id=None,
        success=None,
        error=None,
        progress=None,
        finished=None,
        snapshot=None,
        fingerprint=None,
    ):
        if self._closing:
            raise ControlError(
                "APP_CLOSING", "Приложение завершает работу.", "Откройте приложение снова."
            )
        job_id = _id("j_")
        now = time.time()
        token = CancellationToken()
        task = _Task(job_id, operation, token)
        task.signals.progress.connect(self._progress)
        task.signals.done.connect(self._done)
        revisions = {key: self.read_record(key).revision for key in media_ids}
        captured = self._snapshot(media_ids)
        if snapshot is not None:
            captured["operation"] = _metadata(snapshot)
        self._jobs[job_id] = {
            "job_id": job_id,
            "session_id": self.session_id,
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Ожидает очереди.",
            "media_ids": list(media_ids),
            "revisions": revisions,
            "snapshot": captured,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "request_id": request_id,
            "warnings": [],
            "result": None,
            "error": None,
            "cancel_requested": False,
            "stale_preview": False,
        }
        self._work[job_id] = _Work(task, success, error, progress, finished, revisions)
        if request_id and fingerprint:
            self._requests[(self.session_id, kind, request_id)] = (fingerprint, job_id)
        self._queue.append(job_id)
        self._persist()
        self.job_changed.emit(job_id)
        self._dispatch()
        return job_id

    def _dispatch(self):
        if self._running is not None or self._closing:
            if self._closing and self.is_idle():
                self._clean_temp()
            return
        while self._queue:
            job_id = self._queue.popleft()
            if self._jobs[job_id]["status"] != "queued":
                continue
            self._running = job_id
            job = self._jobs[job_id]
            job.update(
                status="running", stage="starting", started_at=time.time(), message="Обработка…"
            )
            self._persist()
            self.job_changed.emit(job_id)
            self.pool.start(self._work[job_id].task)
            return
        if self._closing:
            self._clean_temp()

    @Slot(str, int, str)
    def _progress(self, job_id, value, message):
        if job_id not in self._work:
            return
        job = self._jobs[job_id]
        job.update(
            progress=max(0, min(100, int(value))),
            stage=message,
            message=message,
            updated_at=time.time(),
        )
        work = self._work[job_id]
        if work.progress:
            work.progress(job["progress"], message)
        if time.monotonic() - self._last_journal > 0.5:
            self._persist()
        self.job_changed.emit(job_id)

    @Slot(str, object, object)
    def _done(self, job_id, result, error):
        work = self._work.get(job_id)
        if work is None:
            return
        job = self._jobs[job_id]
        stale = job["session_id"] != self.session_id or any(
            key not in self._records or self._records[key].revision != revision
            for key, revision in work.media_revisions.items()
        )
        is_preview = "preview" in job["kind"] or job["kind"] in {"ui_view", "ui_render", "ui_load"}
        job["stale_preview"] = bool(stale and is_preview)
        try:
            if (
                error is None
                and work.task.token.event.is_set()
                and not isinstance(result, (dict, BatchResult))
            ):
                error = _error(Cancelled())
            if error is None:
                public_result = _metadata(result)
                cancelled = isinstance(public_result, dict) and public_result.get("cancelled")
                failed = isinstance(public_result, dict) and public_result.get("failed", 0)
                status = "cancelled" if cancelled else "failed" if failed else "succeeded"
                job.update(
                    status=status,
                    progress=100,
                    stage="finished",
                    message="Задание отменено; завершённые результаты сохранены."
                    if cancelled
                    else "В части файлов произошла ошибка."
                    if failed
                    else "Задание завершено.",
                    result=public_result,
                )
                if work.success and not (stale and is_preview):
                    work.success(result)
                    job["result"] = _metadata(result)
                if failed:
                    job["error"] = ControlError(
                        "PARTIAL_FAILURE",
                        "Часть файлов не обработана.",
                        "Готовые результаты сохранены; подробности находятся в result.records.",
                    ).to_dict()
            else:
                job.update(
                    status="cancelled" if error.code == "CANCELLED" else "failed",
                    error=error.to_dict(),
                    message=error.message,
                    stage="finished",
                )
                if work.error:
                    work.error(error.message, error.code == "CANCELLED")
        except Exception:
            logger.exception("GUI completion callback failed")
            job["warnings"].append(
                "Не удалось обновить представление окна; результаты задания сохранены."
            )
        finally:
            job.update(updated_at=time.time(), finished_at=time.time())
            self._work.pop(job_id, None)
            if self._running == job_id:
                self._running = None
            self._persist()
            self.job_changed.emit(job_id)
            try:
                if work.finished:
                    work.finished()
            except Exception:
                logger.exception("GUI finalization callback failed")
            finally:
                self._dispatch()

    def _cancel(self, job_id):
        job = self.get_job(job_id)
        if job["status"] in TERMINAL:
            return
        work = self._work[job_id]
        work.task.token.cancel()
        self._jobs[job_id]["cancel_requested"] = True
        if job["status"] == "queued":
            self._queue.remove(job_id)
            self._done(job_id, None, _error(Cancelled()))
        else:
            self._persist()
            self.job_changed.emit(job_id)

    def _persist(self):
        self._last_journal = time.monotonic()
        if self._journal is None:
            return
        recent = list(self._jobs.values())[-200:]
        payload = json_bytes({"schema_version": 1, "kind": "freedive_job_journal", "jobs": recent})
        try:
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".jobs-", dir=self._journal.parent)
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._journal)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not persist local job journal")

    def _load_journal(self):
        if self._journal is None or not self._journal.is_file():
            return
        try:
            if self._journal.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("Large journal")
            data = json.loads(self._journal.read_text(encoding="utf-8"))
            if data.get("kind") != "freedive_job_journal" or data.get("schema_version") != 1:
                raise ValueError("Unknown journal")
            for job in data["jobs"][-200:]:
                if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
                    continue
                if job.get("status") not in TERMINAL:
                    job.update(
                        status="interrupted",
                        stage="finished",
                        finished_at=time.time(),
                        message="Работа приложения была прервана; автоматический повтор отключён.",
                    )
                self._jobs[job["job_id"]] = job
            self._persist()
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Ignoring invalid local job journal")

    def _state(self):
        statuses = [job["status"] for job in self._jobs.values()]
        return {
            "session_id": self.session_id,
            "project_revision": self.project_revision,
            "project_path": str(self.project_path) if self.project_path else None,
            "media_count": len(self._records),
            "selected_media_id": self._selected,
            "time_seconds": self._positions.get(self._selected, 0),
            "selected": self.read_record(self._selected).to_dict() if self._selected else None,
            "reference": _clone(self._base.reference_data),
            "reference_path": str(self._base.reference_path) if self._base.reference_path else None,
            "temp_directory": str(self._base.temp_directory) if self._base.temp_directory else None,
            "output_directory": str(self._base.output_directory)
            if self._base.output_directory
            else None,
            "export_options": self._base.export_options.to_dict(),
            "queue": {
                "running": self._running,
                "queued": list(self._queue),
                "active_count": statuses.count("running") + statuses.count("queued"),
            },
            "recent_jobs": [
                {key: job.get(key) for key in ("job_id", "kind", "status", "progress", "message")}
                for job in list(self._jobs.values())[-20:]
            ],
        }

    def execute(self, method: str, params: dict) -> dict:
        result = _wire(self._execute(method, params))
        # A job can contain both a complete batch snapshot and its results. Prefer
        # actual output/recipe paths over duplicating the snapshot in a wire reply.
        if len(_encoded(result)) > 850_000 and isinstance(result.get("job"), dict):
            job = result["job"]
            if isinstance(job.get("snapshot"), dict):
                job["snapshot"] = _summary(job["snapshot"])
            records = (
                job.get("result", {}).get("records", [])
                if isinstance(job.get("result"), dict)
                else []
            )
            for row in records:
                if isinstance(row, dict) and "recipe" in row:
                    recipe = row["recipe"]
                    row["recipe"] = {
                        name: recipe[name]
                        for name in ("kind", "settings", "local_settings", "output", "warnings")
                        if isinstance(recipe, dict) and name in recipe
                    }
        if len(_encoded(result)) > 900_000 and isinstance(result.get("media"), list):
            result["media"] = [
                {
                    name: row[name]
                    for name in (
                        "media_id",
                        "revision",
                        "path",
                        "settings",
                        "local_settings",
                        "status",
                    )
                    if name in row
                }
                for row in result["media"]
            ]
        if len(_encoded(result)) > 950_000:
            raise ControlError(
                "RESPONSE_TOO_LARGE",
                "Объём результата превышает размер сообщения.",
                "Уменьшите limit в list_media. Полный рецепт доступен локально по recipe_path.",
            )
        return result

    def _execute(self, method: str, params: dict) -> dict:
        from .schemas import get_capabilities, validate_params

        try:
            params = validate_params(method, params)
            if method == "get_capabilities":
                return {**get_capabilities(), "message": "Возможности локального движка."}
            if method == "connect_session":
                self._emit("window_requested", visible=True)
                return {**self._state(), "message": "Подключено к локальному проекту."}
            if method in {"get_job", "cancel_job"}:
                job = self.get_job(params["job_id"])
                if params["session_id"] not in {self.session_id, job["session_id"]}:
                    self._check_session(params["session_id"])
                if method == "cancel_job":
                    self._cancel(params["job_id"])
                return {
                    "session_id": self.session_id,
                    "job": self.get_job(params["job_id"]),
                    "message": "Состояние задания.",
                }
            self._check_session(params["session_id"])
            if method == "get_state":
                return {**self._state(), "message": "Текущее состояние проекта."}
            if method == "show_window":
                self._emit("window_requested", visible=params["visible"])
                return {
                    "session_id": self.session_id,
                    "visible": params["visible"],
                    "message": "Состояние окна изменено.",
                }
            if method == "list_media":
                return self._list_media(params)
            if method == "select_media":
                record = self.read_record(params["media_id"])
                seconds = params["time_seconds"]
                duration = record.item.media_info.get("duration")
                if duration and seconds > duration:
                    raise ControlError("INVALID_PARAMS", "Позиция находится за концом видео.")
                self._selected = record.media_id
                self._positions[record.media_id] = seconds
                self._emit("selection", [record.media_id], time_seconds=seconds)
                return {
                    "session_id": self.session_id,
                    "media_id": record.media_id,
                    "time_seconds": seconds,
                    "message": "Выбран файл и позиция просмотра.",
                }
            if method == "set_grade":
                return self._set_grade(params)
            if method == "set_reference" and params["source"] == "clear":
                self._check_revisions(list(self._records), params.get("expected_revisions"))
                self._reference_generation += 1
                self.update_session_options(reference_path=None, reference_data=None)
                return {
                    "session_id": self.session_id,
                    "reference": None,
                    "message": "Эталон очищен.",
                }
            # Check idempotency before mutable revisions or file-exists preflight.
            request_id = params.get("request_id")
            fingerprint = hashlib.sha256(
                json.dumps(params, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            identity = (self.session_id, method, request_id)
            if request_id and identity in self._requests:
                prior, job_id = self._requests[identity]
                if prior != fingerprint:
                    raise ControlError(
                        "REQUEST_CONFLICT",
                        "request_id уже использован для другой команды.",
                        "Используйте новый request_id.",
                    )
                return {
                    "session_id": self.session_id,
                    "job_id": job_id,
                    "media_ids": self._jobs[job_id]["media_ids"],
                    "deduplicated": True,
                    "message": "Возвращено существующее задание.",
                }
            handlers = {
                "import_media": self._import_media,
                "render_preview": self._render_preview,
                "export_media": self._export_media,
                "set_reference": self._set_reference,
                "save_project": self._save_project,
                "load_project": self._load_project,
            }
            if method not in handlers:
                raise ControlError(
                    "METHOD_NOT_FOUND", "Операция не поддерживается.", "Получите get_capabilities."
                )
            result = handlers[method](params, fingerprint)
            return {
                "session_id": self.session_id,
                **result,
                "message": "Задание добавлено в общую очередь.",
            }
        except ControlError:
            raise
        except (StudioError, OSError, ValueError, TypeError) as exc:
            raise _error(exc) from exc

    def _list_media(self, params):
        ids = list(self._records)
        cursor = params.get("cursor")
        if cursor:
            try:
                start = ids.index(cursor) + 1
            except ValueError as exc:
                raise ControlError(
                    "INVALID_CURSOR", "Курсор списка устарел.", "Начните list_media без cursor."
                ) from exc
        else:
            start = 0
        limit = params["limit"]
        selected, rows, size = [], [], 0
        for key in ids[start : start + limit]:
            row = _wire(self.read_record(key).to_dict())
            length = len(_encoded(row))
            if selected and size + length > 450_000:
                break
            selected.append(key)
            rows.append(row)
            size += length
        return {
            "session_id": self.session_id,
            "media": rows,
            "next_cursor": selected[-1] if selected and start + len(selected) < len(ids) else None,
            "total": len(ids),
            "message": "Список файлов проекта.",
        }

    def _set_grade(self, params):
        records = self._check_revisions(params["media_ids"], params["expected_revisions"])
        updates = []
        for record in records:
            base = get_preset(params["preset"]) if params.get("preset") else record.item.settings
            grade = GradeSettings.from_dict({**base.to_dict(), **params.get("grade", {})})
            local = record.item.local_settings
            if params.get("local"):
                local = LocalSettings.from_dict(
                    {**LocalSettings.from_dict(local).to_dict(), **params["local"]}
                ).to_dict()
            updates.append(
                (record.media_id, replace(record.item, settings=grade, local_settings=local))
            )
        # No state mutation until every ID, revision and complete resulting settings validate.
        changed_ids = []
        for media_id, item in updates:
            old = self._records[media_id]
            if _processing(item) != _processing(old.item):
                self._records[media_id] = replace(
                    old, item=_clone(item), revision=old.revision + 1, preview=None
                )
                changed_ids.append(media_id)
        if changed_ids:
            self.project_revision += 1
            self._emit("grade_changed", changed_ids)
        return {
            "session_id": self.session_id,
            "media": [self.read_record(key).to_dict() for key, _ in updates],
            "message": "Параметры применены ко всем выбранным файлам.",
        }

    def _launch(self, method, params, fingerprint, operation, ids=(), success=None):
        return self._enqueue(
            method,
            operation,
            ids,
            request_id=params.get("request_id"),
            fingerprint=fingerprint,
            success=success,
            snapshot=params,
        )

    def _check_source(self, item, token):
        token.check()
        if not item.path.is_file():
            raise ControlError("FILE_NOT_FOUND", "Исходник не найден.", "Повторите импорт файла.")
        if item.source_sha256 and sha256_file(item.path) != item.source_sha256:
            raise ControlError(
                "SOURCE_CHANGED",
                "Исходник изменился после импорта.",
                "Повторите импорт перед обработкой.",
            )
        token.check()

    def _import_media(self, params, fingerprint):
        paths = [_local_path(path) for path in params["paths"]]
        for path in paths:
            if not path.is_file():
                raise ControlError(
                    "FILE_NOT_FOUND",
                    "Исходник не найден.",
                    "Укажите существующий локальный файл.",
                    {"path": str(path)},
                )
            if path.suffix.lower() not in PHOTO_SUFFIXES | VIDEO_SUFFIXES:
                raise ControlError(
                    "UNSUPPORTED_MEDIA",
                    "Формат медиа не поддерживается.",
                    "Выберите фото JPEG/PNG/TIFF/WebP или видео MP4/MOV.",
                )
        ids = [self.add_item(SessionItem(path=path)) for path in paths]
        stamps = {key: signature(self.read_record(key).item.path) for key in ids}
        items = {key: self.read_record(key).item for key in ids}

        def operation(token, progress):
            rows = []
            for index, key in enumerate(dict.fromkeys(ids)):
                item = items[key]
                try:
                    token.check()
                    if signature(item.path) != stamps[key]:
                        raise ControlError(
                            "SOURCE_CHANGED", "Исходник изменился после постановки импорта."
                        )
                    digest = sha256_file(item.path)
                    if item.path.suffix.lower() in VIDEO_SUFFIXES:
                        info = probe_video(item.path).to_dict()
                        analysis = None
                    else:
                        photo = load_photo(item.path, max_edge=1440)
                        info, analysis = photo.metadata(), analyze_scene(photo.rgb).to_dict()
                    token.check()
                    if signature(item.path) != stamps[key]:
                        raise ControlError("SOURCE_CHANGED", "Исходник изменился во время импорта.")
                    rows.append(
                        {
                            "media_id": key,
                            "metadata": info,
                            "analysis": analysis,
                            "sha256": digest,
                            "status": "ready",
                        }
                    )
                except Cancelled:
                    raise
                except Exception as exc:
                    rows.append(
                        {"media_id": key, "status": "failed", "error": _error(exc).to_dict()}
                    )
                progress(round((index + 1) / len(ids) * 100), "Импорт медиа…")
            return {
                "records": rows,
                "completed": sum(row["status"] == "ready" for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
            }

        def complete(result):
            for row in result["records"]:
                key = row["media_id"]
                old = self.read_record(key)
                if row["status"] == "ready":
                    item = replace(
                        old.item,
                        media_info=row["metadata"],
                        analysis=row["analysis"],
                        source_sha256=row["sha256"],
                    )
                    self._records[key] = replace(old, item=item, status="ready")
                else:
                    self._records[key] = replace(
                        old, status="failed", warnings=(row["error"]["message"],)
                    )
            self._emit("media_imported", ids)

        job_id = self._launch("import_media", params, fingerprint, operation, ids, complete)
        return {"job_id": job_id, "media_ids": ids}

    def _reference(self):
        return (
            PaletteReference.from_dict(_clone(self._base.reference_data))
            if self._base.reference_data
            else None
        )

    def _reference_context(self):
        reference = self._reference()
        path = self._base.reference_path if reference is None else None
        return reference, path, signature(path) if path else None

    @staticmethod
    def _resolve_reference(context, token):
        reference, path, stamp = context
        token.check()
        if path is not None:
            if signature(path) != stamp:
                raise ControlError(
                    "SOURCE_CHANGED", "Исходник эталона изменился после постановки задания."
                )
            reference = create_image_reference(path)
            if signature(path) != stamp:
                raise ControlError("SOURCE_CHANGED", "Исходник эталона изменился во время чтения.")
        token.check()
        return reference

    def _render_preview(self, params, fingerprint):
        key = params["media_id"]
        expected = (
            {key: params["expected_revision"]}
            if params.get("expected_revision") is not None
            else None
        )
        record = self._check_revisions([key], expected)[0]
        item, reference_context = record.item, self._reference_context()
        folder = self._job_temp()
        video = item.path.suffix.lower() in VIDEO_SUFFIXES
        destination = folder / ("preview.mp4" if video else "preview.png")
        start, duration = params["start_seconds"], params["duration_seconds"]
        temp_directory = self._base.temp_directory

        def operation(token, progress):
            self._check_source(item, token)
            reference = self._resolve_reference(reference_context, token)
            if video:
                report = export_video(
                    item.path,
                    destination,
                    item.settings,
                    local_settings=item.local_settings,
                    mask_edits=item.mask_edits,
                    reference=reference,
                    token=token,
                    progress=progress,
                    preview_start=start,
                    preview_seconds=duration,
                    proxy_edge=1280,
                    plan=item.media_info.get("shot_plan"),
                    temp_directory=temp_directory,
                )
            else:
                photo = load_photo(item.path, max_edge=1440)
                analysis = (
                    SceneAnalysis.from_dict(item.analysis)
                    if item.analysis
                    else analyze_scene(photo.rgb)
                )
                rgb, masks, warnings = render_rgb(
                    photo.rgb,
                    item.settings,
                    analysis,
                    local_settings=item.local_settings,
                    mask_edits=item.mask_edits,
                    reference=reference,
                    check_cancelled=token.check,
                )
                token.check()
                encode_photo(destination, rgb, photo.alpha, b"", "PNG")
                report = {
                    "kind": "image_preview",
                    "schema_version": 1,
                    "source": {"path": str(item.path), "sha256": sha256_file(item.path)},
                    "settings": item.settings.to_dict(),
                    "local_settings": item.local_settings,
                    "mask_edits": item.mask_edits,
                    "reference": reference.to_dict() if reference else None,
                    "analysis": analysis.to_dict(),
                    "output": {
                        "path": str(destination),
                        "width": photo.width,
                        "height": photo.height,
                    },
                    "warnings": list(warnings),
                }
                destination.with_suffix(".png.freedive.json").write_bytes(json_bytes(report))
            self._check_source(item, token)
            return {
                "media_id": key,
                "revision": record.revision,
                "path": str(destination),
                "recipe_path": str(destination.with_suffix(destination.suffix + ".freedive.json")),
                "recipe": report,
                "start_seconds": start,
                "duration_seconds": duration,
            }

        def complete(result):
            old = self.read_record(key)
            info = dict(old.item.media_info)
            if video and result["recipe"].get("plan"):
                info["shot_plan"] = result["recipe"]["plan"]
            self._records[key] = replace(
                old, preview=_clone(result), item=replace(old.item, media_info=info)
            )
            self._emit("preview_ready", [key], preview=_clone(result))

        return {
            "job_id": self._launch(
                "render_preview", params, fingerprint, operation, [key], complete
            )
        }

    def _export_media(self, params, fingerprint):
        records = self._check_revisions(params["media_ids"], params.get("expected_revisions"))
        directory = _local_path(params["directory"])
        if directory.exists() and not directory.is_dir():
            raise ControlError("OUTPUT_BUSY", "Выходной путь занят файлом.", "Выберите папку.")
        options = (
            ExportOptions.from_dict(params["options"])
            if params.get("options")
            else self._base.export_options
        )
        items = tuple(replace(record.item, included=True) for record in records)
        destinations = list(plan_batch(items, directory, options))
        if params.get("video_suffix"):
            destinations = [
                path.with_suffix(params["video_suffix"])
                if item.path.suffix.lower() in VIDEO_SUFFIXES
                else path
                for item, path in zip(items, destinations, strict=True)
            ]
        names = [str(path).casefold() for path in destinations]
        destinations = [
            path.with_name(
                path.stem
                + "_"
                + hashlib.sha256(str(item.path).encode()).hexdigest()[:10]
                + path.suffix
            )
            if names.count(str(path).casefold()) > 1
            else path
            for item, path in zip(items, destinations, strict=True)
        ]
        for path in destinations:
            if path.exists() or path.with_suffix(path.suffix + ".freedive.json").exists():
                raise ControlError(
                    "OUTPUT_BUSY",
                    "Результат или рецепт уже существует.",
                    "Выберите другую папку; существующие файлы не перезаписываются.",
                    {"path": str(path)},
                )
            if path in {record.item.path for record in self._records.values()}:
                raise ControlError("OUTPUT_BUSY", "Выходной путь совпадает с оригиналом.")
        reference_context, temp_directory = self._reference_context(), self._base.temp_directory

        def operation(token, progress):
            reference = self._resolve_reference(reference_context, token)
            rows = []
            for index, (record, item, destination) in enumerate(
                zip(records, items, destinations, strict=True)
            ):
                if token.event.is_set():
                    break
                try:
                    self._check_source(item, token)

                    def callback(value, message, i=index):
                        progress(round((i + value / 100) / len(items) * 100), message)

                    if item.path.suffix.lower() in VIDEO_SUFFIXES:
                        report = export_video(
                            item.path,
                            destination,
                            item.settings,
                            local_settings=item.local_settings,
                            mask_edits=item.mask_edits,
                            reference=reference,
                            token=token,
                            progress=callback,
                            temp_directory=temp_directory,
                            plan=item.media_info.get("shot_plan"),
                            keep_gps=options.keep_gps,
                        )
                    else:
                        report = export_photo(
                            item.path,
                            destination,
                            item.settings,
                            analysis=SceneAnalysis.from_dict(item.analysis)
                            if item.analysis
                            else None,
                            bit_depth=options.bit_depth,
                            quality=options.quality,
                            keep_gps=options.keep_gps,
                            local_settings=item.local_settings,
                            mask_edits=item.mask_edits,
                            reference=reference,
                            token=token,
                            progress=callback,
                        )
                    rows.append(
                        {
                            "media_id": record.media_id,
                            "status": "succeeded",
                            "path": str(destination),
                            "recipe_path": str(
                                destination.with_suffix(destination.suffix + ".freedive.json")
                            ),
                            "recipe": report,
                        }
                    )
                except Cancelled:
                    break
                except Exception as exc:
                    rows.append(
                        {
                            "media_id": record.media_id,
                            "status": "failed",
                            "error": _error(exc).to_dict(),
                        }
                    )
            return {
                "records": rows,
                "cancelled": token.event.is_set(),
                "completed": sum(row["status"] == "succeeded" for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
            }

        return {
            "job_id": self._launch(
                "export_media", params, fingerprint, operation, params["media_ids"]
            )
        }

    def _set_reference(self, params, fingerprint):
        self._check_revisions(list(self._records), params.get("expected_revisions"))
        ids = [params["media_id"]] if params["source"] == "frame" else []
        record = self.read_record(ids[0]) if ids else None
        path = _local_path(params["path"]) if not record else record.item.path
        if not path.is_file():
            raise ControlError("FILE_NOT_FOUND", "Файл эталона не найден.")
        reference_context = self._reference_context() if record else (None, None, None)
        temp_directory = self._base.temp_directory
        self._reference_generation += 1
        generation = self._reference_generation
        stamp = signature(path)

        def operation(token, progress):
            token.check()
            original_reference = self._resolve_reference(reference_context, token)
            if signature(path) != stamp:
                raise ControlError("SOURCE_CHANGED", "Эталон изменился после постановки задания.")
            progress(10, "Подготовка эталонной палитры…")
            if not record:
                reference = create_image_reference(path)
            else:
                item = record.item
                self._check_source(item, token)
                if item.path.suffix.lower() in VIDEO_SUFFIXES:
                    # Use the actual existing video pipeline, including its fixed per-shot
                    # reference transform and source analysis. Decode only the requested proxy.
                    if temp_directory:
                        temp_directory.mkdir(parents=True, exist_ok=True)
                    folder = Path(tempfile.mkdtemp(prefix="freedive-ref-", dir=temp_directory))
                    try:
                        out = folder / "frame.mp4"
                        export_video(
                            path,
                            out,
                            item.settings,
                            local_settings=item.local_settings,
                            mask_edits=item.mask_edits,
                            reference=original_reference,
                            token=token,
                            preview_start=params.get("time_seconds", 0),
                            preview_seconds=1,
                            proxy_edge=640,
                            plan=item.media_info.get("shot_plan"),
                            temp_directory=temp_directory,
                        )
                        rgb, _ = read_video_frame(out, 0, max_edge=640)
                        reference = create_reference(
                            rgb,
                            build_masks(rgb, MaskEdits.from_dict(item.mask_edits)),
                            source_name=path.name,
                            source_sha256=sha256_file(path),
                        )
                    finally:
                        shutil.rmtree(folder, ignore_errors=True)
                else:
                    photo = load_photo(path, max_edge=1440)
                    rgb, masks, _ = render_rgb(
                        photo.rgb,
                        item.settings,
                        SceneAnalysis.from_dict(item.analysis)
                        if item.analysis
                        else analyze_scene(photo.rgb),
                        local_settings=item.local_settings,
                        mask_edits=item.mask_edits,
                        reference=original_reference,
                        check_cancelled=token.check,
                    )
                    reference = create_reference(
                        rgb,
                        masks or build_masks(rgb),
                        source_name=path.name,
                        source_sha256=sha256_file(path),
                    )
            token.check()
            if signature(path) != stamp:
                raise ControlError("SOURCE_CHANGED", "Эталон изменился во время обработки.")
            return {
                "reference": reference.to_dict(),
                "reference_path": str(path) if not record else None,
            }

        def complete(result):
            if generation != self._reference_generation:
                result["applied"] = False
                result["warning"] = (
                    "Эталон не применён: поступила более новая команда изменения эталона."
                )
                return
            result["applied"] = True
            self.update_session_options(
                reference_path=result["reference_path"], reference_data=result["reference"]
            )

        return {
            "job_id": self._launch("set_reference", params, fingerprint, operation, ids, complete)
        }

    def _save_project(self, params, fingerprint):
        session, path = self.get_session(), _local_path(params["path"])
        overwrite = params.get("overwrite", False)
        if overwrite and path != self.project_path:
            raise ControlError(
                "OUTPUT_BUSY",
                "Перезапись разрешена только для текущего проекта.",
                "Сохраните новый проект под другим именем.",
            )

        def operation(token, progress):
            token.check()
            save_session(path, session, overwrite=overwrite)
            return {"path": str(path)}

        def complete(result):
            self.project_path = path
            self._emit("project_saved", path=str(path))

        return {
            "job_id": self._launch(
                "save_project", params, fingerprint, operation, list(self._records), complete
            )
        }

    def _load_project(self, params, fingerprint):
        if not self.is_idle():
            raise ControlError(
                "PROJECT_BUSY",
                "Загрузка проекта запрещена при активной очереди.",
                "Дождитесь завершения или отмените задания.",
            )
        path = _local_path(params["path"])

        def operation(token, progress):
            token.check()
            session = load_session(path)
            token.check()
            return session

        def complete(session):
            self._replace_session(session, path)

        return {
            "job_id": self._launch("load_project", params, fingerprint, operation, success=complete)
        }
