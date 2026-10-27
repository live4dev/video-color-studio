"""Bounded-memory sequential series export with explicit outcomes for every input."""

import hashlib
import logging
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyze import SceneAnalysis
from .errors import Cancelled, StudioError
from .image_pipeline import CancellationToken, export_photo
from .project import sha256_file
from .session import MAX_ITEMS, ExportOptions, SessionItem, _local_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchRecord:
    source: Path
    destination: Path | None
    status: str
    error: str | None = None
    report: dict | None = None

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "destination": str(self.destination) if self.destination else None,
            "status": self.status,
            "error": self.error,
            "report": self.report,
        }


@dataclass(frozen=True)
class BatchResult:
    records: tuple[BatchRecord, ...]
    cancelled: bool = False

    @property
    def completed(self) -> int:
        return sum(record.status == "completed" for record in self.records)

    @property
    def failed(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    @property
    def excluded(self) -> int:
        return sum(record.status == "excluded" for record in self.records)

    def to_dict(self) -> dict:
        return {
            "completed": self.completed,
            "failed": self.failed,
            "excluded": self.excluded,
            "cancelled": self.cancelled,
            "records": [record.to_dict() for record in self.records],
        }


def _safe_stem(path: Path) -> str:
    stem = unicodedata.normalize("NFC", path.stem)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).rstrip(" .")
    return (stem or "photo")[:150]


def plan_batch(
    items: Sequence[SessionItem], output_directory: str | Path, options: ExportOptions | None = None
) -> tuple[Path | None, ...]:
    if options is not None and not isinstance(options, ExportOptions):
        raise StudioError("Некорректные параметры экспорта серии.")
    options = options or ExportOptions()
    if len(items) > MAX_ITEMS or any(not isinstance(item, SessionItem) for item in items):
        raise StudioError("Некорректный список файлов серии.")
    if len({item.path for item in items}) != len(items):
        raise StudioError("Один исходник добавлен в очередь дважды.")
    directory = _local_path(output_directory)
    suffixes = [
        item.path.suffix.lower() if item.path.suffix.lower() in (".mp4", ".mov") else options.suffix
        for item in items
    ]
    bases = [
        f"{_safe_stem(item.path)}_{item.settings.preset}{suffix}"
        for item, suffix in zip(items, suffixes, strict=True)
    ]
    # Count excluded entries too: toggling an exclusion never renames another input.
    counts = Counter(name.casefold() for name in bases)
    try:
        common = (
            Path(os.path.commonpath([str(item.path.parent) for item in items])) if items else None
        )
    except ValueError:
        common = None
    paths = []
    for item, base, suffix in zip(items, bases, suffixes, strict=True):
        if not item.included:
            paths.append(None)
            continue
        if counts[base.casefold()] > 1:
            identity = item.path.relative_to(common).as_posix() if common else str(item.path)
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
            base = f"{_safe_stem(item.path)}_{item.settings.preset}_{digest}{suffix}"
        paths.append(directory / base)
    if len({str(path).casefold() for path in paths if path}) != sum(
        path is not None for path in paths
    ):
        raise StudioError(
            "Не удалось подобрать разные имена результатов. Выберите другие имена файлов."
        )
    return tuple(paths)


def export_batch(
    items: Sequence[SessionItem],
    output_directory: str | Path,
    options: ExportOptions | None = None,
    *,
    token: CancellationToken | None = None,
    progress: Callable[[int, int, str], None] = lambda _done, _total, _message: None,
    exporter: Callable[..., dict] | None = None,
    reference_path: str | Path | None = None,
    reference: Any = None,
    temp_directory: str | Path | None = None,
) -> BatchResult:
    """Export one media file at a time. Completed outputs survive cancellation/errors."""
    items = tuple(items)
    if any(not isinstance(item, SessionItem) for item in items):
        raise StudioError("Некорректный список файлов серии.")
    options = options or ExportOptions()
    token = token or CancellationToken()
    supplied_exporter = exporter
    destinations = plan_batch(items, output_directory, options)
    original_paths = {item.path for item in items}
    total = sum(item.included for item in items)
    attempted = 0
    records = []
    was_cancelled = False

    def not_started_summary() -> BatchResult:
        return BatchResult(
            tuple(
                BatchRecord(item.path, destination, "not_started" if item.included else "excluded")
                for item, destination in zip(items, destinations, strict=True)
            ),
            cancelled=True,
        )

    try:
        token.check()
        if reference is None and reference_path is not None:
            from .image_pipeline import create_image_reference

            reference = create_image_reference(_local_path(reference_path))
            token.check()
    except Cancelled:
        return not_started_summary()
    for index, (item, destination) in enumerate(zip(items, destinations, strict=True)):
        if not item.included:
            records.append(BatchRecord(item.path, None, "excluded"))
            continue
        try:
            token.check()
            if destination.resolve() in original_paths:
                raise StudioError(
                    "Результат совпадает с исходником другого файла. Выберите другую папку."
                )
            if (
                destination.exists()
                or destination.with_suffix(destination.suffix + ".freedive.json").exists()
            ):
                raise StudioError(
                    "Результат или его JSON уже существует. Выберите другую папку экспорта."
                )
            if item.source_sha256 is not None and sha256_file(item.path) != item.source_sha256:
                raise StudioError(
                    "Исходник изменился после сохранения проекта. Импортируйте его заново."
                )
            kwargs = {
                "bit_depth": options.bit_depth,
                "quality": options.quality,
                "keep_gps": options.keep_gps,
                "token": token,
            }
            is_video = item.path.suffix.lower() in (".mp4", ".mov")
            if is_video and supplied_exporter is None:
                from .video_pipeline import export_video

                exporter = export_video
                kwargs = {
                    "token": token,
                    "temp_directory": temp_directory,
                    "keep_gps": options.keep_gps,
                }
                if item.media_info.get("shot_plan") is not None:
                    kwargs["plan"] = item.media_info["shot_plan"]
            else:
                exporter = supplied_exporter or export_photo
            if item.analysis is not None and not is_video:
                kwargs["analysis"] = SceneAnalysis.from_dict(item.analysis)
            if item.local_settings:
                kwargs["local_settings"] = item.local_settings
            if item.mask_edits:
                kwargs["mask_edits"] = item.mask_edits
            if reference is not None:
                kwargs["reference"] = reference

            def file_progress(
                value: int, message: str, *, source=item.path, done=attempted
            ) -> None:
                progress(done, total, f"{source.name}: {message} ({value}%)")

            kwargs["progress"] = file_progress
            progress(attempted, total, f"Обработка {item.path.name}")
            report = exporter(item.path, destination, item.settings, **kwargs)
            records.append(BatchRecord(item.path, destination, "completed", report=report))
        except Cancelled:
            was_cancelled = True
            records.append(BatchRecord(item.path, destination, "cancelled", "Обработка отменена."))
            for rest, target in zip(items[index + 1 :], destinations[index + 1 :], strict=True):
                records.append(
                    BatchRecord(rest.path, target, "not_started" if rest.included else "excluded")
                )
            break
        except Exception as exc:
            if not isinstance(exc, (StudioError, OSError)):
                logger.exception("Series input failed: %s", item.path.name)
            records.append(BatchRecord(item.path, destination, "failed", str(exc)))
        attempted += 1
        try:
            progress(attempted, total, f"Обработано {attempted} из {total}")
        except Cancelled:
            was_cancelled = True
            for rest, target in zip(items[index + 1 :], destinations[index + 1 :], strict=True):
                records.append(
                    BatchRecord(rest.path, target, "not_started" if rest.included else "excluded")
                )
            break
    return BatchResult(tuple(records), cancelled=was_cancelled)
