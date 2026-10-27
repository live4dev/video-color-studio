"""Shared photo workflow for CLI and GUI; stages all output before committing it."""

import errno
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from freedive_color import __version__

from .analyze import SceneAnalysis, analyze_scene
from .color_pipeline import grade_image
from .errors import Cancelled, StudioError
from .media_io import Photo, encode_photo, load_photo, signature
from .presets import GradeSettings
from .project import SCHEMA_VERSION, json_bytes, sha256_file

FORMAT_BY_SUFFIX = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF"}


class CancellationToken:
    def __init__(self) -> None:
        self.event = Event()

    def cancel(self) -> None:
        self.event.set()

    def check(self) -> None:
        if self.event.is_set():
            raise Cancelled()


@dataclass
class Preview:
    photo: Photo
    analysis: SceneAnalysis
    result: np.ndarray


def _publish(
    staging: Path, destination: Path, published: list[Path], token: CancellationToken
) -> None:
    try:
        os.link(staging, destination)
        published.append(destination)
    except OSError as exc:
        unsupported = {errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV}
        if exc.errno not in unsupported:
            raise
        # FAT/exFAT media do not support hard links. Exclusive creation still protects every
        # existing file; a failed/cancelled copy is tracked immediately and removed by the caller.
        with destination.open("xb") as output:
            published.append(destination)
            with staging.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    token.check()
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())


def make_preview(
    path: Path,
    settings: GradeSettings,
    max_edge: int = 1440,
    token: CancellationToken | None = None,
) -> Preview:
    token = token or CancellationToken()
    token.check()
    photo = load_photo(path, max_edge=max_edge)
    token.check()
    analysis = analyze_scene(photo.rgb)
    return Preview(photo, analysis, grade_image(photo.rgb, settings, analysis, token.check))


def export_photo(
    source: str | Path,
    destination: str | Path,
    settings: GradeSettings,
    *,
    analysis: SceneAnalysis | None = None,
    bit_depth: int = 8,
    quality: int = 95,
    keep_gps: bool = False,
    token: CancellationToken | None = None,
    progress: Callable[[int, str], None] = lambda _value, _message: None,
    expected_signature: tuple[int, int] | None = None,
) -> dict:
    started = time.perf_counter()
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    token = token or CancellationToken()
    sidecar = destination.with_suffix(destination.suffix + ".freedive.json")
    staging: list[Path] = []
    published: list[Path] = []
    try:
        token.check()
        if destination == source or (
            destination.exists() and os.path.samefile(source, destination)
        ):
            raise StudioError("Оригинал защищён. Выберите другое имя или папку для результата.")
        if destination.exists() or sidecar.exists():
            raise StudioError("Результат или его JSON уже существует. Выберите новое имя.")
        output_format = FORMAT_BY_SUFFIX.get(destination.suffix.lower())
        if output_format is None:
            raise StudioError("Для экспорта выберите расширение .jpg, .png или .tiff.")
        if expected_signature is not None and signature(source) != expected_signature:
            raise StudioError(
                "Оригинал изменился после импорта. Откройте его заново перед экспортом."
            )
        progress(5, "Чтение оригинала…")
        photo = load_photo(source, keep_gps=keep_gps)
        token.check()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Worst-case uncompressed image plus a staging copy and metadata margin.
        estimated = photo.width * photo.height * 4 * (bit_depth // 8) * 2 + 2_000_000
        if shutil.disk_usage(destination.parent).free < estimated:
            raise StudioError("Недостаточно свободного места. Выберите другую папку экспорта.")
        analysis = analysis or analyze_scene(photo.rgb)
        progress(25, "Цветокоррекция полного разрешения…")
        result = grade_image(photo.rgb, settings, analysis, token.check)
        token.check()
        progress(65, "Запись изображения…")
        fd, name = tempfile.mkstemp(prefix=".freedive-", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temp_image = Path(name)
        staging.append(temp_image)
        encode_photo(
            temp_image, result, photo.alpha, photo.exif_bytes, output_format, bit_depth, quality
        )
        token.check()
        digest = sha256_file(source)
        if signature(source) != photo.stat_signature:
            raise StudioError("Оригинал изменился во время экспорта. Повторите импорт.")
        warnings = photo.warnings + list(analysis.warnings)
        if bit_depth == 16 and output_format == "TIFF":
            warnings.append(
                "TIFF 16 бит: перенесены только базовые TIFF-теги; EXIF/GPS не перенесены."
            )
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "image_recipe",
            "application_version": __version__,
            "source": {"path": str(source), "sha256": digest, **photo.metadata()},
            "output": {
                "path": str(destination),
                "format": output_format,
                "width": photo.width,
                "height": photo.height,
                "bit_depth": bit_depth,
                "jpeg_quality": quality if output_format == "JPEG" else None,
                "color": "sRGB",
                "gps_requested": keep_gps,
                "sha256": sha256_file(temp_image),
            },
            "settings": settings.to_dict(),
            "analysis": analysis.to_dict(),
            "warnings": warnings,
            "metrics": {
                "seconds": round(time.perf_counter() - started, 3),
                "clipped_channel_fraction": float(np.mean((result <= 0) | (result >= 1))),
            },
        }
        fd, name = tempfile.mkstemp(prefix=".freedive-", suffix=".json.tmp", dir=destination.parent)
        os.close(fd)
        temp_json = Path(name)
        staging.append(temp_json)
        temp_json.write_bytes(json_bytes(report))
        token.check()
        progress(95, "Сохранение результата и настроек…")
        # Hard-link publication is atomic and fails if a file appeared since preflight.
        # Both staging files are on the destination volume. Never use os.replace on user paths.
        _publish(temp_image, destination, published, token)
        _publish(temp_json, sidecar, published, token)
        progress(100, "Экспорт завершён")
        return report
    except BaseException as exc:
        for path in published:
            path.unlink(missing_ok=True)
        if isinstance(exc, (StudioError, KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, OSError):
            raise StudioError(
                "Не удалось записать результат. Проверьте место, права и имя файла."
            ) from exc
        raise
    finally:
        for path in staging:
            path.unlink(missing_ok=True)
