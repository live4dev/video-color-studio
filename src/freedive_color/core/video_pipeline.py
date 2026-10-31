"""Streaming, timestamp-aware SDR video grading with unchanged encoded audio packets."""

import hashlib
import math
import os
import re
import shutil
import tempfile
import time
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from freedive_color import __version__
from freedive_color.integrations.ffmpeg import backend_versions, open_local_media, require_av

from .analyze import linear_to_srgb, srgb_to_linear
from .errors import StudioError
from .image_pipeline import CancellationToken, _publish, render_rgb
from .local_pipeline import LocalSettings
from .masks import MaskEdits, MaskSet
from .media_io import signature
from .presets import GradeSettings
from .project import json_bytes, sha256_file
from .reference import PaletteReference
from .temporal import ShotPlan, TemporalAnalyzer, TemporalMaskTracker, _resize_masks

VIDEO_EXTENSIONS = {".mp4", ".mov"}
MAX_VIDEO_PIXELS = 4096 * 2304
# libswscale's public SWS_ACCURATE_RND flag, combined with bilinear sampling.
_SWS_PRECISE = (1 << 18) | 2
_PRIMARIES_TO_709 = {
    5: np.linalg.inv(
        np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
    )
    @ np.array(
        [
            [0.4306190, 0.3415419, 0.1783091],
            [0.2220379, 0.7066384, 0.0713236],
            [0.0201853, 0.1295504, 0.9390944],
        ]
    ),
    6: np.linalg.inv(
        np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
    )
    @ np.array(
        [
            [0.3935891, 0.3652497, 0.1916313],
            [0.2124132, 0.7010437, 0.0865432],
            [0.0187423, 0.1119313, 0.9581563],
        ]
    ),
}


def _fraction(value, fallback=Fraction(0)) -> Fraction:
    return Fraction(value) if value is not None else fallback


def _rational(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator, "seconds": float(value)}


def _path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith(("//", "\\\\")) or "://" in text:
        raise StudioError("Для видео выберите локальный файл, а не сетевой адрес.")
    return Path(value).expanduser().resolve()


def _timestamp(frame, fallback_time_base: Fraction) -> Fraction:
    if frame.pts is None:
        raise StudioError("В видео отсутствуют временные метки кадра. Экспорт без них небезопасен.")
    return frame.pts * _fraction(frame.time_base, fallback_time_base)


def _tick(value: Fraction, time_base: Fraction) -> int:
    value /= time_base
    return (
        (value.numerator * 2 + value.denominator) // (value.denominator * 2)
        if value >= 0
        else -_tick(-value, Fraction(1))
    )


@dataclass(frozen=True)
class AudioInfo:
    index: int
    codec: str
    time_base: Fraction
    start_time: Fraction
    duration: float
    sample_rate: int
    channels: int
    layout: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "codec": self.codec,
            "time_base": str(self.time_base),
            "start_time": _rational(self.start_time),
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    duration: float
    fps: float
    rate: Fraction
    time_base: Fraction
    video_start: Fraction
    video_stream_index: int
    codec: str
    pixel_format: str
    frame_count: int | None
    sample_aspect_ratio: Fraction
    rotation: float
    color_space: int
    color_range: int
    color_trc: int
    color_primaries: int
    container_duration: float
    audio_streams: tuple[AudioInfo, ...]
    warnings: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    coded_width: int = 0
    coded_height: int = 0

    @property
    def bit_depth(self) -> int:
        return max(
            component.bits for component in require_av().VideoFormat(self.pixel_format).components
        )

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "coded_width": self.coded_width or self.width,
            "coded_height": self.coded_height or self.height,
            "duration": self.duration,
            "fps": self.fps,
            "rate": str(self.rate),
            "time_base": str(self.time_base),
            "video_start": _rational(self.video_start),
            "video_stream_index": self.video_stream_index,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "bit_depth": self.bit_depth,
            "frame_count": self.frame_count,
            "sample_aspect_ratio": str(self.sample_aspect_ratio),
            "rotation": self.rotation,
            "color_space": self.color_space,
            "color_range": self.color_range,
            "color_trc": self.color_trc,
            "color_primaries": self.color_primaries,
            "container_duration": self.container_duration,
            "audio_streams": [stream.to_dict() for stream in self.audio_streams],
            "warnings": list(self.warnings),
            "unsupported_reason": self.unsupported_reason,
        }


def probe_video(source: str | Path) -> VideoInfo:
    source = _path(source)
    if source.suffix.lower() not in VIDEO_EXTENSIONS or not source.is_file():
        raise StudioError("Откройте существующий локальный MP4 или MOV.")
    av = require_av()
    try:
        with open_local_media(source) as container:
            if len(container.streams.video) != 1:
                raise StudioError("Поддерживается один видеопоток. Выберите отдельный ролик.")
            video = container.streams.video[0]
            context = video.codec_context
            time_base = _fraction(video.time_base, Fraction(1, 90_000))
            rate = _fraction(video.average_rate or video.guessed_rate, Fraction(30))
            if rate <= 0:
                rate = Fraction(30)
            first = next(container.decode(video), None)
            if first is None:
                raise StudioError("В файле нет декодируемых видеокадров.")
            video_start = _timestamp(first, time_base)
            duration = float(video.duration * time_base) if video.duration else 0
            container_duration = (
                float(container.duration / av.time_base) if container.duration else duration
            )
            duration = duration or max(0, container_duration - float(video_start))
            sar = _fraction(video.sample_aspect_ratio, Fraction(1)) or Fraction(1)
            rotation = float(first.rotation or video.metadata.get("rotate", 0))
            colorspace = int(context.colorspace)
            color_range = int(context.color_range or first.color_range)
            color_trc = int(context.color_trc)
            primaries = int(context.color_primaries)
            metadata = " ".join(
                f"{key}={value}" for key, value in {**container.metadata, **video.metadata}.items()
            )
            side_names = {item.type.name for item in first.side_data}
            unsupported = None
            if context.name not in ("h264", "hevc"):
                unsupported = "Поддерживается декодирование H.264 и HEVC в MP4/MOV."
            if colorspace not in (0, 1, 2, 5, 6) or primaries not in (0, 1, 2, 5, 6):
                unsupported = (
                    "Это цветовое пространство видео не поддерживается. Экспортируйте SDR Rec.709."
                )
            if color_trc not in (0, 1, 2, 4, 5, 6, 7, 13):
                unsupported = "PQ/HLG/HDR или неизвестная transfer function: нужен SDR Rec.709."
            if re.search(
                r"\b(hdr|hlg|pq|logc?|s-log[23]?|v-log|dolby)\b", metadata, re.I
            ) or side_names.intersection(
                {
                    "MASTERING_DISPLAY_METADATA",
                    "CONTENT_LIGHT_LEVEL",
                    "DYNAMIC_HDR_PLUS",
                    "DOVI_RPU_BUFFER",
                    "DOVI_METADATA",
                }
            ):
                unsupported = (
                    "HDR/Log/Dolby Vision не поддерживается без явного преобразования в SDR."
                )
            if abs(rotation - 90 * round(rotation / 90)) > 0.01:
                unsupported = (
                    "Произвольный угол поворота не поддерживается без интерполяции. "
                    "Поддерживаются повороты на 90°, 180° и 270°."
                )
            for item in first.side_data:
                if item.type.name != "DISPLAYMATRIX":
                    continue
                matrix = np.frombuffer(item, dtype=np.int32)
                # Accept only unit, proper quarter-turn rotations, without a
                # reflection, perspective or translation hidden in the matrix.
                if len(matrix) != 9 or any(matrix[[2, 5, 6, 7]]) or matrix[8] != 1 << 30:
                    unsupported = "Неподдерживаемая геометрия display matrix видео."
                elif tuple(matrix[[0, 1, 3, 4]]) not in {
                    (65536, 0, 0, 65536),
                    (0, -65536, 65536, 0),
                    (-65536, 0, 0, -65536),
                    (0, 65536, -65536, 0),
                }:
                    unsupported = "Масштабирование или отражение display matrix не поддерживается."
            if sar != 1:
                unsupported = (
                    "Анаморфное видео пока не поддерживается. "
                    "Подготовьте копию с квадратными пикселями."
                )
            if context.width * context.height > MAX_VIDEO_PIXELS:
                unsupported = "Разрешение видео выше 4096×2304 пока не поддерживается."
            warnings = []
            if round(rotation / 90) % 4:
                warnings.append(
                    "Поворот камеры применён перестановкой пикселей, без обрезки и интерполяции."
                )
            if colorspace in (0, 2) and not first.format.name.startswith(("rgb", "bgr", "gbr")):
                warnings.append(
                    "Матрица цвета не указана: предполагается Rec.709 для HD, Rec.601 для SD."
                )
            if color_trc in (0, 2):
                warnings.append(
                    "Transfer function не указана: предполагается SDR Rec.709 (sRGB для RGB-видео)."
                )
            if color_range == 0:
                warnings.append(
                    "Диапазон видео не указан: предполагается limited для YUV, full для RGB."
                )
            audio = []
            for stream in container.streams.audio:
                audio_tb = _fraction(
                    stream.time_base, Fraction(1, stream.codec_context.sample_rate or 48_000)
                )
                audio.append(
                    AudioInfo(
                        stream.index,
                        stream.codec_context.name,
                        audio_tb,
                        (stream.start_time or 0) * audio_tb,
                        float(stream.duration * audio_tb) if stream.duration else 0,
                        stream.codec_context.sample_rate,
                        stream.codec_context.channels,
                        stream.codec_context.layout.name,
                    )
                )
            return VideoInfo(
                source,
                context.height if round(rotation / 90) % 2 else context.width,
                context.width if round(rotation / 90) % 2 else context.height,
                duration,
                float(rate),
                rate,
                time_base,
                video_start,
                video.index,
                context.name,
                first.format.name,
                video.frames or None,
                sar,
                rotation,
                colorspace,
                color_range,
                color_trc,
                primaries,
                container_duration,
                tuple(audio),
                tuple(warnings),
                unsupported,
                context.width,
                context.height,
            )
    except StudioError:
        raise
    except (av.FFmpegError, ValueError, OSError, StopIteration) as exc:
        raise StudioError(
            "Не удалось прочитать видео. Файл повреждён или декодер недоступен."
        ) from exc


def _validate_info(info: VideoInfo) -> None:
    if info.unsupported_reason:
        raise StudioError(info.unsupported_reason)
    if info.width % 2 or info.height % 2:
        raise StudioError(
            "Для H.264 4:2:0 нужны чётные размеры. Автоматический кроп не выполняется."
        )


def decode_video_frame(frame, info: VideoInfo | None = None) -> np.ndarray:
    """Convert coded SDR samples/matrix/range/transfer to the shared sRGB processing space."""
    av = require_av()
    if info is not None:
        _validate_info(info)
    colorspace = info.color_space if info else int(frame.colorspace)
    color_range = info.color_range if info else int(frame.color_range)
    transfer = info.color_trc if info else 1
    primaries = info.color_primaries if info else 1
    rgb_input = frame.format.name.startswith(("rgb", "bgr", "gbr"))
    matrix = (
        "ITU709" if colorspace == 1 or (colorspace in (0, 2) and frame.height >= 720) else "ITU601"
    )
    converted = av.video.reformatter.VideoReformatter().reformat(
        frame,
        # libswscale 9 corrupts packed RGB48 pixels with accurate rounding.
        # Planar 16-bit RGB keeps the precision; PyAV returns it as RGB HWC.
        format="gbrp16le",
        src_colorspace=matrix,
        dst_colorspace="ITU709",
        src_color_range="JPEG"
        if rgb_input or color_range == 2 or frame.format.name.startswith("yuvj")
        else "MPEG",
        dst_color_range="JPEG",
        interpolation=_SWS_PRECISE,
    )
    coded = converted.to_ndarray().astype(np.float32) / np.float32(65535)
    if transfer == 13 or (rgb_input and transfer in (0, 2)):
        linear = srgb_to_linear(coded)
    elif transfer in (4, 5):
        linear = coded ** (2.2 if transfer == 4 else 2.8)
    elif transfer == 7:
        linear = np.where(
            coded < 0.0912, coded / 4, ((coded + 0.1115) / 1.1115) ** (1 / 0.45)
        ).astype(np.float32)
    else:
        linear = np.where(
            coded < 0.081, coded / 4.5, ((coded + 0.099) / 1.099) ** (1 / 0.45)
        ).astype(np.float32)
    if primaries in _PRIMARIES_TO_709:
        linear = linear @ _PRIMARIES_TO_709[primaries].T.astype(np.float32)
    rgb = np.clip(linear_to_srgb(np.clip(linear, 0, 1)), 0, 1).astype(np.float32)
    rotation = info.rotation if info else float(frame.rotation)
    if round(rotation / 90) % 4:
        rgb = np.rot90(rgb, round(rotation / 90) % 4)
    return np.ascontiguousarray(rgb)


def encode_video_frame(rgb: np.ndarray, *, width: int | None = None, height: int | None = None):
    """Convert shared sRGB to actual Rec.709 limited-range YUV samples, then tag that space."""
    av = require_av()
    linear = srgb_to_linear(rgb)
    coded = np.where(linear < 0.018, linear * 4.5, 1.099 * linear**0.45 - 0.099)
    pixels = np.rint(np.clip(coded, 0, 1) * 65535).astype(np.uint16)
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
    frame.colorspace, frame.color_range = 0, 2
    frame = av.video.reformatter.VideoReformatter().reformat(
        frame,
        width=width,
        height=height,
        format="yuv420p",
        src_colorspace="ITU709",
        dst_colorspace="ITU709",
        src_color_range="JPEG",
        dst_color_range="MPEG",
        interpolation=_SWS_PRECISE,
    )
    frame.colorspace, frame.color_range = 1, 1
    return frame


def _read_source_video_frame(info, time_seconds, token):
    """Decode the selected source frame without changing its geometry or exact PTS."""
    target = info.video_start + Fraction(str(time_seconds)).limit_denominator(1_000_000)
    with open_local_media(info.path) as container:
        stream = container.streams[info.video_stream_index]
        container.seek(_tick(target, info.time_base), stream=stream, backward=True, any_frame=False)
        chosen = None
        for frame in container.decode(stream):
            token.check()
            chosen = frame
            if _timestamp(frame, info.time_base) >= target:
                break
        if chosen is None:
            raise StudioError("В этой позиции нет видеокадра.")
        token.check()
        return decode_video_frame(chosen, info), _timestamp(chosen, info.time_base)


def read_video_frame(
    source: str | Path, time_seconds: float = 0, max_edge: int = 1440
) -> tuple[np.ndarray, float]:
    """Read at/after a position relative to first video PTS, without loading the whole clip."""
    if not math.isfinite(time_seconds) or time_seconds < 0 or max_edge < 1:
        raise StudioError("Некорректная позиция или размер предпросмотра видео.")
    info = probe_video(source)
    _validate_info(info)
    av = require_av()
    try:
        rgb, timestamp = _read_source_video_frame(info, time_seconds, CancellationToken())
        if max(rgb.shape[:2]) > max_edge:
            scale = max_edge / max(rgb.shape[:2])
            rgb = cv2.resize(
                rgb,
                (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return rgb.astype(np.float32), float(timestamp - info.video_start)
    except StudioError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise StudioError("Не удалось прочитать кадр видео.") from exc


def analyze_video(
    source: str | Path,
    *,
    token: CancellationToken | None = None,
    progress: Callable[[int, str], None] = lambda _value, _message: None,
) -> ShotPlan:
    token = token or CancellationToken()
    token.check()
    info = probe_video(source)
    _validate_info(info)
    av = require_av()
    analyzer = TemporalAnalyzer()
    previous = None
    count = 0
    try:
        with open_local_media(info.path) as container:
            stream = container.streams[info.video_stream_index]
            for count, frame in enumerate(container.decode(stream), 1):
                token.check()
                timestamp = _timestamp(frame, info.time_base)
                if previous is not None and timestamp <= previous:
                    raise StudioError(
                        "Временные метки видео не возрастают. Автоматическая правка не выполняется."
                    )
                previous = timestamp
                scale = min(1, 640 / max(frame.width, frame.height))
                proxy = frame.reformat(
                    width=max(1, round(frame.width * scale)),
                    height=max(1, round(frame.height * scale)),
                )
                analyzer.add(decode_video_frame(proxy, info), float(timestamp), count - 1)
                if count % 12 == 0:
                    percent = min(
                        99,
                        round(float(timestamp - info.video_start) / max(info.duration, 0.1) * 100),
                    )
                    progress(percent, f"Анализ видео: {count} кадров")
        if not count:
            raise StudioError("В ролике нет видеокадров.")
        token.check()
        return analyzer.finish(info.duration)
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise StudioError("Ошибка потокового анализа видео.") from exc


class _AudioAudit:
    def __init__(self):
        self.payload = hashlib.sha256()
        self.timing = hashlib.sha256()
        self.count = 0
        self.first = None
        self.end = None
        self.max_duration = Fraction(0)

    def add(self, packet):
        if not packet.size:
            return
        if packet.pts is None or packet.dts is None or not packet.time_base:
            raise StudioError("У аудиопакета нет временной метки. Звук не будет молча удалён.")
        tb = Fraction(packet.time_base)
        pts, dts, duration = packet.pts * tb, packet.dts * tb, (packet.duration or 0) * tb
        data = bytes(packet)
        self.payload.update(len(data).to_bytes(8, "big"))
        self.payload.update(data)
        # Demuxers may infer the final packet duration from codec cadence rather
        # than MP4's sample table. PTS/DTS and the authoritative track end are
        # checked separately; padding must not lengthen an original audio track.
        self.timing.update(f"{pts}|{dts}\n".encode())
        self.count += 1
        self.first = pts if self.first is None else min(self.first, pts)
        self.end = pts + duration if self.end is None else max(self.end, pts + duration)
        self.max_duration = max(self.max_duration, duration)

    def to_dict(self):
        return {
            "packet_count": self.count,
            "packet_sha256": self.payload.hexdigest(),
            "timing_sha256": self.timing.hexdigest(),
            "first_pts": _rational(self.first) if self.first is not None else None,
            "end_pts": _rational(self.end) if self.end is not None else None,
            "max_packet_duration": float(self.max_duration),
        }


def _metadata(values, keep_gps):
    common = {
        "creation_time",
        "title",
        "artist",
        "copyright",
        "language",
        "handler_name",
        "timecode",
    }
    return {
        key: value
        for key, value in values.items()
        if key.lower() in common
        or (keep_gps and re.search(r"location|gps|latitude|longitude|xyz", key, re.I))
    }


def _frame_iterator(container, stream, info, start, end):
    if start > info.video_start:
        container.seek(_tick(start, info.time_base), stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        stamp = _timestamp(frame, info.time_base)
        if stamp < start:
            continue
        if end is not None and stamp >= end:
            break
        yield frame


def _warm_temporal_masks(source, info, plan, tracker, end, token, progress):
    """Replay only the current shot's prefix so seeking has full-export mask history.

    Frames before ``end`` update proxy alphas only: they are never encoded or muxed.
    Decode at source resolution exactly as the export loop does; preview resizing must
    not alter the semantic model's input or the optical-flow trajectory.
    """
    shot_id = plan.shot_at(float(end))
    start = (
        _tick(Fraction(str(plan.cuts[shot_id - 1])), info.time_base) * info.time_base
        if shot_id
        else info.video_start
    )
    if end <= start:
        return 0, start
    progress(20, "Подготовка видеомасок к выбранному кадру…")
    count = 0
    token.check()
    with open_local_media(source) as container:
        stream = container.streams[info.video_stream_index]
        for frame in _frame_iterator(container, stream, info, start, end):
            token.check()
            timestamp = _timestamp(frame, info.time_base)
            tracker.update(
                decode_video_frame(frame, info),
                float(timestamp),
                shot_id=plan.shot_at(float(timestamp)),
            )
            count += 1
            if count % 6 == 0:
                progress(20, f"Подготовка видеомасок: {count} кадров")
    return count, start


@dataclass(frozen=True)
class VideoFramePreview:
    rgb: np.ndarray
    timestamp: float
    masks: MaskSet
    plan: ShotPlan


def read_video_preview_frame(
    source: str | Path,
    time_seconds: float = 0,
    max_edge: int = 1440,
    *,
    edits: MaskEdits | None = None,
    feather: float = 0.006,
    plan: ShotPlan | dict | None = None,
    token: CancellationToken | None = None,
    progress: Callable[[int, str], None] = lambda _value, _message: None,
) -> VideoFramePreview:
    """A seekable GUI frame with exactly the mask history used by a full export.

    Semantic inference and flow see source pixels. Only the final frame and alpha maps
    are resized for the viewer. The returned masks precede waterline/color rendering so
    repeated color adjustments can reuse them without applying soft alpha twice.
    """
    if not math.isfinite(time_seconds) or time_seconds < 0 or max_edge < 1:
        raise StudioError("Некорректная позиция или размер предпросмотра видео.")
    token = token or CancellationToken()
    token.check()
    info = probe_video(source)
    _validate_info(info)
    av = require_av()
    try:
        if isinstance(plan, dict):
            plan = ShotPlan.from_dict(plan)
        if plan is None:
            plan = analyze_video(
                source,
                token=token,
                progress=lambda value, message: progress(round(value * 0.2), message),
            )
        if not isinstance(plan, ShotPlan) or abs(plan.start_time - float(info.video_start)) > float(
            info.time_base
        ):
            raise StudioError("План анализа не соответствует временной шкале исходника.")
        rgb, timestamp = _read_source_video_frame(info, time_seconds, token)
        tracker = TemporalMaskTracker(edits or MaskEdits(), feather=feather)
        _warm_temporal_masks(source, info, plan, tracker, timestamp, token, progress)
        token.check()
        masks = tracker.update(rgb, float(timestamp), shot_id=plan.shot_at(float(timestamp)))
        if max(rgb.shape[:2]) > max_edge:
            scale = max_edge / max(rgb.shape[:2])
            size = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            masks = _resize_masks(masks, size)
        token.check()
        return VideoFramePreview(rgb, float(timestamp - info.video_start), masks, plan)
    except StudioError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise StudioError("Не удалось подготовить кадр и маски видео.") from exc


def _audio_iterator(container, streams, start, end):
    if not streams:
        return
    if start is not None and start > 0:
        container.seek(
            _tick(start, Fraction(1, require_av().time_base)), backward=True, any_frame=False
        )
    track_ends = {
        stream.index: ((stream.start_time or 0) + stream.duration) * Fraction(stream.time_base)
        for stream in streams
        if stream.duration is not None
    }
    for packet in container.demux(*streams):
        if not packet.size:
            continue
        if packet.pts is None or packet.dts is None:
            raise StudioError("Аудиопакет не содержит PTS/DTS. Синхронизация не подтверждена.")
        stamp = packet.pts * Fraction(packet.time_base)
        duration = (packet.duration or 0) * Fraction(packet.time_base)
        track_end = track_ends.get(packet.stream.index)
        if track_end is not None and stamp + duration > track_end:
            duration = max(Fraction(0), track_end - stamp)
            packet.duration = _tick(duration, Fraction(packet.time_base))
        if start is not None and stamp + duration <= start:
            continue
        if end is not None and stamp >= end:
            continue
        yield packet


def _frozen_references(source, info, plan, settings, local, edits, reference, token):
    if reference is None or (local is not None and not local.enabled):
        return {}
    from .masks import build_masks
    from .reference import freeze_reference

    result = {}
    for shot_id, stamp in enumerate((plan.start_time, *plan.cuts)):
        token.check()
        rgb, _relative = read_video_frame(source, max(0, stamp - float(info.video_start)), 640)
        masks = build_masks(rgb, edits or MaskEdits(), feather=(local or LocalSettings()).feather)
        processed, _masks, _warnings = render_rgb(
            rgb,
            settings,
            plan.analysis_at(stamp),
            local_settings=local or LocalSettings(),
            mask_edits=edits or MaskEdits(),
            masks=masks,
            reference=None,
            check_cancelled=token.check,
        )
        result[shot_id] = freeze_reference(processed, _masks or masks, reference)
    return result


def _verify_output(
    path,
    expected_pts,
    expected_durations,
    encode_tb,
    source_info,
    expected_audio,
    out_audio_indices,
    width,
    height,
    token,
):
    actual_info = probe_video(path)
    if (actual_info.width, actual_info.height) != (width, height):
        raise StudioError("Проверка экспорта: размеры видео не совпали.")
    if actual_info.sample_aspect_ratio != source_info.sample_aspect_ratio or actual_info.rotation:
        raise StudioError("Проверка экспорта: изменились пропорции или поворот дисплея.")
    if (
        actual_info.color_space,
        actual_info.color_primaries,
        actual_info.color_trc,
        actual_info.color_range,
    ) != (1, 1, 1, 1):
        raise StudioError("Проверка экспорта: не подтверждено SDR Rec.709 limited range.")
    timing_hash, expected_hash = hashlib.sha256(), hashlib.sha256()
    max_error, actual_end, count = Fraction(0), Fraction(0), 0
    with open_local_media(path) as container:
        for count, frame in enumerate(container.decode(video=0), 1):
            token.check()
            if count > len(expected_pts):
                raise StudioError("Проверка экспорта: появились дополнительные кадры.")
            actual, expected = (
                _timestamp(frame, actual_info.time_base),
                expected_pts[count - 1] * encode_tb,
            )
            max_error = max(max_error, abs(actual - expected))
            timing_hash.update(f"{actual}\n".encode())
            expected_hash.update(f"{expected}\n".encode())
            actual_end = actual + (frame.duration or 0) * Fraction(frame.time_base)
    if count != len(expected_pts) or max_error > max(encode_tb, actual_info.time_base):
        raise StudioError("Проверка экспорта: число кадров или временные метки не совпали.")
    # MOV/MP4 track duration retains VFR's final hold even when a decoder reports
    # the codec's nominal frame duration for its last frame.
    actual_end = (
        actual_info.video_start
        + _tick(Fraction(str(actual_info.duration)), actual_info.time_base) * actual_info.time_base
    )
    expected_end = (expected_pts[-1] + expected_durations[-1]) * encode_tb
    if abs(actual_end - expected_end) > max(encode_tb, actual_info.time_base):
        raise StudioError("Проверка экспорта: длительность видеоряда не совпала.")
    actual_audio = {index: _AudioAudit() for index in out_audio_indices.values()}
    with open_local_media(path) as container:
        for packet in container.demux(*container.streams.audio):
            token.check()
            if packet.stream.type == "audio" and packet.size:
                if packet.stream.index not in actual_audio:
                    raise StudioError("Проверка экспорта: неожиданная звуковая дорожка.")
                actual_audio[packet.stream.index].add(packet)
    tracks = []
    actual_track_info = {track.index: track for track in actual_info.audio_streams}
    for index, expected in expected_audio.items():
        track = actual_track_info[out_audio_indices[index]]
        if expected.count:
            actual_audio[out_audio_indices[index]].end = (
                track.start_time
                + _tick(Fraction(str(track.duration)), track.time_base) * track.time_base
            )
        wanted, actual = expected.to_dict(), actual_audio[out_audio_indices[index]].to_dict()
        if any(
            wanted[key] != actual[key] for key in ("packet_count", "packet_sha256", "timing_sha256")
        ):
            raise StudioError("Проверка экспорта: звук или его временные метки изменились.")
        if (
            expected.end is not None
            and abs(actual_audio[out_audio_indices[index]].end - expected.end) > track.time_base
        ):
            raise StudioError("Проверка экспорта: изменилась длительность звуковой дорожки.")
        tracks.append(
            {
                "source_stream_index": index,
                "output_stream_index": out_audio_indices[index],
                "source_selected": wanted,
                "output": actual,
                "packets_identical": True,
                "timestamps_identical": True,
            }
        )
    return {
        "frame_count": count,
        "pts_exact": timing_hash.hexdigest() == expected_hash.hexdigest(),
        "pts_verified": True,
        "pts_max_error_seconds": float(max_error),
        "source_pts_sha256": expected_hash.hexdigest(),
        "output_pts_sha256": timing_hash.hexdigest(),
        "video_end_seconds": float(actual_end),
        "expected_video_end_seconds": float(expected_end),
        "audio_tracks": tracks,
        "audio_verified": True,
        "actual_output_info": actual_info.to_dict(),
    }


def export_video(
    source: str | Path,
    destination: str | Path,
    settings: GradeSettings,
    *,
    local_settings: LocalSettings | dict | None = None,
    mask_edits: MaskEdits | dict | None = None,
    reference: PaletteReference | dict | None = None,
    token: CancellationToken | None = None,
    progress: Callable[[int, str], None] = lambda _value, _message: None,
    preview_start: float = 0,
    preview_seconds: float | None = None,
    temp_directory: str | Path | None = None,
    proxy_edge: int | None = None,
    plan: ShotPlan | dict | None = None,
    keep_gps: bool = False,
) -> dict:
    """Stream frames and original audio packets; verify timestamps before publishing outputs."""
    started, av = time.perf_counter(), require_av()
    source, destination = _path(source), _path(destination)
    token = token or CancellationToken()
    staging, published = [], []
    sidecar = destination.with_suffix(destination.suffix + ".freedive.json")
    try:
        token.check()
        if source == destination or (
            destination.exists() and os.path.samefile(source, destination)
        ):
            raise StudioError("Оригинал защищён. Выберите другое имя результата видео.")
        if destination.exists() or sidecar.exists():
            raise StudioError("Видео или его JSON уже существует. Выберите новое имя.")
        if destination.suffix.lower() not in (".mp4", ".mov"):
            raise StudioError("Для экспорта видео выберите .mp4 или .mov.")
        if not math.isfinite(preview_start) or preview_start < 0:
            raise StudioError("Начало предпросмотра должно быть неотрицательным числом секунд.")
        if preview_seconds is not None and (
            not math.isfinite(preview_seconds) or preview_seconds <= 0
        ):
            raise StudioError("Длительность предпросмотра должна быть положительной.")
        if proxy_edge is not None and (type(proxy_edge) is not int or not 64 <= proxy_edge <= 8192):
            raise StudioError("Размер proxy должен быть целым числом от 64 до 8192.")
        if type(keep_gps) is not bool:
            raise StudioError("Сохранение GPS должно быть логическим значением.")
        local = (
            LocalSettings.from_dict(local_settings)
            if isinstance(local_settings, dict)
            else local_settings
        )
        edits = MaskEdits.from_dict(mask_edits) if isinstance(mask_edits, dict) else mask_edits
        reference = (
            PaletteReference.from_dict(reference) if isinstance(reference, dict) else reference
        )
        if local is not None and not isinstance(local, LocalSettings):
            raise StudioError("Некорректные параметры локальной обработки видео.")
        if edits is not None and not isinstance(edits, MaskEdits):
            raise StudioError("Некорректные маски видео.")
        if reference is not None and not isinstance(reference, PaletteReference):
            raise StudioError("Некорректная палитра эталона видео.")
        info = probe_video(source)
        _validate_info(info)
        if preview_start >= info.duration and info.duration:
            raise StudioError("Начало предпросмотра находится за концом видеоряда.")
        from freedive_color.integrations.ffmpeg import MP4_AUDIO_CODECS

        if destination.suffix.lower() == ".mp4" and any(
            track.codec not in MP4_AUDIO_CODECS for track in info.audio_streams
        ):
            raise StudioError(
                "Звуковой кодек несовместим с MP4. Выберите MOV для сохранения оригинального звука."
            )
        if "libx264" not in av.codecs_available:
            raise StudioError("В локальном бэкенде нет H.264 encoder libx264. Нужна полная сборка.")
        initial_signature, source_hash = signature(source), sha256_file(source)
        token.check()
        destination.parent.mkdir(parents=True, exist_ok=True)
        working = _path(temp_directory) if temp_directory is not None else destination.parent
        working.mkdir(parents=True, exist_ok=True)
        selected_duration = min(
            preview_seconds or info.duration, max(0.1, info.duration - preview_start)
        )
        estimate = max(
            16_000_000,
            round(source.stat().st_size * selected_duration / max(info.duration, 0.1) * 2),
            round(info.width * info.height * info.fps * selected_duration * 0.025),
        )
        if (
            min(shutil.disk_usage(working).free, shutil.disk_usage(destination.parent).free)
            < estimate
        ):
            raise StudioError("Недостаточно места для временного видео и результата.")
        if isinstance(plan, dict):
            plan = ShotPlan.from_dict(plan)
        if plan is None:
            plan = analyze_video(
                source,
                token=token,
                progress=lambda value, message: progress(round(value * 0.2), message),
            )
        if not isinstance(plan, ShotPlan) or abs(plan.start_time - float(info.video_start)) > float(
            info.time_base
        ):
            raise StudioError("План анализа не соответствует временной шкале исходника.")
        token.check()
        frozen = _frozen_references(source, info, plan, settings, local, edits, reference, token)
        local_active = (local is not None or edits is not None or reference is not None) and (
            local or LocalSettings()
        ).enabled
        tracker = (
            TemporalMaskTracker(edits or MaskEdits(), feather=(local or LocalSettings()).feather)
            if local_active
            else None
        )
        start = info.video_start + Fraction(str(preview_start)).limit_denominator(1_000_000)
        is_preview = preview_seconds is not None or preview_start > 0
        end = (
            start + Fraction(str(preview_seconds)).limit_denominator(1_000_000)
            if preview_seconds is not None
            else None
        )
        scale = min(1, proxy_edge / max(info.width, info.height)) if proxy_edge else 1
        width, height = (
            max(2, round(info.width * scale) // 2 * 2),
            max(2, round(info.height * scale) // 2 * 2),
        )
        fd, name = tempfile.mkstemp(
            prefix=".freedive-video-", suffix=destination.suffix, dir=working
        )
        os.close(fd)
        temp_video = Path(name)
        staging.append(temp_video)
        expected_pts, expected_durations = array("q"), array("q")
        expected_audio = {track.index: _AudioAudit() for track in info.audio_streams}
        warnings = list(info.warnings)
        with open_local_media(source) as video_input, open_local_media(source) as audio_input:
            in_video = video_input.streams[info.video_stream_index]
            frames = _frame_iterator(video_input, in_video, info, start, end)
            frame = next(frames, None)
            if frame is None:
                raise StudioError("В выбранном диапазоне нет видеокадров.")
            actual_start = _timestamp(frame, info.time_base)
            shift = actual_start if is_preview else Fraction(0)
            warmup_frames, mask_history_start = 0, actual_start
            if tracker and is_preview and actual_start > info.video_start:
                warmup_frames, mask_history_start = _warm_temporal_masks(
                    source, info, plan, tracker, actual_start, token, progress
                )
            audio_packets = _audio_iterator(
                audio_input,
                list(audio_input.streams.audio),
                actual_start if is_preview else None,
                end,
            )
            audio_packet = next(audio_packets, None)
            with av.open(
                str(temp_video),
                "w",
                format="mp4" if destination.suffix.lower() == ".mp4" else "mov",
                options={
                    "movflags": "+faststart",
                    "avoid_negative_ts": "disabled",
                    "max_interleave_delta": "1000000",
                    "movie_timescale": str(
                        math.lcm(
                            info.time_base.denominator,
                            *(track.time_base.denominator for track in info.audio_streams),
                        )
                    ),
                },
            ) as output:
                output.metadata.update(_metadata(video_input.metadata, keep_gps))
                out_video = output.add_stream("libx264", rate=info.rate)
                out_video.width, out_video.height, out_video.pix_fmt = width, height, "yuv420p"
                out_video.time_base = out_video.codec_context.time_base = info.time_base
                out_video.codec_context.sample_aspect_ratio = info.sample_aspect_ratio
                out_video.codec_context.color_primaries = out_video.codec_context.color_trc = (
                    out_video.codec_context.colorspace
                ) = 1
                out_video.codec_context.color_range = 1
                out_video.options = {
                    "crf": "18",
                    "preset": "medium",
                    "tune": "zerolatency",
                    "bf": "0",
                }
                out_video.metadata.update(_metadata(in_video.metadata, keep_gps))
                audio_outputs = {}
                for in_audio in audio_input.streams.audio:
                    out_audio = output.add_stream_from_template(in_audio)
                    out_audio.time_base = in_audio.time_base
                    out_audio.metadata.update(_metadata(in_audio.metadata, keep_gps))
                    audio_outputs[in_audio.index] = out_audio
                out_audio_indices = {key: stream.index for key, stream in audio_outputs.items()}
                output.start_encoding()
                encode_tb, durations = Fraction(out_video.codec_context.time_base), {}

                def mux_encoded(packet):
                    if packet.pts in durations:
                        packet.duration = durations.pop(packet.pts)
                    output.mux(packet)

                while frame is not None or audio_packet is not None:
                    token.check()
                    video_time = _timestamp(frame, info.time_base) if frame is not None else None
                    audio_time = (
                        audio_packet.dts * Fraction(audio_packet.time_base)
                        if audio_packet is not None
                        else None
                    )
                    if frame is not None and (audio_packet is None or video_time <= audio_time):
                        following_frame = next(frames, None)
                        rgb = decode_video_frame(frame, info)
                        shot_id = plan.shot_at(float(video_time))
                        masks = (
                            tracker.update(rgb, float(video_time), shot_id=shot_id)
                            if tracker
                            else None
                        )
                        if scale < 1:
                            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                            if masks is not None:
                                masks = _resize_masks(masks, (width, height))
                        if masks:
                            warnings.extend(
                                message for message in masks.warnings if message not in warnings
                            )
                        rendered, _masks, _warnings = render_rgb(
                            rgb,
                            settings,
                            plan.analysis_at(float(video_time)),
                            local_settings=(local or LocalSettings()) if local_active else None,
                            mask_edits=edits if local_active else None,
                            masks=masks,
                            reference=frozen.get(shot_id),
                            check_cancelled=token.check,
                        )
                        warnings.extend(message for message in _warnings if message not in warnings)
                        encoded = encode_video_frame(rendered)
                        encoded.pts, encoded.time_base = (
                            _tick(video_time - shift, encode_tb),
                            encode_tb,
                        )
                        next_time = (
                            _timestamp(following_frame, info.time_base)
                            if following_frame is not None
                            else info.video_start
                            + _tick(Fraction(str(info.duration)), info.time_base) * info.time_base
                        )
                        duration = next_time - video_time
                        if duration <= 0:
                            raise StudioError("Временные метки кадров не возрастают.")
                        if end is not None:
                            duration = min(duration, end - video_time)
                        encoded.duration = max(1, _tick(duration, encode_tb))
                        durations[encoded.pts] = encoded.duration
                        expected_pts.append(encoded.pts)
                        expected_durations.append(encoded.duration)
                        for packet in out_video.encode(encoded):
                            mux_encoded(packet)
                        if len(expected_pts) % 6 == 0:
                            value = 20 + min(
                                65,
                                round(
                                    float(video_time - actual_start)
                                    / max(selected_duration, 0.1)
                                    * 65
                                ),
                            )
                            progress(value, f"Обработка видео: {len(expected_pts)} кадров")
                        frame = following_frame
                    else:
                        index, packet_tb = (
                            audio_packet.stream.index,
                            Fraction(audio_packet.time_base),
                        )
                        if shift:
                            offset = _tick(shift, packet_tb)
                            audio_packet.pts -= offset
                            audio_packet.dts -= offset
                        expected_audio[index].add(audio_packet)
                        audio_packet.stream = audio_outputs[index]
                        output.mux(audio_packet)
                        audio_packet = next(audio_packets, None)
                for packet in out_video.encode(None):
                    mux_encoded(packet)
        token.check()
        progress(88, "Проверка кадров, PTS и звука…")
        verified = _verify_output(
            temp_video,
            expected_pts,
            expected_durations,
            encode_tb,
            info,
            expected_audio,
            out_audio_indices,
            width,
            height,
            token,
        )
        if signature(source) != initial_signature or sha256_file(source) != source_hash:
            raise StudioError("Исходник изменился во время обработки. Повторите импорт.")
        verified["actual_output_info"]["path"] = str(destination)
        report = {
            "schema_version": 1,
            "kind": "video_recipe",
            "application_version": __version__,
            "source": {**info.to_dict(), "sha256": source_hash},
            "output": {
                "path": str(destination),
                "width": width,
                "height": height,
                "codec": "H.264/libx264",
                "pixel_format": "yuv420p",
                "color": "SDR Rec.709 limited",
                "sha256": sha256_file(temp_video),
                "proxy_edge": proxy_edge,
                "frame_count": len(expected_pts),
                "gps_requested": keep_gps,
                "source_rotation_applied": info.rotation,
            },
            "settings": settings.to_dict(),
            "local_settings": local.to_dict() if local else None,
            "mask_edits": edits.to_dict() if edits else None,
            "reference": reference.to_dict() if reference else None,
            "reference_transforms": {str(key): value.to_dict() for key, value in frozen.items()},
            "plan": plan.to_dict(),
            "verification": verified,
            "warnings": list(dict.fromkeys(warnings)),
            "preview": {
                "enabled": is_preview,
                "requested_start": preview_start,
                "requested_seconds": preview_seconds,
                "actual_source_start": _rational(actual_start),
                "common_pts_shift": _rational(shift),
                "audio_cut": "Original packets, common shift, no audio re-encoding",
                "mask_warmup_frames": warmup_frames,
                "mask_history_source_start": _rational(mask_history_start),
            },
            "temporal_masks": tracker.to_metadata() if tracker else None,
            "backend": backend_versions(),
            "metrics": {"seconds": round(time.perf_counter() - started, 3)},
        }
        fd, name = tempfile.mkstemp(prefix=".freedive-video-", suffix=".json.tmp", dir=working)
        os.close(fd)
        temp_json = Path(name)
        staging.append(temp_json)
        temp_json.write_bytes(json_bytes(report))
        token.check()
        progress(97, "Сохранение видео и параметров…")
        _publish(temp_video, destination, published, token)
        _publish(temp_json, sidecar, published, token)
        progress(100, "Видео сохранено")
        return report
    except BaseException as exc:
        for path in published:
            path.unlink(missing_ok=True)
        if isinstance(exc, (StudioError, KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, (av.FFmpegError, OSError, ValueError)):
            raise StudioError(
                "Не удалось обработать видео локальным кодеком: " + str(exc)[:240]
            ) from exc
        raise
    finally:
        for path in staging:
            path.unlink(missing_ok=True)
