"""Isolated access to bundled FFmpeg libraries and optional local diagnostic executables."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from freedive_color.core.errors import StudioError

MP4_AUDIO_CODECS = frozenset(("aac", "mp3", "alac", "ac3", "eac3"))


def preferred_video_suffix(audio_codecs) -> str:
    """Keep original audio; PCM and other MOV-only tracks must not be discarded."""
    return ".mov" if any(codec not in MP4_AUDIO_CODECS for codec in audio_codecs) else ".mp4"


def require_av():
    try:
        import av

        return av
    except (ImportError, OSError) as exc:
        raise StudioError(
            "Локальный видеобэкенд недоступен. Переустановите полную сборку приложения."
        ) from exc


def open_local_media(source: str | Path):
    """Only the local MOV/MP4 demuxer may read media; nested URLs cannot open."""
    text = str(source)
    if text.startswith(("//", "\\\\")) or "://" in text:
        raise StudioError("Выберите локальный медиафайл.")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise StudioError("Локальный медиафайл не найден.")
    return require_av().open(
        str(path),
        mode="r",
        options={
            "protocol_whitelist": "file,pipe",
            "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    )


def find_media_tool(name: str) -> Path | None:
    if name not in ("ffmpeg", "ffprobe"):
        raise ValueError("Only FFmpeg media tools may be selected")
    executable = name + (".exe" if os.name == "nt" else "")
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    for directory in (bundle / "media", bundle / "assets" / "bin"):
        candidate = directory / executable
        if candidate.is_file():
            return candidate
    # A packaged app must not acquire an accidental dependency on tools in a user's PATH.
    if not getattr(sys, "frozen", False):
        found = shutil.which(name)
        return Path(found) if found else None
    return None


def ffprobe_json(source: str | Path) -> dict | None:
    executable = find_media_tool("ffprobe")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                str(executable),
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mov,mp4,m4a,3gp,3g2,mj2",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                "-i",
                str(Path(source).resolve()),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
        if len(result.stdout) > 16 * 1024 * 1024:
            raise StudioError("Метаданные видео слишком велики.")
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        return data
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise StudioError("Не удалось проверить метаданные видео локальным ffprobe.") from exc


def backend_versions() -> dict:
    av = require_av()
    return {
        "pyav": av.__version__,
        "ffmpeg_libraries": {
            name: ".".join(map(str, version)) for name, version in av.library_versions.items()
        },
    }
