"""Licensed photo-based SDR videos, deterministic temporal changes and original synthetic sound."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from freedive_color.core.media_io import load_photo
from freedive_color.core.project import sha256_file
from freedive_color.core.video_pipeline import encode_video_frame, probe_video

ROOT = Path(__file__).resolve().parents[1]


def _photo_canvas(width: int, height: int) -> np.ndarray:
    photo = load_photo(ROOT / "tests/fixtures/freediver_moalboal.jpg").rgb
    scale = min(width / photo.shape[1], height / photo.shape[0])
    size = (round(photo.shape[1] * scale), round(photo.shape[0] * scale))
    resized = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
    # Neutral letterbox bars are explicit fixture framing, not reconstructed underwater content.
    canvas = np.full((height, width, 3), 0.04, np.float32)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas


def _one(path: Path, base: np.ndarray, duration: float, rate: int, *, depth=False, vfr=False):
    if path.exists():
        raise FileExistsError(f"Fixture already exists: {path}")
    video_tb = Fraction(1, 12_000)
    video_start = 3000 if vfr else 0
    audio_start = 24_000 if vfr else 0
    total_audio_samples = round(duration * 48_000)
    video_pts = []
    timestamp = video_start
    index = 0
    while timestamp < video_start + round(duration / video_tb):
        video_pts.append(timestamp)
        timestamp += (500, 500, 1000)[index % 3] if vfr else round(12_000 / rate)
        index += 1
    with av.open(str(path), "w", format="mp4", options={"avoid_negative_ts": "disabled"}) as output:
        video = output.add_stream("libx264", rate=rate)
        video.width, video.height = base.shape[1], base.shape[0]
        video.pix_fmt = "yuv420p"
        video.time_base = video.codec_context.time_base = video_tb
        video.codec_context.color_primaries = video.codec_context.color_trc = (
            video.codec_context.colorspace
        ) = 1
        video.codec_context.color_range = 1
        video.options = {"crf": "18", "preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
        audio = output.add_stream("aac", rate=48_000)
        audio.layout = "stereo"
        audio.bit_rate = 128_000
        audio.time_base = audio.codec_context.time_base = Fraction(1, 48_000)
        position_video, position_audio = 0, 0
        while position_video < len(video_pts) or position_audio < total_audio_samples:
            video_time = (
                video_pts[position_video] * video_tb
                if position_video < len(video_pts)
                else float("inf")
            )
            audio_time = (
                Fraction(audio_start + position_audio, 48_000)
                if position_audio < total_audio_samples
                else float("inf")
            )
            if video_time <= audio_time:
                t = float(video_time - video_start * video_tb) / max(duration, 0.01)
                rgb = base
                if depth:
                    factor = np.array([1 - 0.55 * t, 1 - 0.12 * t, 1 - 0.03 * t], np.float32)
                    if t >= 0.55:
                        factor *= [0.45, 0.72, 0.9]
                    rgb = np.clip(base * factor, 0, 1)
                frame = encode_video_frame(rgb)
                frame.pts, frame.time_base = video_pts[position_video], video_tb
                next_pts = (
                    video_pts[position_video + 1]
                    if position_video + 1 < len(video_pts)
                    else video_start + round(duration / video_tb)
                )
                frame.duration = next_pts - frame.pts
                for packet in video.encode(frame):
                    output.mux(packet)
                position_video += 1
            else:
                samples = min(1024, total_audio_samples - position_audio)
                time_axis = (np.arange(samples, dtype=np.float64) + position_audio) / 48_000
                waveform = np.stack(
                    (
                        0.08 * np.sin(2 * np.pi * 220 * time_axis),
                        0.06 * np.sin(2 * np.pi * 330 * time_axis),
                    )
                ).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout="stereo")
                frame.sample_rate = 48_000
                frame.pts, frame.time_base = audio_start + position_audio, Fraction(1, 48_000)
                for packet in audio.encode(frame):
                    output.mux(packet)
                position_audio += samples
        for packet in video.encode(None):
            output.mux(packet)
        for packet in audio.encode(None):
            output.mux(packet)


def generate(
    directory: Path, *, duration: float = 2.0, width: int = 1920, height: int = 1080, rate: int = 24
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    canvas = _photo_canvas(width, height)
    paths = {
        "static": directory / "static.mp4",
        "depth_change": directory / "depth_change.mp4",
        "vfr_offset": directory / "vfr_offset.mp4",
    }
    _one(paths["static"], canvas, duration, rate)
    _one(paths["depth_change"], canvas, duration, rate, depth=True)
    _one(paths["vfr_offset"], canvas, duration, rate, vfr=True)
    manifest = {
        "license": "Synthetic changes/audio MIT; source photo public domain, Andy Moreno",
        "source": "tests/fixtures/freediver_moalboal.jpg",
        "framing": "Aspect-preserving resize + neutral letterbox",
        "duration_requested": duration,
        "dimensions": [width, height],
        "fixtures": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "info": probe_video(path).to_dict(),
            }
            for name, path in paths.items()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/video-fixtures"))
    parser.add_argument("--duration", type=float, default=2)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--rate", type=int, default=24)
    args = parser.parse_args()
    generate(
        args.output, duration=args.duration, width=args.width, height=args.height, rate=args.rate
    )
