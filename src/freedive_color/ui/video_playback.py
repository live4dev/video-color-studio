"""Bounded paired playback of an already graded proxy; no frame-by-frame regrading."""

import time
from fractions import Fraction
from pathlib import Path
from threading import Event

import cv2
from PySide6.QtCore import QThread, Signal


class VideoPlayback(QThread):
    frame_ready = Signal(object, object, float)
    failed = Signal(str)

    def __init__(self, source: Path, proxy: Path, start_seconds: float, parent=None):
        super().__init__(parent)
        self.source, self.proxy, self.start_seconds = source, proxy, start_seconds
        self.stop_event = Event()
        self.acknowledged = Event()

    def stop(self):
        self.stop_event.set()
        self.acknowledged.set()

    def run(self):
        try:
            from freedive_color.core.video_pipeline import decode_video_frame, probe_video
            from freedive_color.integrations.ffmpeg import open_local_media

            source_info, proxy_info = probe_video(self.source), probe_video(self.proxy)
            with (
                open_local_media(self.source) as original,
                open_local_media(self.proxy) as processed,
            ):
                original_stream, proxy_stream = (
                    original.streams.video[0],
                    processed.streams.video[0],
                )
                origin = source_info.video_start
                # UI seconds are floats; recover an exact source tick before frame selection.
                # Otherwise .208333… can land infinitesimally before its intended VFR frame.
                offset_ticks = round(Fraction(str(self.start_seconds)) / source_info.time_base)
                target_start = origin + offset_ticks * source_info.time_base
                original.seek(
                    int(target_start / original_stream.time_base),
                    stream=original_stream,
                    backward=True,
                )
                source_frames = iter(original.decode(original_stream))
                next_source = next(source_frames, None)
                current_source = next_source
                proxy_origin = None
                wall_start = time.monotonic()
                for frame in processed.decode(proxy_stream):
                    if self.stop_event.is_set():
                        break
                    timestamp = Fraction(frame.pts) * frame.time_base
                    if proxy_origin is None:
                        proxy_origin = timestamp
                    elapsed = float(timestamp - proxy_origin)
                    target = target_start + (timestamp - proxy_origin)
                    while next_source is not None:
                        source_time = Fraction(next_source.pts) * next_source.time_base
                        if source_time > target:
                            break
                        current_source = next_source
                        next_source = next(source_frames, None)
                    chosen = current_source or next_source
                    if chosen is None:
                        break
                    before = decode_video_frame(chosen, source_info)
                    after = decode_video_frame(frame, proxy_info)
                    if before.shape != after.shape:
                        before = cv2.resize(
                            before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA
                        )
                    delay = wall_start + elapsed - time.monotonic()
                    if delay > 0 and self.stop_event.wait(delay):
                        break
                    self.acknowledged.clear()
                    self.frame_ready.emit(
                        before, after, float(Fraction(chosen.pts) * chosen.time_base - origin)
                    )
                    while not self.stop_event.is_set() and not self.acknowledged.wait(0.1):
                        pass
        except Exception as exc:
            if not self.stop_event.is_set():
                self.failed.emit(str(exc))
