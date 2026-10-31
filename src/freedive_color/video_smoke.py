"""Run real import, seek, paired playback and video export in the packaged GUI."""

import argparse
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtWidgets import QApplication

from .core.masks import MODEL_PATH, MODEL_SHA256
from .core.project import sha256_file
from .core.skin import SKIN_MODEL_ID, SKIN_MODEL_PATH, SKIN_MODEL_SHA256
from .ui.main_window import MainWindow
from .ui.theme import apply_theme


class VideoSmoke(QObject):
    def __init__(self, app: QApplication, source: Path, directory: Path):
        super().__init__(app)
        self.app, self.source, self.directory = app, source, directory
        self.source_hash = sha256_file(source)
        self.started = self.last_tick = time.monotonic()
        self.phase = "import"
        self.finished = False
        self.ticks = 0
        self.max_tick_gap = 0.0
        self.ready_count = 0
        self.ready_before_stop = 0
        self.playback_times = []
        self.preview_report = self.export_report = None
        self.events, self.errors = [], []
        self.preview_path = directory / "video_preview.mp4"
        self.output_path = directory / "video_export.mp4"
        self.screenshot_path = directory / "video_app_window.png"
        self.record_path = directory / "video_smoke.json"
        self.model_present = MODEL_PATH.is_file() and sha256_file(MODEL_PATH) == MODEL_SHA256
        self.skin_model_present = (
            SKIN_MODEL_PATH.is_file() and sha256_file(SKIN_MODEL_PATH) == SKIN_MODEL_SHA256
        )
        self.mask_metadata = None
        self.video_info = None
        self.seek_requested = self.seek_actual = 0.0
        self.window = MainWindow()
        self.window.preview_ready.connect(self._ready)
        self.window.video_preview_completed.connect(self._preview_completed)
        self.window.video_export_completed.connect(self._export_completed)
        self.window.operation_failed.connect(self.fail)
        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(20)
        self.tick_timer.timeout.connect(self._tick)
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(lambda: self.fail(f"Video GUI timed out in {self.phase}"))

    def start(self):
        self.window.show()
        self.window.raise_()
        self.last_tick = time.monotonic()
        self.tick_timer.start()
        self.deadline.start(60_000)
        QTimer.singleShot(0, lambda: self.window.import_paths([self.source]))

    def _event(self, phase):
        self.phase = phase
        self.events.append({"phase": phase, "seconds": round(time.monotonic() - self.started, 3)})

    def _ready(self):
        self.ready_count += 1

    def _view_ready(self):
        return (
            self.window.preview is not None
            and not self.window._render_pending
            and self.window._view_job is None
        )

    def _preview_completed(self, report):
        self.preview_report = report

    def _export_completed(self, report):
        self.export_report = report

    @Slot(object, object, float)
    def _frame(self, before, after, seconds):
        if self.phase != "playback":
            return
        if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
            self.fail("Paired playback produced incompatible image buffers")
            return
        if self.playback_times and seconds <= self.playback_times[-1]:
            self.fail("Paired playback presentation times did not increase")
            return
        self.playback_times.append(seconds)

    def _tick(self):
        if self.finished:
            return
        now = time.monotonic()
        self.max_tick_gap = max(self.max_tick_gap, now - self.last_tick)
        self.last_tick = now
        self.ticks += 1
        try:
            if self.phase == "import" and self._view_ready():
                if self.window.video_info is None:
                    raise RuntimeError("The actual GUI did not recognize the source video")
                self.video_info = self.window.video_info.to_dict()
                masks = self.window.preview.masks
                if not self.model_present or not self.skin_model_present or masks is None:
                    raise RuntimeError("Bundled segmentation model or actual local masks missing")
                self.mask_metadata = masks.to_metadata()
                if (
                    masks.skin_source != SKIN_MODEL_ID
                    or masks.skin_model_sha256 != SKIN_MODEL_SHA256
                ):
                    raise RuntimeError("The packaged GUI did not execute the bundled skin model")
                if masks.method != "farneback-alpha-v2":
                    raise RuntimeError(
                        "The packaged GUI did not use the shared temporal mask pipeline"
                    )
                self.seek_requested = min(0.19, self.window.video_info.duration / 4)
                self._event("seek")
                self.window.seek_video(self.seek_requested)
            elif self.phase == "seek" and self._view_ready():
                self.seek_actual = self.window.video_frame_seconds
                if self.seek_actual + 1e-6 < self.seek_requested:
                    raise RuntimeError("Seek returned a frame earlier than its requested position")
                remaining = self.window.video_info.duration - self.seek_requested
                if remaining < 0.2:
                    raise RuntimeError(
                        "Use a longer smoke fixture with at least three video frames"
                    )
                self._event("preview")
                self.window.create_video_preview(
                    self.seek_requested, min(1.2, remaining), self.preview_path
                )
            elif (
                self.phase == "preview"
                and self.preview_report is not None
                and self.window._export_job is None
            ):
                if not self.preview_path.is_file():
                    raise RuntimeError("The actual preview command did not produce its media file")
                info = self.preview_report["verification"]["actual_output_info"]
                if self.video_info["audio_streams"] and not info["audio_streams"]:
                    raise RuntimeError("The actual GUI preview lost the source audio stream")
                self._event("playback")
                self.window.toggle_video_playback()
                player = self.window.video_playback
                if player is None or not self.window._video_playing:
                    raise RuntimeError("Actual paired proxy playback did not start")
                player.frame_ready.connect(self._frame)
                player.failed.connect(self.fail)
            elif self.phase == "playback" and len(self.playback_times) >= 3:
                self.ready_before_stop = self.ready_count
                self._event("paused_frame")
                self.window.toggle_video_playback()
            elif (
                self.phase == "paused_frame"
                and not self.window._video_playing
                and self.ready_count > self.ready_before_stop
                and self._view_ready()
            ):
                self._event("export")
                self.window.start_video_export(self.output_path)
            elif (
                self.phase == "export"
                and self.export_report is not None
                and self.window._export_job is None
                and self._view_ready()
            ):
                verified = self.export_report["verification"]
                if not verified["pts_verified"] or not verified["audio_verified"]:
                    raise RuntimeError("The production exporter did not verify clocks and audio")
                if not self.output_path.is_file() or self.ticks < 10:
                    raise RuntimeError("Missing full export or insufficient live GUI timer events")
                if self.max_tick_gap > 2:
                    raise RuntimeError("The GUI event loop stalled for more than two seconds")
                self._event("capture")
                self.window.viewer.fit()
                QTimer.singleShot(150, lambda: self.finish(True))
        except Exception as exc:
            self.fail(str(exc))

    def fail(self, message):
        if self.finished:
            return
        self.errors.append(str(message))
        self.window.cancel_operation()
        self.window.stop_video_playback()
        self.finish(False)

    def finish(self, passed):
        if self.finished:
            return
        self.finished = True
        self.tick_timer.stop()
        self.deadline.stop()
        unchanged = self.source.is_file() and sha256_file(self.source) == self.source_hash
        if not unchanged:
            self.errors.append("The source video changed")
        focused = self.app.focusWidget()
        if focused:
            focused.clearFocus()
        screenshot_saved = self.window.grab().save(str(self.screenshot_path))
        if not screenshot_saved:
            self.errors.append("Could not save the actual application window screenshot")
        passed = bool(passed and unchanged and screenshot_saved and not self.errors)
        record = {
            "passed": passed,
            "phase": self.phase,
            "source_path": str(self.source),
            "source_sha256": self.source_hash,
            "source_unchanged": unchanged,
            "output_path": str(self.output_path),
            "preview_path": str(self.preview_path),
            "screenshot_path": str(self.screenshot_path),
            "qt_platform": QApplication.platformName(),
            "model_present": self.model_present,
            "model_sha256": MODEL_SHA256,
            "skin_model_present": self.skin_model_present,
            "skin_model_sha256": SKIN_MODEL_SHA256,
            "masks": self.mask_metadata,
            "video_info": self.video_info,
            "seek_requested_seconds": self.seek_requested,
            "seek_actual_seconds": self.seek_actual,
            "playback_frame_count": len(self.playback_times),
            "playback_times": self.playback_times,
            "pause_waited_for_preview_ready": self.ready_count > self.ready_before_stop,
            "responsiveness": {
                "tick_count": self.ticks,
                "timer_interval_ms": 20,
                "maximum_tick_gap_seconds": self.max_tick_gap,
            },
            "preview_report": self.preview_report,
            "export_report": self.export_report,
            "events": self.events,
            "errors": self.errors,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        try:
            with self.record_path.open("x", encoding="utf-8") as output:
                json.dump(record, output, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"Cannot write video smoke evidence: {exc}", file=sys.stderr)
            passed = False
        self.window.close()
        self.app.exit(0 if passed else 2)


def run(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-video", required=True, type=Path, metavar="VIDEO")
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args(arguments)
    source, directory = (
        args.smoke_video.expanduser().resolve(),
        args.evidence.expanduser().resolve(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "video_smoke.json",
        "video_app_window.png",
        "video_preview.mp4",
        "video_export.mp4",
    ):
        path = directory / name
        if path == source or path.exists():
            print("Choose a fresh video smoke evidence directory; existing files are protected.")
            return 2
    app = QApplication([sys.argv[0]])
    app.setApplicationName("Freedive Color Studio")
    apply_theme(app)
    try:
        controller = VideoSmoke(app, source, directory)
    except Exception as exc:
        with (directory / "video_smoke.json").open("x", encoding="utf-8") as output:
            json.dump({"passed": False, "phase": "startup", "errors": [str(exc)]}, output, indent=2)
        return 2
    controller.start()
    return app.exec()
