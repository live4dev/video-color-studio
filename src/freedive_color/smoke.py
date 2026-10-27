"""Exercise the actual desktop window, asynchronous preview, controls and export."""

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .core.project import sha256_file
from .ui.main_window import MainWindow
from .ui.theme import apply_theme


def run(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", required=True, type=Path, metavar="PHOTO")
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args(arguments)
    args.evidence.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(args.smoke_test)
    app = QApplication([sys.argv[0]])
    apply_theme(app)
    window = MainWindow()
    window.show()
    window.raise_()
    phase = 0
    completed = False
    errors = []

    def fail(message):
        errors.append(message)
        window.close()
        app.exit(2)

    def ready():
        nonlocal phase
        if phase == 0:
            phase = 1
            window.controls.set_preset("natural")
            window.controls.sliders["exposure"].setValue(15)
        elif phase == 1:
            phase = 2
            window.viewer.split = 0.48
            window.start_export(args.evidence / "gui_export.png")

    def finish(report):
        nonlocal completed
        completed = True
        unchanged = sha256_file(args.smoke_test) == source_hash
        record = {
            "passed": unchanged,
            "source_unchanged": unchanged,
            "source_sha256": source_hash,
            "controls_used": {"preset": "natural", "exposure": 0.15},
            "output": report,
            "errors": errors,
        }
        (args.evidence / "gui_smoke.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        def capture_and_exit():
            window.grab().save(str(args.evidence / "app_window.png"))
            window.close()
            app.exit(0 if unchanged else 2)

        QTimer.singleShot(150, capture_and_exit)

    window.preview_ready.connect(ready)
    window.export_completed.connect(finish)
    window.operation_failed.connect(fail)
    QTimer.singleShot(100, lambda: window.import_paths([args.smoke_test]))
    QTimer.singleShot(45_000, lambda: fail("GUI smoke timed out") if not completed else None)
    return app.exec()
