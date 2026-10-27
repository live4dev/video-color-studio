import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import apply_theme


def run(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    app = QApplication([sys.argv[0]])
    app.setApplicationName("Freedive Color Studio")
    app.setOrganizationName("Freedive Color Studio")
    app.setApplicationVersion("0.1.0")
    apply_theme(app)
    window = MainWindow()
    window.show()
    paths = [Path(value) for value in arguments if not value.startswith("-")]
    if paths:
        QTimer.singleShot(0, lambda: window.import_paths(paths))
    return app.exec()
