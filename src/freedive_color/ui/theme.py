from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

STYLE = """
QMainWindow, QWidget { background: #20262b; color: #dbe3e7; }
QWidget { font-size: 13px; }
QLabel { background: transparent; }
QWidget#header { background: #263038; border-bottom: 1px solid #38444c; }
QLabel#brand { font-size: 20px; font-weight: 600; }
QLabel#subtle, QLabel#metadata { color: #99aab5; }
QLabel#metadata { font-size: 12px; }
QLabel#sectionTitle { color: #dbe3e7; font-weight: 600; font-size: 14px; }
QLabel#value { color: #b9d6db; min-width: 40px; }
QLabel#warning { color: #dcc294; font-size: 12px; }
QLabel#status { color: #b6c6d0; }
QPushButton, QToolButton {
  background: #303b43; border: 1px solid #43515a; border-radius: 5px;
  padding: 7px 12px; min-height: 19px; color: #dbe3e7;
}
QPushButton:hover, QToolButton:hover { background: #3a4953; border-color: #748c98; }
QPushButton:pressed { background: #243f48; }
QPushButton:focus, QToolButton:focus, QComboBox:focus { border: 1px solid #92d4d5; }
QPushButton:checked { background: #335358; border: 1px solid #75c9cb; color: #e3f7f7; }
QPushButton:disabled, QToolButton:disabled {
  color: #6e808c; background: #252e34; border-color: #35414a;
}
QPushButton#primary {
  background: #75c9cb; color: #14282e; border-color: #75c9cb; font-weight: 600;
}
QPushButton#primary:hover { background: #94dcdd; }
QPushButton#primary:disabled { background: #35494e; color: #72888f; border-color: #35494e; }
QListWidget { background: #20282f; border: none; outline: none; padding: 5px; }
QListWidget::item {
  border: 1px solid transparent; border-radius: 5px; padding: 9px 5px; margin: 3px 0;
}
QListWidget::item:selected { background: #30454e; border: 1px solid #6ca7af; }
QListWidget::item:hover { background: #2d3b44; }
QScrollArea { border: none; background: #20262b; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #2b363e; color: #b9cbd3; padding: 9px 12px; }
QTabBar::tab:selected { background: #36535a; color: #edf7f7; border-bottom: 2px solid #75c9cb; }
QSplitter::handle { background: #39454e; width: 1px; }
QSlider::groove:horizontal { height: 3px; background: #42515d; border-radius: 1px; }
QSlider::sub-page:horizontal { background: #79b7c1; border-radius: 1px; }
QSlider::handle:horizontal {
  background: #d5e7ed; width: 12px; margin: -5px 0; border-radius: 6px; border: 1px solid #91afb8;
}
QSlider::handle:horizontal:focus { border: 2px solid #75c9cb; }
QComboBox, QSpinBox {
  background: #2b363e; border: 1px solid #43515a; border-radius: 4px; padding: 6px;
}
QComboBox QAbstractItemView { background: #2b363e; selection-background-color: #38545c; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QProgressBar { border: none; background: #36434d; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: #75c9cb; border-radius: 2px; }
QToolTip { background: #35454e; color: #f0f4f6; border: 1px solid #607b87; padding: 6px; }
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    families = QFontDatabase.families()
    family = next(
        (name for name in ("Avenir Next", "Segoe UI", "Noto Sans") if name in families),
        app.font().family(),
    )
    app.setFont(QFont(family, 11))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#20262b"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#dbe3e7"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#263038"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#dbe3e7"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#303b43"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#dbe3e7"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#335358"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#edf8f8"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
