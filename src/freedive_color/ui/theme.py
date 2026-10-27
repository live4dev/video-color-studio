"""Neutral editing surfaces and a restrained accent for the desktop workspace."""

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

STYLE = """
QMainWindow, QWidget { background: #202023; color: #ededf0; }
QWidget { font-size: 12px; }
QLabel { background: transparent; border: none; }
QWidget#header { background: #18181b; border-bottom: 1px solid #303034; }
QLabel#brand { font-size: 15px; font-weight: 600; color: #f6f6f7; }
QLabel#workspaceLabel { font-size: 11px; color: #96969e; }
QLabel#localBadge {
  color: #b8b8bf; font-size: 11px; background: #242427;
  border: 1px solid #343438; border-radius: 4px; padding: 4px 8px;
}
QWidget#libraryPanel, QWidget#inspectorPanel { background: #202023; }
QWidget#viewerPanel { background: #101012; }
QWidget#panelHeader { background: #202023; border-bottom: 1px solid #303034; }
QLabel#panelHeader { padding: 15px 16px; font-size: 13px; font-weight: 600; }
QLabel#panelTitle { font-size: 13px; font-weight: 600; }
QWidget#viewerToolbar { background: #202023; border-bottom: 1px solid #303034; }
QWidget#viewerFooter { background: #202023; border-top: 1px solid #303034; }
QWidget#footer { background: #18181b; border-top: 1px solid #303034; }
QWidget#videoPanel { background: #202023; border-top: 1px solid #343438; }
QLabel#subtle, QLabel#metadata { color: #96969e; }
QLabel#metadata { font-size: 11px; }
QLabel#sectionTitle { font-size: 12px; font-weight: 600; color: #ededf0; }
QLabel#sectionTitle[separated="true"] {
  border-top: 1px solid #343438; padding-top: 14px;
}
QLabel#value {
  color: #d4d4da; font-size: 11px; background: #2c2c30;
  border: 1px solid #37373c; border-radius: 3px;
  padding: 3px 6px; min-width: 45px;
}
QLabel#warning { color: #e5bd89; font-size: 11px; }
QLabel#status { color: #a6a6af; font-size: 11px; }
QPushButton, QToolButton {
  background: #2c2c30; color: #ededf0; border: 1px solid #3c3c42;
  border-radius: 4px; padding: 5px 10px; min-height: 18px;
}
QPushButton:hover, QToolButton:hover { background: #38383e; border-color: #505058; }
QPushButton:pressed, QToolButton:pressed { background: #242428; }
QPushButton:focus, QToolButton:focus { border: 1px solid #63e4d4; }
QPushButton:checked, QToolButton:checked {
  background: #263a38; color: #75ebdc; border: 1px solid #63e4d4;
}
QPushButton:disabled, QToolButton:disabled {
  color: #686871; background: #262629; border-color: #303034;
}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {
  background: transparent; border-color: transparent;
}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {
  background: #303035; border-color: #414147;
}
QPushButton[variant="ghost"]:focus, QToolButton[variant="ghost"]:focus {
  border-color: #63e4d4;
}
QPushButton[variant="segment"] {
  background: transparent; border-color: transparent; padding: 4px 9px;
}
QPushButton[variant="segment"]:hover { background: #303035; }
QPushButton[variant="segment"]:checked {
  background: #3b3b41; color: #f6f6f7; border-color: #4b4b53;
}
QPushButton[variant="segment"]:focus { border-color: #63e4d4; }
QPushButton[variant="preset"] {
  background: #28282c; border-color: #414147; padding: 5px 3px;
  min-height: 30px; font-size: 11px;
}
QPushButton[variant="preset"]:hover { background: #343439; border-color: #72727b; }
QPushButton[variant="preset"]:checked {
  background: #263a38; color: #81efe1; border-color: #63e4d4;
}
QPushButton#primary {
  background: #63e4d4; color: #102723; border-color: #63e4d4;
  font-weight: 600; padding-left: 15px; padding-right: 15px;
}
QPushButton#primary:hover { background: #85eee1; border-color: #85eee1; }
QPushButton#primary:pressed { background: #4dc9ba; border-color: #4dc9ba; }
QPushButton#primary:focus { border: 1px solid #d5fff9; }
QPushButton#primary:disabled {
  background: #2c4540; color: #6a8b84; border-color: #2c4540;
}
QListWidget { background: #202023; border: none; outline: none; padding: 3px; }
QListWidget::item {
  border: 1px solid transparent; border-radius: 4px; padding: 7px 4px; margin: 2px;
}
QListWidget::item:selected { background: #303b39; border-color: #63e4d4; }
QListWidget::item:hover { background: #2c2c30; }
QListWidget::item:selected:!active { background: #303b39; border-color: #63e4d4; }
QScrollArea { border: none; background: #202023; }
QScrollArea > QWidget > QWidget { background: #202023; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 1px; }
QScrollBar::handle:vertical { background: #505057; min-height: 28px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #71717b; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 1px 2px; }
QScrollBar::handle:horizontal { background: #505057; min-width: 28px; border-radius: 3px; }
QScrollBar::handle:horizontal:hover { background: #71717b; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QTabWidget::pane { border: none; background: #202023; }
QTabBar { background: #202023; }
QTabBar::tab {
  background: #202023; color: #96969e; border: none;
  border-bottom: 2px solid transparent; padding: 13px 12px 11px;
}
QTabBar::tab:hover { color: #ededf0; background: #27272b; }
QTabBar::tab:selected { color: #ededf0; border-bottom: 2px solid #63e4d4; }
QTabBar::tab:focus { border-bottom-color: #c1fff6; }
QSplitter::handle { background: #101012; width: 5px; }
QSplitter::handle:hover { background: #4a4a50; }
QSlider { min-height: 18px; background: transparent; }
QSlider::groove:horizontal { height: 3px; background: #45454d; border-radius: 1px; }
QSlider::sub-page:horizontal { background: #b8b8c1; border-radius: 1px; }
QSlider::handle:horizontal {
  background: #ededf0; width: 10px; height: 10px; margin: -4px 0;
  border-radius: 5px; border: 1px solid #ededf0;
}
QSlider::handle:horizontal:hover { background: #63e4d4; border-color: #63e4d4; }
QSlider::handle:horizontal:focus { border: 2px solid #63e4d4; }
QSlider::groove:horizontal:disabled { background: #303035; }
QSlider::sub-page:horizontal:disabled { background: #515159; }
QSlider::handle:horizontal:disabled { background: #74747c; border-color: #74747c; }
QSlider#video_timeline::sub-page:horizontal { background: #63e4d4; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
  background: #2c2c30; color: #dedee3; border: 1px solid #3c3c42;
  border-radius: 4px; padding: 5px 7px; min-height: 18px;
  selection-background-color: #367e73; selection-color: #ffffff;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: #5a5a63; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #63e4d4; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
  color: #74747d; background: #252528; border-color: #303034;
}
QComboBox { padding-right: 22px; }
QComboBox::drop-down { width: 20px; border: none; }
QComboBox::down-arrow {
  image: url("__ICON_ROOT__/chevron-down.svg"); width: 12px; height: 12px;
}
QComboBox QAbstractItemView {
  background: #2c2c30; color: #ededf0; border: 1px solid #4b4b52;
  selection-background-color: #3a514b; selection-color: #ededf0; outline: 0;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
  width: 16px; border: none; background: transparent;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
  image: url("__ICON_ROOT__/chevron-up.svg"); width: 10px; height: 10px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
  image: url("__ICON_ROOT__/chevron-down.svg"); width: 10px; height: 10px;
}
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox:disabled { color: #74747d; }
QCheckBox::indicator {
  width: 14px; height: 14px; border: 1px solid #777780;
  border-radius: 3px; background: #2c2c30;
}
QCheckBox::indicator:checked {
  background: #63e4d4; border-color: #63e4d4; image: url("__ICON_ROOT__/check.svg");
}
QCheckBox::indicator:hover { border-color: #63e4d4; }
QCheckBox::indicator:disabled { background: #303034; border-color: #505057; }
QCheckBox:focus { color: #93f4e7; }
QProgressBar { border: none; background: #343438; border-radius: 1px; max-height: 2px; }
QProgressBar::chunk { background: #63e4d4; border-radius: 1px; }
QMenuBar { background: #18181b; color: #bdbdc5; border: none; }
QMenuBar::item { background: transparent; padding: 5px 9px; }
QMenuBar::item:selected { background: #333338; color: #ededf0; }
QMenu { background: #28282c; color: #ededf0; border: 1px solid #44444b; padding: 5px; }
QMenu::item { padding: 7px 24px 7px 10px; border-radius: 3px; }
QMenu::item:selected { background: #3a3a41; }
QMenu::item:disabled { color: #777780; }
QMenu::separator { height: 1px; background: #414147; margin: 4px 5px; }
QToolTip {
  background: #343439; color: #f1f1f4; border: 1px solid #55555e;
  border-radius: 3px; padding: 6px 8px;
}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    families = QFontDatabase.families()
    family = next(
        (
            name
            for name in ("SF Pro Text", "Segoe UI", ".AppleSystemUIFont", "Noto Sans")
            if name in families
        ),
        app.font().family(),
    )
    font = QFont(family)
    font.setPixelSize(12)
    app.setFont(font)
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#202023"),
        (QPalette.ColorRole.WindowText, "#ededf0"),
        (QPalette.ColorRole.Base, "#18181b"),
        (QPalette.ColorRole.AlternateBase, "#2c2c30"),
        (QPalette.ColorRole.Text, "#ededf0"),
        (QPalette.ColorRole.Button, "#2c2c30"),
        (QPalette.ColorRole.ButtonText, "#ededf0"),
        (QPalette.ColorRole.Highlight, "#367e73"),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.ToolTipBase, "#343439"),
        (QPalette.ColorRole.ToolTipText, "#f1f1f4"),
        (QPalette.ColorRole.Link, "#63e4d4"),
        (QPalette.ColorRole.PlaceholderText, "#96969e"),
        (QPalette.ColorRole.Light, "#62626b"),
        (QPalette.ColorRole.Mid, "#45454d"),
        (QPalette.ColorRole.Dark, "#101012"),
    ):
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#74747d"))
    app.setPalette(palette)
    icons = (Path(__file__).resolve().parents[1] / "assets" / "icons").as_posix()
    app.setStyleSheet(STYLE.replace("__ICON_ROOT__", icons))
