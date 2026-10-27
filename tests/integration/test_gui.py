import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QMimeData, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest

from freedive_color.core.media_io import load_photo
from freedive_color.ui.main_window import MainWindow
from freedive_color.ui.theme import apply_theme


@pytest.fixture
def window(qapp, qtbot):
    apply_theme(qapp)
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    widget.close()


def wait_ready(qtbot, window):
    qtbot.waitUntil(
        lambda: (
            window.preview is not None and not window._render_pending and window._view_job is None
        ),
        timeout=15_000,
    )


def test_drag_drop_controls_split_and_export(window, qtbot, photo_path, tmp_path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(photo_path))])
    drag = QDragEnterEvent(
        QPoint(40, 40),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QCoreApplication.sendEvent(window, drag)
    assert drag.isAccepted()
    drop = QDropEvent(
        QPointF(40, 40),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QCoreApplication.sendEvent(window, drop)
    wait_ready(qtbot, window)
    assert window.library.count() == 1
    assert not window.viewer.after.isNull()
    image_rect = window.viewer.image_rect()
    assert image_rect.width() > window.viewer.width() * 0.6
    assert abs(image_rect.center().y() - window.viewer.height() / 2) < 1
    before = window.preview.result.copy()
    QTest.mouseClick(window.controls.preset_buttons["vivid"], Qt.MouseButton.LeftButton)
    wait_ready(qtbot, window)
    assert np.mean(np.abs(before - window.preview.result)) > 0.001
    slider = window.controls.sliders["exposure"]
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    wait_ready(qtbot, window)
    assert window.controls.settings().exposure == 0.01
    viewer = window.viewer
    rect = viewer.image_rect()
    point = QPoint(round(rect.center().x()), round(rect.center().y()))
    QTest.mousePress(viewer, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(viewer, QPoint(point.x() + 40, point.y()))
    QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton)
    assert viewer.split > 0.5
    QTest.mouseClick(window.mode_buttons["after"], Qt.MouseButton.LeftButton)
    assert viewer.mode == "after"
    output = tmp_path / "gui.png"
    with qtbot.waitSignal(window.export_completed, timeout=15_000):
        window.start_export(output)
    qtbot.waitUntil(lambda: window._export_job is None)
    assert output.exists() and window.export_button.isEnabled()
    np.testing.assert_allclose(load_photo(output).rgb, window.preview.result, atol=1.1 / 255)


def test_ui_stays_responsive_during_processing(window, qtbot, photo_path, monkeypatch):
    import threading

    import freedive_color.ui.main_window as ui

    original_load = ui.load_photo

    def slow_load(*args, **kwargs):
        threading.Event().wait(0.3)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(ui, "load_photo", slow_load)
    window.import_paths([photo_path])
    events = []
    QTimer.singleShot(30, lambda: events.append("main event loop ran"))
    qtbot.waitUntil(lambda: bool(events), timeout=200)
    assert window.preview is None
    wait_ready(qtbot, window)


def test_rapid_selection_ignores_stale_worker(window, qtbot, photo_path, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (90, 60), (20, 80, 100)).save(other)
    window.import_paths([photo_path, other])
    window.library.setCurrentRow(1)
    wait_ready(qtbot, window)
    assert window.preview.photo.path == other
    assert window.preview.photo.width == 90


def test_full_resolution_one_to_one(window, qtbot, tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (1800, 1500), (40, 90, 130)).save(path)
    window.import_paths([path])
    wait_ready(qtbot, window)
    assert window.preview.photo.rgb.shape[1] < 1800
    QTest.mouseClick(window.one_button, Qt.MouseButton.LeftButton)
    wait_ready(qtbot, window)
    assert window.preview.photo.rgb.shape == (1500, 1800, 3)
    assert window.viewer.before.width() == 1800
    assert window.viewer.fitting is False


def test_read_error_is_visible_and_next_import_works(window, qtbot, photo_path, tmp_path):
    broken = tmp_path / "corrupt.jpg"
    broken.write_bytes(b"bad photo")
    with qtbot.waitSignal(window.operation_failed, timeout=10_000):
        window.import_paths([broken])
    assert "повреждён" in window.status_label.text()
    window.import_paths([photo_path])
    wait_ready(qtbot, window)
    assert window.export_button.isEnabled()


def test_loading_precise_settings_does_not_round_them(window):
    from dataclasses import replace

    from freedive_color.core.presets import get_preset

    exact = replace(get_preset("natural"), temperature=0.012345, exposure=0.125)
    window.controls.set_settings(exact, emit=False)
    assert window.controls.settings() == exact
    window.controls.sliders["contrast"].setValue(20)
    assert window.controls.settings().temperature == 0.012345
