"""Inspector navigation remains usable at the supported compact desktop size."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QScrollArea

from freedive_color.ui.main_window import MainWindow
from freedive_color.ui.theme import apply_theme


def test_compact_inspector_navigation_and_visible_export_controls(qapp, qtbot, photo_path):
    apply_theme(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1080, 720)
    window.show()
    try:
        window.import_paths([photo_path])
        qtbot.waitUntil(
            lambda: (
                window.preview is not None
                and not window._render_pending
                and window._view_job is None
            ),
            timeout=15_000,
        )
        tabs = window.settings_tabs
        tab_bar = tabs.tabBar()
        correction_page = tabs.currentWidget()
        assert isinstance(correction_page, QScrollArea)
        assert correction_page.verticalScrollBar().maximum() > 0
        tab_position = tab_bar.mapTo(window, QPoint(0, 0))
        correction_page.verticalScrollBar().setValue(correction_page.verticalScrollBar().maximum())
        qapp.processEvents()
        assert tab_bar.mapTo(window, QPoint(0, 0)) == tab_position
        assert not tab_bar.visibleRegion().isEmpty()
        assert all(tab_bar.rect().contains(tab_bar.tabRect(i).center()) for i in range(3))

        # Existing mask-preview/smoke callers select the controls directly, not their wrapper.
        tabs.setCurrentWidget(window.local_controls)
        assert tabs.currentWidget().widget() is window.local_controls
        assert window.local_controls.tool_combo.isVisible()
        QTest.mouseClick(tab_bar, Qt.MouseButton.LeftButton, pos=tab_bar.tabRect(2).center())
        assert tabs.currentWidget().widget() is window.export_panel
        assert window.format_combo.isVisible()
        assert not window.format_combo.visibleRegion().isEmpty()

        # Drive the newly separate Export page with the keyboard as a user would.
        window.format_combo.setFocus()
        QTest.keyClick(window.format_combo, Qt.Key.Key_Home)
        QTest.keyClick(window.format_combo, Qt.Key.Key_Down)
        assert window.format_combo.currentData() == ".png"
        assert window.depth_combo.isEnabled() and not window.quality_spin.isEnabled()
        window.depth_combo.setFocus()
        QTest.keyClick(window.depth_combo, Qt.Key.Key_End)
        assert window._export_options().bit_depth == 16
        window.format_combo.setFocus()
        QTest.keyClick(window.format_combo, Qt.Key.Key_End)
        assert window.format_combo.currentData() == ".tiff"
        assert not window.gps_check.isEnabled()
        QTest.keyClick(window.format_combo, Qt.Key.Key_Home)
        assert window._export_options().suffix == ".jpg"
        assert window._export_options().bit_depth == 8
        assert window.quality_spin.isEnabled()
        assert not window.depth_combo.isEnabled()
        assert window.width() == 1080 and window.height() == 720
    finally:
        window.close()
