"""The custom media cards retain native selection and real series check state."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QListWidgetItem

from freedive_color.ui.library import MediaCardDelegate, MediaLibrary


@pytest.fixture
def library(qapp, qtbot):
    widget = MediaLibrary()
    widget.resize(235, 450)
    qtbot.addWidget(widget)
    thumbnail = QPixmap(200, 112)
    thumbnail.fill(QColor("#287a9c"))
    for filename in ("погружение.png", "второе очень длинное имя.mp4", "third.tiff"):
        item = QListWidgetItem(QIcon(thumbnail), filename + "\nГотово")
        item.setData(Qt.ItemDataRole.UserRole, "/fixtures/" + filename)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        widget.addItem(item)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def test_card_checkbox_and_space_change_real_batch_inclusion(library):
    item = library.item(0)
    rect = library.visualItemRect(item)
    checkbox = MediaCardDelegate.checkbox_rect(rect)
    QTest.mouseClick(library.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center())
    assert item.checkState() == Qt.CheckState.Unchecked
    QTest.mouseClick(library.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center())
    assert item.checkState() == Qt.CheckState.Checked
    # A card body click selects without accidentally toggling the old native checkbox area.
    QTest.mouseClick(
        library.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.topLeft() + QPoint(20, 70),
    )
    assert library.currentItem() is item
    assert item.checkState() == Qt.CheckState.Checked
    QTest.keyClick(library, Qt.Key.Key_Space)
    assert item.checkState() == Qt.CheckState.Unchecked


def test_cards_keep_control_selection_and_fit_narrow_library(library, qtbot):
    first, second = library.item(0), library.item(1)
    QTest.mouseClick(
        library.viewport(),
        Qt.MouseButton.LeftButton,
        pos=library.visualItemRect(first).center(),
    )
    QTest.mouseClick(
        library.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        pos=library.visualItemRect(second).center(),
    )
    assert len(library.selectedItems()) == 2
    assert first in library.selectedItems() and second in library.selectedItems()
    assert all(item.checkState() == Qt.CheckState.Checked for item in (first, second))
    library.resize(200, 400)
    qtbot.waitUntil(lambda: library.viewport().width() <= 200)
    assert library.visualItemRect(first).width() <= library.viewport().width()
    assert library.horizontalScrollBar().maximum() == 0
    # Exercise the custom painter with real icon, Unicode filename, selection and check state.
    assert not library.grab().isNull()
