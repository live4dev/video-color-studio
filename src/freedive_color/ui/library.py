"""Media cards over the native item view, including real batch inclusion checkboxes."""

from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
)

from .strings import _


class MediaCardDelegate(QStyledItemDelegate):
    """Draw the existing filename, thumbnail and export state as a compact card."""

    @staticmethod
    def checkbox_rect(rect: QRect) -> QRect:
        return QRect(rect.right() - 30, rect.top() + 12, 18, 18)

    def sizeHint(self, option, index) -> QSize:
        return QSize(1, 177)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(1, 1, -1, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        included = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        # Some models expose CheckStateRole as the enum's integer value.
        included = included or index.data(Qt.ItemDataRole.CheckStateRole) == 2
        painter.setOpacity(1.0 if enabled else 0.5)
        painter.setPen(QPen(QColor("#63e4d4" if selected else "#45454b" if hovered else "#343438")))
        painter.setBrush(QColor("#27272b" if selected or hovered else "#202023"))
        painter.drawRoundedRect(QRectF(rect), 6, 6)

        image_rect = rect.adjusted(7, 7, -7, -55)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#101012"))
        painter.drawRoundedRect(QRectF(image_rect), 3, 3)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and not icon.isNull():
            pixmap = icon.pixmap(image_rect.size())
            logical_size = pixmap.deviceIndependentSize().toSize()
            target = QRect(QRect(image_rect).center(), logical_size)
            target.moveCenter(image_rect.center())
            painter.save()
            painter.setOpacity((1.0 if included else 0.45) * (1.0 if enabled else 0.5))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(target, pixmap)
            painter.restore()

        path = str(index.data(Qt.ItemDataRole.UserRole) or "")
        suffix = Path(path).suffix.removeprefix(".").upper()
        if suffix:
            font = option.font
            font.setPixelSize(10)
            font.setWeight(font.Weight.DemiBold)
            painter.setFont(font)
            type_rect = QRect(image_rect.left() + 6, image_rect.top() + 6, 43, 19)
            painter.setBrush(QColor(24, 24, 27, 225))
            painter.drawRoundedRect(QRectF(type_rect), 3, 3)
            painter.setPen(QColor("#ededf0"))
            painter.drawText(type_rect, Qt.AlignmentFlag.AlignCenter, suffix)

        check = self.checkbox_rect(option.rect)
        painter.setPen(QPen(QColor("#63e4d4" if included else "#777780"), 1))
        painter.setBrush(QColor("#63e4d4" if included else "#18181b"))
        painter.drawRoundedRect(QRectF(check), 4, 4)
        if included:
            painter.setPen(QPen(QColor("#112422"), 1.7))
            painter.drawLine(check.left() + 4, check.top() + 9, check.left() + 8, check.top() + 13)
            painter.drawLine(check.left() + 8, check.top() + 13, check.left() + 14, check.top() + 5)

        parts = str(index.data(Qt.ItemDataRole.DisplayRole) or "").split("\n", 1)
        font = option.font
        font.setPixelSize(12)
        font.setWeight(font.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor("#ededf0"))
        title_rect = QRect(rect.left() + 11, rect.bottom() - 45, rect.width() - 22, 21)
        title = painter.fontMetrics().elidedText(
            parts[0], Qt.TextElideMode.ElideMiddle, title_rect.width()
        )
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, title)
        font.setPixelSize(11)
        font.setWeight(font.Weight.Normal)
        painter.setFont(font)
        status = parts[1] if len(parts) == 2 else _("Импортирован")
        if not included:
            status = _("Исключён из серии")
        painter.setPen(QColor("#96969e"))
        status_rect = QRect(rect.left() + 11, rect.bottom() - 24, rect.width() - 22, 17)
        painter.drawText(
            status_rect,
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                status, Qt.TextElideMode.ElideRight, status_rect.width()
            ),
        )
        if option.state & QStyle.StateFlag.State_HasFocus:
            painter.setPen(QPen(QColor("#ededf0"), 1, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(rect.adjusted(3, 3, -3, -3)), 4, 4)
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if not (index.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return False
        if not (index.flags() & Qt.ItemFlag.ItemIsEnabled):
            return False
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            return event.button() == Qt.MouseButton.LeftButton and self.checkbox_rect(
                option.rect
            ).contains(event.position().toPoint())
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton or not self.checkbox_rect(
                option.rect
            ).contains(event.position().toPoint()):
                return False
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() not in (Qt.Key.Key_Space, Qt.Key.Key_Select):
                return False
        else:
            return False
        checked = index.data(Qt.ItemDataRole.CheckStateRole)
        checked = checked == Qt.CheckState.Checked or checked == 2
        return model.setData(
            index,
            Qt.CheckState.Unchecked.value if checked else Qt.CheckState.Checked.value,
            Qt.ItemDataRole.CheckStateRole,
        )


class MediaLibrary(QListWidget):
    """A QListWidget-compatible library with keyboard and multiselection intact."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library")
        self.setAccessibleName(_("Медиафайлы. Отметки включают файлы в экспорт серии."))
        self.setItemDelegate(MediaCardDelegate(self))
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(5)
        self.setMouseTracking(True)
        self.setUniformItemSizes(True)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count():
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor("#96969e"))
        painter.drawText(
            self.viewport().rect().adjusted(18, 30, -18, -30),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
            _("Здесь появятся ваши файлы"),
        )
