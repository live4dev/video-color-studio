import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QColorSpace,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .strings import _


def to_qimage(rgb: np.ndarray, alpha: np.ndarray | None = None) -> QImage:
    pixels = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if alpha is not None:
        pixels = np.dstack((pixels, np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)))
    pixels = np.ascontiguousarray(pixels)
    fmt = QImage.Format.Format_RGBA8888 if alpha is not None else QImage.Format.Format_RGB888
    result = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], fmt).copy()
    result.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    return result


class CompareViewer(QWidget):
    zoom_changed = Signal(float)
    open_requested = Signal()
    stroke_created = Signal(object)
    waterline_changed = Signal(object)
    person_box_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.before = QImage()
        self.after = QImage()
        self.source_width = 1
        self.mode = "compare"
        self.split = 0.5
        self.scale = 1.0
        self.offset = QPointF()
        self.fitting = True
        self._dragging = ""
        self._last_position = QPointF()
        self.overlay = QImage()
        self.tool = "navigate"
        self.brush_radius = 0.02
        self.edits_enabled = True
        self._stroke_points = []
        self._hover_position = QPointF(-100, -100)
        self.setAccessibleName(_("Просмотр фото и видео до и после обработки"))

    def set_images(
        self, before: QImage, after: QImage, source_width: int, reset_transform: bool = False
    ) -> None:
        old_width = self.before.width()
        self.before, self.after, self.source_width = before, after, source_width
        if reset_transform:
            self.fitting = True
            self.offset = QPointF()
        elif old_width and not self.fitting:
            self.scale *= old_width / before.width()
        if self.fitting:
            self._fit_scale()
        self._emit_zoom()
        self.update()

    def clear(self) -> None:
        self.before, self.after = QImage(), QImage()
        self.overlay = QImage()
        self._stroke_points = []
        self.update()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._stroke_points = []
        self._dragging = ""
        self.update()

    def set_mask_overlay(self, mask: np.ndarray | None, target: str = "skin") -> None:
        if mask is None:
            self.overlay = QImage()
        else:
            colors = {
                "skin": [1, 0.57, 0.28],
                "water": [0.15, 0.8, 1],
                "person": [0.7, 0.55, 1],
                "underwater": [0.2, 1, 0.6],
            }
            rgb = np.empty((*mask.shape, 3), np.float32)
            rgb[:] = colors.get(target, colors["skin"])
            self.overlay = to_qimage(rgb, np.clip(mask, 0, 1) * 0.42)
        self.update()

    def normalized_point(self, position: QPointF) -> tuple[float, float] | None:
        rect = self.image_rect()
        if self.before.isNull() or not rect.contains(position):
            return None
        return (
            (position.x() - rect.left()) / rect.width(),
            (position.y() - rect.top()) / rect.height(),
        )

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def _fit_scale(self) -> None:
        if not self.before.isNull():
            self.scale = min(
                max(1, self.width() - 48) / self.before.width(),
                max(1, self.height() - 48) / self.before.height(),
            )

    def fit(self) -> None:
        self.fitting = True
        self.offset = QPointF()
        self._fit_scale()
        self._emit_zoom()
        self.update()

    def one_to_one(self) -> None:
        if self.before.isNull():
            return
        self.fitting = False
        self.scale = self.source_width / self.before.width() / self.devicePixelRatioF()
        self.offset = QPointF()
        self._emit_zoom()
        self.update()

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(
            self.scale * self.before.width() / max(1, self.source_width) * self.devicePixelRatioF()
        )

    def image_rect(self) -> QRectF:
        width, height = self.before.width() * self.scale, self.before.height() * self.scale
        return QRectF(
            (self.width() - width) / 2 + self.offset.x(),
            (self.height() - height) / 2 + self.offset.y(),
            width,
            height,
        )

    def resizeEvent(self, event) -> None:
        if self.fitting:
            self._fit_scale()
            self._emit_zoom()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101012"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale < 1)
        if self.before.isNull():
            self._empty_state(painter)
            return
        rect = self.image_rect()
        # Transparent source media stays transparent; checkerboard exists only in the viewer.
        painter.save()
        painter.setClipRect(rect)
        cell = 16
        painter.fillRect(rect, QColor("#29292d"))
        for y in range(max(0, int(rect.top())), min(self.height(), int(rect.bottom()) + 1), cell):
            for x in range(
                max(0, int(rect.left())), min(self.width(), int(rect.right()) + 1), cell
            ):
                if ((x - int(rect.left())) // cell + (y - int(rect.top())) // cell) % 2:
                    painter.fillRect(x, y, cell, cell, QColor("#333337"))
        painter.drawImage(rect, self.before if self.mode == "before" else self.after)
        if self.mode == "compare":
            boundary = rect.left() + rect.width() * self.split
            painter.setClipRect(
                QRectF(rect.left(), rect.top(), rect.width() * self.split, rect.height())
            )
            painter.drawImage(rect, self.before)
            painter.restore()
            painter.setPen(QPen(QColor("#ededf0"), 1.2))
            top = max(0.0, rect.top())
            bottom = min(float(self.height()), rect.bottom())
            if 0 <= boundary <= self.width():
                painter.drawLine(QPointF(boundary, top), QPointF(boundary, bottom))
                middle = min(max(self.height() * 0.5, top + 22), bottom - 22)
                painter.setBrush(QColor(29, 29, 32, 235))
                painter.setPen(QPen(QColor(237, 237, 240, 180), 1))
                painter.drawRoundedRect(QRectF(boundary - 17, middle - 14, 34, 28), 5, 5)
                painter.setPen(QPen(QColor("#ededf0"), 1.4))
                for offset in (-3, 3):
                    painter.drawLine(
                        QPointF(boundary + offset, middle - 5),
                        QPointF(boundary + offset, middle + 5),
                    )
            self._badge(painter, rect, _("До"), left=True)
            self._badge(painter, rect, _("После"), left=False)
        else:
            painter.restore()
            self._badge(painter, rect, _("До") if self.mode == "before" else _("После"), left=True)
        painter.save()
        painter.setClipRect(rect)
        if not self.overlay.isNull():
            painter.drawImage(rect, self.overlay)
        if self._stroke_points:
            points = [
                QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())
                for x, y in self._stroke_points
            ]
            painter.setPen(QPen(QColor("#ededf0"), 2))
            if self.tool == "person_box" and len(points) > 1:
                painter.drawRect(QRectF(points[0], points[-1]).normalized())
            else:
                path = QPainterPath(points[0])
                for point in points[1:]:
                    path.lineTo(point)
                painter.drawPath(path)
        if self.edits_enabled and self.tool not in ("navigate", "waterline", "person_box"):
            radius = self.brush_radius * min(rect.width(), rect.height())
            painter.setPen(QPen(QColor("#ededf0"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._hover_position, radius, radius)
        painter.restore()

    def _empty_state(self, painter: QPainter) -> None:
        center = QPointF(self.width() / 2, self.height() / 2 - 12)
        painter.setPen(QPen(QColor("#3e3e44"), 1.4))
        painter.setBrush(QColor("#1c1c1f"))
        frame = QRectF(center.x() - 28, center.y() - 88, 56, 44)
        painter.drawRoundedRect(frame, 7, 7)
        painter.setBrush(QColor("#96969e"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(center.x() - 12, center.y() - 75), 3, 3)
        landscape = QPainterPath(QPointF(center.x() - 19, center.y() - 53))
        landscape.lineTo(center.x() - 5, center.y() - 65)
        landscape.lineTo(center.x() + 3, center.y() - 58)
        landscape.lineTo(center.x() + 10, center.y() - 64)
        landscape.lineTo(center.x() + 19, center.y() - 55)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#96969e"), 1.5))
        painter.drawPath(landscape)

        font = painter.font()
        font.setPixelSize(17)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#ededf0"))
        painter.drawText(
            QRectF(20, center.y() - 20, self.width() - 40, 30),
            Qt.AlignmentFlag.AlignCenter,
            _("Добавьте фото или видео"),
        )
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#96969e"))
        painter.drawText(
            QRectF(24, center.y() + 17, self.width() - 48, 40),
            Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
            _("Перетащите файлы сюда или нажмите «Импорт»"),
        )
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor("#6e6e78"))
        painter.drawText(
            QRectF(20, center.y() + 65, self.width() - 40, 24),
            Qt.AlignmentFlag.AlignCenter,
            "JPEG, PNG, TIFF, MP4, MOV",
        )

    def _badge(self, painter: QPainter, image_rect: QRectF, text: str, left: bool) -> None:
        x = (
            max(12, image_rect.left() + 12)
            if left
            else min(self.width() - 76, image_rect.right() - 76)
        )
        area = QRectF(x, max(12, image_rect.top() + 12), 64, 25)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 24, 27, 215))
        painter.drawRoundedRect(area, 4, 4)
        painter.setPen(QColor("#ededf0"))
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.before.isNull():
            return
        rect = self.image_rect()
        boundary = rect.left() + rect.width() * self.split
        point = self.normalized_point(event.position())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.edits_enabled
            and self.tool != "navigate"
            and point is not None
        ):
            self._dragging = "paint"
            self._stroke_points = [point]
            self.update()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = "pan"
            self.fitting = False
            self._last_position = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "compare" and abs(event.position().x() - boundary) < 23:
                self._dragging = "split"
            else:
                self._dragging = "pan"
                self.fitting = False
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._last_position = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.before.isNull():
            return
        rect = self.image_rect()
        self._hover_position = event.position()
        if self._dragging == "paint":
            point = self.normalized_point(event.position())
            if point is not None:
                self._stroke_points.append(point)
            self.update()
        elif self._dragging == "split":
            self.split = float(
                np.clip((event.position().x() - rect.left()) / max(1, rect.width()), 0, 1)
            )
            self.update()
        elif self._dragging == "pan":
            self.offset += event.position() - self._last_position
            self._last_position = event.position()
            self.update()
        else:
            boundary = rect.left() + rect.width() * self.split
            near = self.mode == "compare" and abs(event.position().x() - boundary) < 23
            if self.tool != "navigate" and self.edits_enabled:
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.update()
            else:
                self.setCursor(
                    Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.OpenHandCursor
                )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging == "paint" and self._stroke_points:
            point = self.normalized_point(event.position())
            if point is not None:
                self._stroke_points.append(point)
            points = tuple(self._stroke_points)
            if self.tool == "waterline":
                self.waterline_changed.emit(points)
            elif self.tool == "person_box":
                first, last = points[0], points[-1]
                box = (
                    min(first[0], last[0]),
                    min(first[1], last[1]),
                    max(first[0], last[0]),
                    max(first[1], last[1]),
                )
                if box[2] - box[0] > 0.01 and box[3] - box[1] > 0.01:
                    self.person_box_changed.emit(box)
            else:
                target, operation = self.tool.split("_", 1)
                self.stroke_created.emit(
                    {
                        "target": target,
                        "operation": operation,
                        "points": points,
                        "radius": self.brush_radius,
                    }
                )
            self._stroke_points = []
        self._dragging = ""
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.before.isNull():
            self.open_requested.emit()
        elif self.fitting:
            self.one_to_one()
        else:
            self.fit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.before.isNull():
            return
        old = self.scale
        maximum = self.source_width / self.before.width() * 5 / self.devicePixelRatioF()
        minimum = min(0.05, self.source_width / self.before.width() * 0.05)
        self.scale = float(np.clip(old * 1.15 ** (event.angleDelta().y() / 120), minimum, maximum))
        relative = event.position() - QPointF(self.width() / 2, self.height() / 2)
        self.offset = relative - (relative - self.offset) * (self.scale / old)
        self.fitting = False
        self._emit_zoom()
        self.update()
        event.accept()
