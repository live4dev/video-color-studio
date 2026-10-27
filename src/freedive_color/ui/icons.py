"""Small, original outline icons, rendered locally at high DPI."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def icon(name: str, color: str = "#ededf0", size: int = 18) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 24, size / 24)
    painter.setPen(QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if name == "plus":
        painter.drawLine(QPointF(12, 5), QPointF(12, 19))
        painter.drawLine(QPointF(5, 12), QPointF(19, 12))
    elif name == "folder":
        path = QPainterPath(QPointF(3, 7))
        for x, y in ((3, 5), (9, 5), (12, 8), (21, 8), (21, 19), (3, 19), (3, 7)):
            path.lineTo(x, y)
        painter.drawPath(path)
    elif name == "export":
        painter.drawLine(QPointF(12, 3), QPointF(12, 15))
        painter.drawLine(QPointF(8, 7), QPointF(12, 3))
        painter.drawLine(QPointF(16, 7), QPointF(12, 3))
        path = QPainterPath(QPointF(4, 13))
        path.lineTo(4, 20)
        path.lineTo(20, 20)
        path.lineTo(20, 13)
        painter.drawPath(path)
    elif name == "save":
        painter.drawRoundedRect(QRectF(4, 3, 16, 18), 2, 2)
        painter.drawRect(QRectF(8, 3, 8, 6))
        painter.drawRect(QRectF(8, 14, 8, 7))
    elif name == "fit":
        for x, y, dx, dy in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            painter.drawLine(QPointF(x, y + dy * 5), QPointF(x, y))
            painter.drawLine(QPointF(x, y), QPointF(x + dx * 5, y))
    elif name == "layers":
        for offset in (0, 5):
            path = QPainterPath(QPointF(3, 8 + offset))
            path.lineTo(12, 3 + offset)
            path.lineTo(21, 8 + offset)
            path.lineTo(12, 13 + offset)
            path.closeSubpath()
            painter.drawPath(path)
    elif name == "logo":
        painter.setPen(QPen(QColor(color), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for y in (7, 13, 19):
            path = QPainterPath(QPointF(3, y))
            path.cubicTo(8, y - 6, 15, y + 6, 21, y - 2)
            painter.drawPath(path)
    else:
        raise ValueError(f"Unknown icon: {name}")
    painter.end()
    return QIcon(pixmap)
