"""Inspector tabs keep navigation visible while each page scrolls independently."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QTabWidget, QWidget


class InspectorTabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("inspectorTabs")
        self.setDocumentMode(True)
        self.tabBar().setExpanding(True)

    def add_page(self, widget: QWidget, title: str) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(widget)
        self.addTab(scroll, title)

    def setCurrentWidget(self, widget: QWidget) -> None:
        # Existing callers select GradeControls/LocalControls directly.
        for index in range(self.count()):
            page = self.widget(index)
            if page is widget or (isinstance(page, QScrollArea) and page.widget() is widget):
                self.setCurrentIndex(index)
                return
        super().setCurrentWidget(widget)
