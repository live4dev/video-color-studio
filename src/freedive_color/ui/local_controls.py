"""Controls for explicit local masks; tools change recipes, never the source geometry."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from freedive_color.core.local_pipeline import LocalSettings

from .strings import _


class LocalControls(QWidget):
    settings_changed = Signal(object)
    overlay_changed = Signal(str)
    tool_changed = Signal(str)
    radius_changed = Signal(float)
    undo_requested = Signal()
    reset_requested = Signal()
    reference_requested = Signal()
    clear_reference_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._settings = LocalSettings()
        self.sliders = {}
        self.value_labels = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 15, 16, 12)
        layout.setSpacing(10)
        self.enabled_check = QCheckBox(_("Раздельная обработка областей"))
        self.enabled_check.setChecked(True)
        self.enabled_check.toggled.connect(self._changed)
        layout.addWidget(self.enabled_check)
        overlay = QHBoxLayout()
        overlay.addWidget(QLabel(_("Показать маску")))
        self.overlay_combo = QComboBox()
        for title, key in (
            ("Нет", "none"),
            ("Кожа", "skin"),
            ("Вода", "water"),
            ("Человек", "person"),
            ("Под водой", "underwater"),
        ):
            self.overlay_combo.addItem(_(title), key)
        self.overlay_combo.currentIndexChanged.connect(
            lambda: self.overlay_changed.emit(self.overlay_combo.currentData())
        )
        overlay.addWidget(self.overlay_combo)
        layout.addLayout(overlay)
        self.tool_combo = QComboBox()
        for title, key in (
            ("Просмотр / панорамирование", "navigate"),
            ("Кожа: добавить", "skin_add"),
            ("Кожа: вычесть", "skin_subtract"),
            ("Вода: добавить", "water_add"),
            ("Вода: вычесть", "water_subtract"),
            ("Область человека: рамка", "person_box"),
            ("Граница вода / воздух", "waterline"),
        ):
            self.tool_combo.addItem(_(title), key)
        self.tool_combo.currentIndexChanged.connect(
            lambda: self.tool_changed.emit(self.tool_combo.currentData())
        )
        layout.addWidget(self.tool_combo)
        radius = QHBoxLayout()
        radius.addWidget(QLabel(_("Радиус кисти, % кадра")))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 15)
        self.radius_spin.setValue(2)
        self.radius_spin.valueChanged.connect(lambda value: self.radius_changed.emit(value / 100))
        radius.addWidget(self.radius_spin)
        layout.addLayout(radius)
        actions = QHBoxLayout()
        undo = QPushButton(_("Убрать штрих"))
        undo.clicked.connect(self.undo_requested)
        clear = QPushButton(_("Сбросить маски"))
        clear.clicked.connect(self.reset_requested)
        actions.addWidget(undo)
        actions.addWidget(clear)
        layout.addLayout(actions)
        note = QLabel(
            _(
                "Кисть — по снимку. Рамкой выделите человека. "
                "Границу воды проведите слева направо. "
                "Средняя кнопка мыши перемещает изображение."
            )
        )
        note.setObjectName("metadata")
        note.setWordWrap(True)
        layout.addWidget(note)
        for title, specs in (
            (
                "Кожа",
                (
                    ("skin_warmth", "Теплота", -100, 100, 100),
                    ("skin_strength", "Сила коррекции", 0, 100, 100),
                ),
            ),
            (
                "Вода",
                (
                    ("water_hue", "Оттенок воды", -100, 100, 100),
                    ("water_saturation", "Насыщенность воды", -100, 100, 100),
                    ("water_strength", "Сила коррекции воды", 0, 100, 100),
                ),
            ),
            (
                "Граница и воздух",
                (
                    ("above_exposure", "Экспозиция над водой", -300, 300, 100),
                    ("feather", "Растушёвка", 1, 500, 10000),
                    ("reference_strength", "Сила эталона", 0, 100, 100),
                ),
            ),
        ):
            heading = QLabel(_(title))
            heading.setObjectName("sectionTitle")
            layout.addWidget(heading)
            for key, caption, low, high, factor in specs:
                row = QHBoxLayout()
                row.addWidget(QLabel(_(caption)))
                value = QLabel()
                value.setObjectName("value")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                row.addWidget(value)
                layout.addLayout(row)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setObjectName("local_" + key)
                slider.setRange(low, high)
                slider.valueChanged.connect(lambda _v, chosen=key: self._changed(chosen))
                layout.addWidget(slider)
                self.sliders[key] = (slider, factor)
                self.value_labels[key] = value
        reference = QHBoxLayout()
        choose = QPushButton(_("Выбрать эталон…"))
        choose.clicked.connect(self.reference_requested)
        remove = QPushButton(_("Без эталона"))
        remove.clicked.connect(self.clear_reference_requested)
        reference.addWidget(choose)
        reference.addWidget(remove)
        layout.addLayout(reference)
        self.reference_label = QLabel(_("Эталон не выбран"))
        self.reference_label.setObjectName("metadata")
        self.reference_label.setWordWrap(True)
        layout.addWidget(self.reference_label)
        self.confidence_label = QLabel()
        self.confidence_label.setObjectName("warning")
        self.confidence_label.setWordWrap(True)
        layout.addWidget(self.confidence_label)
        self.set_settings(self._settings, emit=False)

    def settings(self) -> LocalSettings:
        return self._settings

    def set_settings(self, settings: LocalSettings, emit=True):
        self._updating = True
        self._settings = settings
        self.enabled_check.setChecked(settings.enabled)
        for key, (slider, factor) in self.sliders.items():
            slider.setValue(round(getattr(settings, key) * factor))
            self.value_labels[key].setText(f"{getattr(settings, key):.3g}")
        self._updating = False
        if emit:
            self.settings_changed.emit(settings)

    def _changed(self, key=None):
        if self._updating:
            return
        values = self._settings.to_dict()
        values["enabled"] = self.enabled_check.isChecked()
        if isinstance(key, str):
            slider, factor = self.sliders[key]
            values[key] = slider.value() / factor
        self._settings = LocalSettings.from_dict(values)
        for field in self.sliders:
            self.value_labels[field].setText(f"{getattr(self._settings, field):.3g}")
        self.settings_changed.emit(self._settings)
