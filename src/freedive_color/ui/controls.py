from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from freedive_color.core.presets import GradeSettings, get_preset

from .strings import _

# key, Russian source label, minimum, maximum, tooltip
CONTROL_GROUPS = [
    (
        "Свет",
        [
            (
                "exposure",
                "Экспозиция",
                -300,
                300,
                "От −3 до +3 EV. Не добавляет новых источников света.",
            ),
            ("contrast", "Контраст", -100, 100, "Мягкий контраст яркости без изменения геометрии."),
            (
                "highlights",
                "Света",
                -100,
                100,
                "Сдерживает или поднимает существующие светлые участки.",
            ),
            (
                "shadows",
                "Тени",
                -100,
                100,
                "Бережное изменение тёмных участков. Может проявлять шум.",
            ),
        ],
    ),
    (
        "Цвет",
        [
            (
                "auto_strength",
                "Автокоррекция",
                0,
                100,
                "Сила консервативного анализа цветового сдвига.",
            ),
            (
                "temperature",
                "Температура",
                -100,
                100,
                "Влево — холоднее, вправо — теплее. Глобальный баланс.",
            ),
            (
                "tint",
                "Оттенок",
                -100,
                100,
                "Влево — зеленее, вправо — пурпурнее. Глобальная коррекция.",
            ),
            (
                "red_recovery",
                "Красный канал",
                0,
                100,
                "Усиливает только существующий красный сигнал с лимитом.",
            ),
            (
                "saturation",
                "Насыщенность",
                -100,
                100,
                "Глобальная насыщенность с ограничением цветового охвата.",
            ),
        ],
    ),
    (
        "Детали",
        [
            (
                "dehaze",
                "Прозрачность",
                0,
                35,
                "Мягкий локальный контраст; не удаляет физическую взвесь.",
            ),
            ("detail", "Резкость", 0, 100, "Умеренное усиление существующих деталей."),
            ("denoise", "Шум цвета", 0, 100, "Подавляет цветной шум, сохраняя яркостную фактуру."),
        ],
    ),
]


class ResetSlider(QSlider):
    def __init__(self, default: int, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.default = default

    def mouseDoubleClickEvent(self, event) -> None:
        self.setValue(self.default)
        event.accept()


class GradeControls(QWidget):
    settings_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.preset_name = "natural"
        self._updating = False
        self.sliders: dict[str, ResetSlider] = {}
        self.values: dict[str, QLabel] = {}
        self.preset_buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 15, 16, 12)
        layout.setSpacing(11)
        title = QLabel(_("Стиль"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        styles = QHBoxLayout()
        styles.setSpacing(5)
        self.preset_group = QButtonGroup(self)
        for name in ("natural", "cinematic", "vivid"):
            button = QPushButton(name.title())
            button.setCheckable(True)
            button.setObjectName("preset_" + name)
            button.setMinimumWidth(0)
            button.setStyleSheet("padding-left: 7px; padding-right: 7px;")
            button.clicked.connect(lambda _checked=False, chosen=name: self.set_preset(chosen))
            self.preset_group.addButton(button)
            self.preset_buttons[name] = button
            styles.addWidget(button)
        layout.addLayout(styles)
        self.preset_note = QLabel()
        self.preset_note.setObjectName("subtle")
        self.preset_note.setWordWrap(True)
        layout.addWidget(self.preset_note)
        for group_name, specs in CONTROL_GROUPS:
            title = QLabel(_(group_name))
            title.setObjectName("sectionTitle")
            layout.addSpacing(7)
            layout.addWidget(title)
            for key, label, low, high, tooltip in specs:
                heading = QHBoxLayout()
                heading.setSpacing(3)
                text = QLabel(_(label))
                text.setToolTip(_(tooltip))
                heading.addWidget(text)
                value = QLabel()
                value.setObjectName("value")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                heading.addWidget(value)
                layout.addLayout(heading)
                slider = ResetSlider(round(getattr(get_preset("natural"), key) * 100))
                slider.setObjectName("control_" + key)
                slider.setRange(low, high)
                slider.setAccessibleName(_(label))
                slider.setToolTip(_(tooltip) + _(" Двойной щелчок — сброс."))
                slider.valueChanged.connect(lambda _value, k=key: self._changed(k))
                self.sliders[key], self.values[key] = slider, value
                layout.addWidget(slider)
        self.reset_button = QPushButton(_("Сбросить стиль"))
        self.reset_button.setObjectName("reset_settings")
        self.reset_button.clicked.connect(
            lambda: self.set_preset(self.preset_name if self.preset_name != "custom" else "natural")
        )
        layout.addWidget(self.reset_button)
        self.set_settings(get_preset("natural"), emit=False)

    def settings(self) -> GradeSettings:
        return GradeSettings(
            preset=self.preset_name,
            **{key: slider.value() / 100 for key, slider in self.sliders.items()},
        )

    def set_settings(self, settings: GradeSettings, emit: bool = True) -> None:
        self._updating = True
        self.preset_name = settings.preset
        self.preset_group.setExclusive(False)
        for name, button in self.preset_buttons.items():
            button.setChecked(name == settings.preset)
        self.preset_group.setExclusive(True)
        for key, slider in self.sliders.items():
            slider.setValue(round(getattr(settings, key) * 100))
            slider.default = round(
                getattr(
                    get_preset(settings.preset if settings.preset != "custom" else "natural"), key
                )
                * 100
            )
            self._display_value(key)
        self._updating = False
        self._update_note()
        if emit:
            self.settings_changed.emit(self.settings())

    def set_preset(self, name: str) -> None:
        self.set_settings(get_preset(name))

    def _display_value(self, key: str) -> None:
        value = self.sliders[key].value()
        self.values[key].setText(f"{value / 100:+.2f} EV" if key == "exposure" else str(value))

    def _update_note(self) -> None:
        base = get_preset(self.preset_name if self.preset_name != "custom" else "natural")
        current = self.settings()
        # Controls have a 0.01 resolution, so compare against the displayed base values.
        changed = any(
            round(getattr(base, key) * 100) != slider.value()
            for key, slider in self.sliders.items()
        )
        text = {
            "natural": "Естественный цвет, умеренный контраст.",
            "cinematic": "Мягкие света и тональная глубина.",
            "vivid": "Выразительный цвет с умеренной насыщенностью.",
            "custom": "Загруженные настройки.",
        }[current.preset]
        self.preset_note.setText(_(text) + (_(" Изменён вручную.") if changed else ""))

    def _changed(self, key: str) -> None:
        self._display_value(key)
        if not self._updating:
            self._update_note()
            self.settings_changed.emit(self.settings())
