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
        self._exact_values = {
            key: getattr(get_preset("natural"), key)
            for _group, specs in CONTROL_GROUPS
            for key, *_rest in specs
        }
        self.sliders: dict[str, ResetSlider] = {}
        self.values: dict[str, QLabel] = {}
        self.preset_buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        title = QLabel(_("Стиль"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        styles = QHBoxLayout()
        styles.setSpacing(6)
        self.preset_group = QButtonGroup(self)
        for name in ("natural", "cinematic", "vivid"):
            button = QPushButton(name.title())
            button.setCheckable(True)
            button.setObjectName("preset_" + name)
            button.setProperty("variant", "preset")
            button.setMinimumWidth(0)
            button.setAccessibleName(_("Стиль ") + name.title())
            button.setToolTip(
                {
                    "natural": _("Естественный цвет, умеренный контраст"),
                    "cinematic": _("Мягкие света и тональная глубина"),
                    "vivid": _("Выразительный цвет с умеренной насыщенностью"),
                }[name]
            )
            button.clicked.connect(lambda _checked=False, chosen=name: self.set_preset(chosen))
            self.preset_group.addButton(button)
            self.preset_buttons[name] = button
            styles.addWidget(button)
        layout.addLayout(styles)
        self.preset_note = QLabel()
        self.preset_note.setObjectName("subtle")
        self.preset_note.setWordWrap(True)
        self.preset_note.setMinimumHeight(30)
        layout.addWidget(self.preset_note)
        for group_name, specs in CONTROL_GROUPS:
            title = QLabel(_(group_name))
            title.setObjectName("sectionTitle")
            title.setProperty("separated", True)
            layout.addSpacing(4)
            layout.addWidget(title)
            for key, label, low, high, tooltip in specs:
                control = QVBoxLayout()
                control.setSpacing(2)
                heading = QHBoxLayout()
                heading.setSpacing(3)
                text = QLabel(_(label))
                text.setToolTip(_(tooltip))
                heading.addWidget(text)
                value = QLabel()
                value.setObjectName("value")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                heading.addWidget(value)
                control.addLayout(heading)
                slider = ResetSlider(round(getattr(get_preset("natural"), key) * 100))
                slider.setObjectName("control_" + key)
                slider.setRange(low, high)
                slider.setAccessibleName(_(label))
                slider.setToolTip(_(tooltip) + _(" Двойной щелчок — сброс."))
                slider.valueChanged.connect(lambda _value, k=key: self._changed(k))
                self.sliders[key], self.values[key] = slider, value
                control.addWidget(slider)
                layout.addLayout(control)
        self.reset_button = QPushButton(_("Сбросить стиль"))
        self.reset_button.setObjectName("reset_settings")
        self.reset_button.setProperty("variant", "ghost")
        self.reset_button.clicked.connect(
            lambda: self.set_preset(self.preset_name if self.preset_name != "custom" else "natural")
        )
        layout.addWidget(self.reset_button)
        self.set_settings(get_preset("natural"), emit=False)

    def settings(self) -> GradeSettings:
        return GradeSettings(
            preset=self.preset_name,
            **self._exact_values,
        )

    def set_settings(self, settings: GradeSettings, emit: bool = True) -> None:
        self._updating = True
        self._exact_values = {key: getattr(settings, key) for key in self.sliders}
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
        value = self._exact_values[key] * 100
        self.values[key].setText(f"{value / 100:+.3g} EV" if key == "exposure" else f"{value:g}")

    def _update_note(self) -> None:
        base = get_preset(self.preset_name if self.preset_name != "custom" else "natural")
        current = self.settings()
        changed = any(
            abs(getattr(base, key) - self._exact_values[key]) > 1e-9 for key in self.sliders
        )
        text = {
            "natural": "Естественный цвет, умеренный контраст.",
            "cinematic": "Мягкие света и тональная глубина.",
            "vivid": "Выразительный цвет с умеренной насыщенностью.",
            "custom": "Загруженные настройки.",
        }[current.preset]
        self.preset_note.setText(_(text) + (_(" Изменён вручную.") if changed else ""))

    def _changed(self, key: str) -> None:
        if not self._updating:
            self._exact_values[key] = self.sliders[key].value() / 100
        self._display_value(key)
        if not self._updating:
            self._update_note()
            self.settings_changed.emit(self.settings())
