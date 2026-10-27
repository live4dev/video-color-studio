import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from freedive_color import __version__
from freedive_color.core.analyze import SceneAnalysis, analyze_scene
from freedive_color.core.batch import export_batch
from freedive_color.core.errors import StudioError
from freedive_color.core.image_pipeline import (
    Preview,
    create_image_reference,
    export_photo,
    render_rgb,
)
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import BrushStroke, MaskEdits
from freedive_color.core.media_io import PHOTO_EXTENSIONS, Photo, load_photo, signature
from freedive_color.core.presets import GradeSettings, get_preset
from freedive_color.core.project import load_settings, read_json, save_settings, sha256_file
from freedive_color.core.reference import create_reference
from freedive_color.core.session import (
    ExportOptions,
    SessionItem,
)

from .controller_bridge import ControllerWindowMixin
from .controls import GradeControls
from .icons import icon
from .inspector import InspectorTabs
from .library import MediaLibrary
from .local_controls import LocalControls
from .strings import _
from .video_playback import VideoPlayback
from .viewer import CompareViewer, to_qimage

PATH_ROLE = Qt.ItemDataRole.UserRole
SETTINGS_ROLE = Qt.ItemDataRole.UserRole + 1
LOCAL_ROLE = Qt.ItemDataRole.UserRole + 2
MASK_ROLE = Qt.ItemDataRole.UserRole + 3
ANALYSIS_ROLE = Qt.ItemDataRole.UserRole + 4
INFO_ROLE = Qt.ItemDataRole.UserRole + 5
OPEN_FILTER = "Фото SDR (*.jpg *.jpeg *.png *.tif *.tiff)"
VIDEO_EXTENSIONS = {".mp4", ".mov"}
MEDIA_FILTER = "Фото и видео SDR (*.jpg *.jpeg *.png *.tif *.tiff *.mp4 *.mov)"


class MainWindow(ControllerWindowMixin, QMainWindow):
    preview_ready = Signal()
    export_completed = Signal(object)
    batch_completed = Signal(object)
    video_preview_completed = Signal(object)
    video_export_completed = Signal(object)
    operation_failed = Signal(str)

    def __init__(self, controller=None, integration_mode: bool = False):
        super().__init__()
        if controller is None:
            from freedive_color.control.controller import ProjectController

            controller = ProjectController(parent=self)
        self.controller = controller
        self.integration_mode = integration_mode
        self._explicit_quit = False
        self.setWindowTitle("Freedive Color Studio")
        self.resize(1440, 920)
        self.setMinimumSize(1080, 720)
        self.setAcceptDrops(True)
        self.pool = controller.pool
        self.preview: Preview | None = None
        self.rendered_settings: GradeSettings | None = None
        self.last_export: dict | None = None
        self.mask_edits = MaskEdits()
        self.reference = None
        self.reference_path = None
        self.session_path = None
        self.temp_directory = None
        self.output_directory = None
        self.last_batch = None
        self.video_info = None
        self.video_frame_seconds = 0.0
        self._video_seek_position = 0.0
        self.video_preview_path = None
        self.video_preview_start = 0.0
        self.video_preview_duration = 0.0
        self._video_playing = False
        self.video_playback = None
        self._video_plan = None
        self._proxy_directories = []
        self._photo_export_suffix = ".jpg"
        self._jobs: dict[int, tuple[object, str, object]] = {}
        self._counter = 0
        self._view_job: int | None = None
        self._export_job: int | None = None
        self._render_pending = False
        self._full_loading = False
        self._closing = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(140)
        self._debounce.timeout.connect(self._render)
        self._build_ui()
        self._build_menu()
        self._attach_controller()
        self._update_actions()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("workspace")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 16, 10)
        header_layout.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(icon("logo", "#63e4d4", 25).pixmap(25, 25))
        header_layout.addWidget(mark)
        brand = QLabel("Freedive Color Studio")
        brand.setObjectName("brand")
        header_layout.addWidget(brand)
        header_layout.addStretch()
        local_badge = QLabel(_("●  Локально"))
        local_badge.setObjectName("localBadge")
        local_badge.setToolTip(_("Фото и видео обрабатываются на этом компьютере."))
        header_layout.addWidget(local_badge)
        project_button = QPushButton(_("Сохранить проект"))
        project_button.setIcon(icon("save"))
        project_button.setProperty("variant", "ghost")
        project_button.clicked.connect(lambda: self.save_project_dialog())
        header_layout.addWidget(project_button)
        self.batch_button = QPushButton(_("Экспорт серии"))
        self.batch_button.setIcon(icon("layers"))
        self.batch_button.clicked.connect(self.batch_export_dialog)
        header_layout.addWidget(self.batch_button)
        self.export_button = QPushButton(_("Экспорт фото"))
        self.export_button.setObjectName("primary")
        self.export_button.setIcon(icon("export", "#112422"))
        self.export_button.clicked.connect(self.export_dialog)
        header_layout.addWidget(self.export_button)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        layout.addWidget(splitter, 1)

        library_panel = QWidget()
        library_panel.setObjectName("libraryPanel")
        library_panel.setMinimumWidth(200)
        library_panel.setMaximumWidth(340)
        library_layout = QVBoxLayout(library_panel)
        library_layout.setContentsMargins(10, 12, 10, 12)
        library_layout.setSpacing(10)
        library_heading = QHBoxLayout()
        title = QLabel(_("Медиа"))
        title.setObjectName("panelTitle")
        library_heading.addWidget(title)
        library_heading.addStretch()
        self.library_count = QLabel(_("Нет файлов"))
        self.library_count.setObjectName("subtle")
        library_heading.addWidget(self.library_count)
        library_layout.addLayout(library_heading)
        import_row = QHBoxLayout()
        import_row.setSpacing(6)
        self.import_button = QPushButton(_("Импорт"))
        self.import_button.setObjectName("import_button")
        self.import_button.setIcon(icon("plus"))
        self.import_button.setToolTip(_("Открыть фото и видео · ⌘O / Ctrl+O"))
        self.import_button.clicked.connect(self.open_dialog)
        import_row.addWidget(self.import_button, 1)
        self.folder_button = QPushButton()
        self.folder_button.setIcon(icon("folder"))
        self.folder_button.setFixedWidth(34)
        self.folder_button.setAccessibleName(_("Открыть папку"))
        self.folder_button.setToolTip(_("Импортировать папку"))
        self.folder_button.clicked.connect(self.open_folder)
        import_row.addWidget(self.folder_button)
        library_layout.addLayout(import_row)
        self.library = MediaLibrary()
        self.library.currentItemChanged.connect(self._selection_changed)
        self.library.itemDoubleClicked.connect(lambda _item: self._load_selected())
        library_layout.addWidget(self.library, 1)
        self.apply_group_button = QPushButton(_("Применить к выбранным"))
        self.apply_group_button.setToolTip(_("Перенести настройки на выбранные файлы серии."))
        self.apply_group_button.clicked.connect(self.apply_to_selected)
        library_layout.addWidget(self.apply_group_button)
        note = QLabel(_("Отметки на карточках включают файлы в экспорт серии."))
        note.setObjectName("metadata")
        note.setWordWrap(True)
        library_layout.addWidget(note)
        splitter.addWidget(library_panel)

        center = QWidget()
        center.setObjectName("viewerPanel")
        center.setMinimumWidth(380)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        toolbar = QWidget()
        toolbar.setObjectName("viewerToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 8, 14, 8)
        toolbar_layout.setSpacing(3)
        viewer_title = QLabel(_("Просмотр"))
        viewer_title.setObjectName("panelTitle")
        toolbar_layout.addWidget(viewer_title)
        toolbar_layout.addStretch()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode, label in (("before", "До"), ("compare", "Сравнение"), ("after", "После")):
            button = QPushButton(_(label))
            button.setCheckable(True)
            button.setObjectName("view_" + mode)
            button.setProperty("variant", "segment")
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            button.clicked.connect(lambda _checked=False, chosen=mode: self.set_view_mode(chosen))
            toolbar_layout.addWidget(button)
        self.mode_buttons["compare"].setChecked(True)
        center_layout.addWidget(toolbar)
        self.viewer = CompareViewer()
        self.viewer.open_requested.connect(self.open_dialog)
        self.viewer.zoom_changed.connect(self._zoom_changed)
        self.viewer.stroke_created.connect(self._add_stroke)
        self.viewer.waterline_changed.connect(self._set_waterline)
        self.viewer.person_box_changed.connect(self._set_person_box)
        center_layout.addWidget(self.viewer, 1)

        viewer_footer = QWidget()
        viewer_footer.setObjectName("viewerFooter")
        info_layout = QVBoxLayout(viewer_footer)
        info_layout.setContentsMargins(14, 10, 14, 10)
        info_layout.setSpacing(4)
        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        self.image_title = QLabel(_("Фото и видео"))
        self.image_title.setMinimumWidth(0)
        self.image_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info_row.addWidget(self.image_title, 1)
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("subtle")
        info_row.addWidget(self.zoom_label)
        self.fit_button = QPushButton(_("Вписать"))
        self.fit_button.setIcon(icon("fit", "#96969e", 14))
        self.fit_button.setProperty("variant", "ghost")
        self.fit_button.setToolTip(_("Вписать в окно · F"))
        self.fit_button.clicked.connect(self.viewer.fit)
        info_row.addWidget(self.fit_button)
        self.one_button = QPushButton("100%")
        self.one_button.setProperty("variant", "ghost")
        self.one_button.setToolTip(_("Полное разрешение, 1 пиксель фото = 1 пиксель просмотра"))
        self.one_button.clicked.connect(self._one_to_one)
        info_row.addWidget(self.one_button)
        info_layout.addLayout(info_row)
        self.metadata_label = QLabel(_("Импортируйте файлы и выберите стиль коррекции."))
        self.metadata_label.setObjectName("metadata")
        self.metadata_label.setWordWrap(True)
        info_layout.addWidget(self.metadata_label)
        self.warning_label = QLabel()
        self.warning_label.setObjectName("warning")
        self.warning_label.setWordWrap(True)
        info_layout.addWidget(self.warning_label)
        center_layout.addWidget(viewer_footer)

        self.video_panel = QWidget()
        self.video_panel.setObjectName("videoPanel")
        video_layout = QVBoxLayout(self.video_panel)
        video_layout.setContentsMargins(14, 10, 14, 10)
        video_layout.setSpacing(8)
        transport = QHBoxLayout()
        self.video_play_button = QPushButton(_("Воспроизвести"))
        self.video_play_button.clicked.connect(self.toggle_video_playback)
        transport.addWidget(self.video_play_button)
        self.video_time_label = QLabel("0.00 / 0.00 с")
        self.video_time_label.setObjectName("subtle")
        transport.addWidget(self.video_time_label)
        transport.addStretch()
        self.video_sound_button = QPushButton(_("Открыть со звуком"))
        self.video_sound_button.setProperty("variant", "ghost")
        self.video_sound_button.clicked.connect(self.open_video_preview_external)
        transport.addWidget(self.video_sound_button)
        video_layout.addLayout(transport)
        self.video_timeline = QSlider(Qt.Orientation.Horizontal)
        self.video_timeline.setObjectName("video_timeline")
        self.video_timeline.setAccessibleName(_("Позиция в видео"))
        self.video_timeline.setRange(0, 0)
        self.video_timeline.valueChanged.connect(lambda value: self.seek_video(value / 1000))
        video_layout.addWidget(self.video_timeline)
        preview_row = QHBoxLayout()
        preview_row.setSpacing(6)
        preview_row.addWidget(QLabel(_("Начало")))
        self.video_start_spin = QDoubleSpinBox()
        self.video_start_spin.setRange(0, 1)
        self.video_start_spin.setDecimals(2)
        self.video_start_spin.setSuffix(" с")
        self.video_start_spin.setAccessibleName(_("Начало превью, секунды"))
        preview_row.addWidget(self.video_start_spin)
        preview_row.addWidget(QLabel(_("Длина")))
        self.video_duration_spin = QDoubleSpinBox()
        self.video_duration_spin.setRange(0.1, 15)
        self.video_duration_spin.setValue(10)
        self.video_start_spin.valueChanged.connect(self._preview_range_changed)
        self.video_duration_spin.setSuffix(" с")
        self.video_duration_spin.setAccessibleName(_("Длина превью, секунды"))
        preview_row.addWidget(self.video_duration_spin)
        preview_row.addStretch()
        self.video_preview_button = QPushButton(_("Создать превью"))
        self.video_preview_button.clicked.connect(
            lambda: self.create_video_preview(
                self.video_start_spin.value(), self.video_duration_spin.value()
            )
        )
        preview_row.addWidget(self.video_preview_button)
        video_layout.addLayout(preview_row)
        self.video_sound_note = QLabel(_("В окне — сравнение без звука."))
        self.video_sound_note.setObjectName("metadata")
        self.video_sound_note.setWordWrap(True)
        video_layout.addWidget(self.video_sound_note)
        mask_note = QLabel(
            _("Автомаски следуют за движением. Ручные мазки и линия воды закреплены в кадре.")
        )
        mask_note.setObjectName("metadata")
        mask_note.setWordWrap(True)
        video_layout.addWidget(mask_note)
        center_layout.addWidget(self.video_panel)
        self.video_panel.hide()
        splitter.addWidget(center)

        self.right_panel = QWidget()
        self.right_panel.setObjectName("inspectorPanel")
        self.right_panel.setMinimumWidth(300)
        self.right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        inspector_heading = QLabel(_("Параметры"))
        inspector_heading.setObjectName("panelHeader")
        right_layout.addWidget(inspector_heading)
        self.settings_tabs = InspectorTabs()
        self.controls = GradeControls()
        self.controls.settings_changed.connect(self._settings_changed)
        presets_row = QHBoxLayout()
        presets_row.setSpacing(6)
        save_button = QPushButton(_("Сохранить стиль…"))
        save_button.setToolTip(_("Сохранить параметры в локальный JSON"))
        save_button.clicked.connect(self.save_settings_dialog)
        load_button = QPushButton(_("Загрузить…"))
        load_button.setToolTip(_("Загрузить JSON пресета или настройки из sidecar"))
        load_button.clicked.connect(self.load_settings_dialog)
        presets_row.addWidget(save_button)
        presets_row.addWidget(load_button)
        self.controls.layout().addLayout(presets_row)
        self.settings_tabs.add_page(self.controls, _("Коррекция"))
        self.local_controls = LocalControls()
        self.local_controls.settings_changed.connect(self._local_settings_changed)
        self.local_controls.overlay_changed.connect(self._update_mask_overlay)
        self.local_controls.tool_changed.connect(self.viewer.set_tool)
        self.local_controls.radius_changed.connect(
            lambda value: setattr(self.viewer, "brush_radius", value)
        )
        self.local_controls.undo_requested.connect(self.undo_mask_edit)
        self.local_controls.reset_requested.connect(self.reset_mask_edits)
        self.local_controls.reference_requested.connect(self.reference_dialog)
        self.local_controls.frame_reference_requested.connect(self.use_frame_reference)
        self.local_controls.clear_reference_requested.connect(self.clear_reference)
        self.settings_tabs.add_page(self.local_controls, _("Маски"))
        self.export_panel = QWidget()
        export_layout = QVBoxLayout(self.export_panel)
        export_layout.setContentsMargins(16, 18, 16, 12)
        export_layout.setSpacing(14)
        export_title = QLabel(_("Параметры экспорта"))
        export_title.setObjectName("sectionTitle")
        export_layout.addWidget(export_title)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        self.format_combo = QComboBox()
        for label, suffix in (("JPEG", ".jpg"), ("PNG", ".png"), ("TIFF", ".tiff")):
            self.format_combo.addItem(label, suffix)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        form.addRow(_("Формат"), self.format_combo)
        self.depth_combo = QComboBox()
        self.depth_combo.addItem(_("8 бит"), 8)
        self.depth_combo.addItem(_("16 бит"), 16)
        self.depth_combo.setEnabled(False)
        self.depth_combo.currentIndexChanged.connect(self._depth_changed)
        form.addRow(_("Глубина"), self.depth_combo)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        form.addRow(_("Качество JPEG"), self.quality_spin)
        export_layout.addLayout(form)
        self.gps_check = QCheckBox(_("Сохранить GPS, если есть"))
        self.gps_check.setToolTip(
            _(
                "По умолчанию координаты исключены из результата.\n"
                "В TIFF 16 бит GPS/EXIF не переносятся."
            )
        )
        export_layout.addWidget(self.gps_check)
        export_note = QLabel(_("Исходный размер · sRGB\nПараметры сохраняются рядом в JSON."))
        export_note.setWordWrap(True)
        export_note.setObjectName("metadata")
        export_layout.addWidget(export_note)
        export_layout.addStretch()
        self.settings_tabs.add_page(self.export_panel, _("Экспорт"))
        right_layout.addWidget(self.settings_tabs, 1)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([244, 858, 328])

        footer = QWidget()
        footer.setObjectName("footer")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(16, 7, 16, 7)
        footer_layout.setSpacing(5)
        status_row = QHBoxLayout()
        self.status_label = QLabel(_("Готово к импорту"))
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.cancel_button = QPushButton(_("Отменить"))
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.hide()
        status_row.addWidget(self.cancel_button)
        footer_layout.addLayout(status_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        footer_layout.addWidget(self.progress_bar)
        layout.addWidget(footer)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu(_("Файл"))
        open_action = QAction(_("Импорт фото и видео…"), self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)
        reimport_action = QAction(_("Повторно импортировать выбранный файл"), self)
        reimport_action.triggered.connect(self.reimport_selected)
        file_menu.addAction(reimport_action)
        project_open = QAction(_("Открыть проект…"), self)
        project_open.triggered.connect(self.open_project_dialog)
        file_menu.addAction(project_open)
        project_save = QAction(_("Сохранить проект"), self)
        project_save.setShortcut(QKeySequence.StandardKey.Save)
        project_save.triggered.connect(lambda: self.save_project_dialog())
        file_menu.addAction(project_save)
        project_save_as = QAction(_("Сохранить проект как…"), self)
        project_save_as.triggered.connect(lambda: self.save_project_dialog(save_as=True))
        file_menu.addAction(project_save_as)
        temp_folder = QAction(_("Папка временных файлов…"), self)
        temp_folder.triggered.connect(self.choose_temp_directory)
        file_menu.addAction(temp_folder)
        self.export_action = QAction(_("Экспорт выбранного файла…"), self)
        self.export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.export_action.triggered.connect(self.export_dialog)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        close = QAction(_("Закрыть"), self)
        close.setShortcut(QKeySequence.StandardKey.Close)
        close.triggered.connect(self.close)
        file_menu.addAction(close)
        view_menu = self.menuBar().addMenu(_("Просмотр"))
        for text, key, callback in (
            ("Вписать", "F", self.viewer.fit),
            ("100%", "Ctrl+1", self._one_to_one),
            ("До", "Alt+1", lambda: self.set_view_mode("before")),
            ("Сравнение", "Alt+2", lambda: self.set_view_mode("compare")),
            ("После", "Alt+3", lambda: self.set_view_mode("after")),
        ):
            action = QAction(_(text), self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            view_menu.addAction(action)
        help_menu = self.menuBar().addMenu(_("Справка"))
        about = help_menu.addAction(_("О программе и лицензиях…"))
        about.triggered.connect(self.show_about)

    def show_about(self) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(_("О программе"))
        dialog.setText(f"Freedive Color Studio {__version__}")
        dialog.setInformativeText(
            _(
                "© 2026 Freedive Color Studio contributors.\n\n"
                "Код приложения — MIT. Автономная сборка включает FFmpeg, x264/x265 "
                "и Qt с условиями GPL/LGPL. Сборка в целом распространяется на условиях GPLv3.\n\n"
                "Программа предоставляется без гарантий. Её можно изучать, изменять и "
                "распространять с соблюдением приложенных лицензий. Исходники и рецепты "
                "сборки доступны в архиве источников рядом с дистрибутивом.\n\n"
                "Обработка медиа выполняется локально."
            )
        )
        licenses = dialog.addButton(_("Открыть лицензии"), QMessageBox.ButtonRole.ActionRole)
        dialog.addButton(QMessageBox.StandardButton.Close)
        dialog.exec()
        if dialog.clickedButton() is licenses:
            root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
            path = root / "licenses"
            if not path.exists():
                path = root / "THIRD_PARTY_LICENSES.md"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _update_actions(self) -> None:
        exporting = self._export_job is not None
        playing = self._video_playing
        ready = (
            self.preview is not None and not self._render_pending and not exporting and not playing
        )
        self.export_button.setEnabled(ready)
        self.export_button.setText(_("Экспорт видео") if self.video_info else _("Экспорт фото"))
        self.export_action.setEnabled(ready)
        self.import_button.setEnabled(not exporting)
        self.folder_button.setEnabled(not exporting)
        self.library.setEnabled(not exporting)
        self.right_panel.setEnabled(not exporting and not playing)
        self.controls.setEnabled(self.preview is not None and not exporting)
        self.local_controls.setEnabled(self.preview is not None and not exporting)
        self.viewer.edits_enabled = self.preview is not None and not exporting and not playing
        self.batch_button.setEnabled(
            self.library.count() > 0 and not exporting and not self._render_pending
        )
        self.apply_group_button.setEnabled(self.preview is not None and not exporting)
        self.fit_button.setEnabled(self.preview is not None)
        self.one_button.setEnabled(self.preview is not None and not self._full_loading)
        self.video_play_button.setEnabled(
            self.video_info is not None and not exporting and not self._render_pending
        )
        self.video_preview_button.setEnabled(self.video_info is not None and ready)
        self.video_sound_button.setEnabled(self.video_preview_path is not None and not exporting)
        self.video_timeline.setEnabled(
            self.video_info is not None and not exporting and not playing
        )
        self.video_start_spin.setEnabled(not exporting and not playing)
        self.video_duration_spin.setEnabled(not exporting and not playing)

    def _status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #edc79a;" if error else "")

    def open_dialog(self) -> None:
        if self._export_job is not None:
            return
        paths, _selected = QFileDialog.getOpenFileNames(
            self, _("Импорт фото и видео"), "", MEDIA_FILTER
        )
        if paths:
            self.import_paths([Path(path) for path in paths])

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, _("Папка с фото и видео"))
        if path:
            self.import_paths([Path(path)])

    def import_paths(self, paths: list[Path]) -> None:
        if self._export_job is not None:
            return
        candidates = []
        unsupported = []
        for path in paths:
            if path.is_dir():
                try:
                    candidates.extend(
                        sorted(
                            (
                                child
                                for child in path.iterdir()
                                if child.is_file()
                                and child.suffix.lower() in PHOTO_EXTENSIONS | VIDEO_EXTENSIONS
                            ),
                            key=lambda child: child.name.casefold(),
                        )[:500]
                    )
                except OSError:
                    unsupported.append(path.name)
            elif path.suffix.lower() in PHOTO_EXTENSIONS | VIDEO_EXTENSIONS:
                candidates.append(path)
            else:
                unsupported.append(path.name)
        existing = {self.library.item(row).data(PATH_ROLE) for row in range(self.library.count())}
        added = []
        for path in candidates:
            resolved = str(path.expanduser().resolve())
            if resolved in existing:
                continue
            item = QListWidgetItem(path.name)
            item.setToolTip(resolved)
            item.setData(PATH_ROLE, resolved)
            item.setData(SETTINGS_ROLE, get_preset("natural").to_dict())
            item.setData(LOCAL_ROLE, LocalSettings().to_dict())
            item.setData(MASK_ROLE, MaskEdits().to_dict())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
            self.library.addItem(item)
            self._sync_controller_item(item)
            existing.add(resolved)
            added.append(item)
        self.library_count.setText(_("Файлов: ") + str(self.library.count()))
        if added:
            self.library.setCurrentItem(added[0])
        elif candidates and self.library.currentItem():
            self._load_selected()
        else:
            self._status(_("В этой папке нет поддерживаемых фото или видео."), error=True)
        if unsupported:
            self._status(
                _("Поддерживаются JPEG, PNG, TIFF, MP4 и MOV. Не открыто: ")
                + ", ".join(unsupported[:3]),
                error=True,
            )

    def _selection_changed(self, current, _previous) -> None:
        if current:
            self.stop_video_playback()
            self.video_info = None
            self.video_preview_path = None
            self._video_plan = None
            self._video_seek_position = 0.0
            self._sync_controller_selection()
            self.controls.set_settings(
                GradeSettings.from_dict(current.data(SETTINGS_ROLE)), emit=False
            )
            self.local_controls.set_settings(
                LocalSettings.from_dict(current.data(LOCAL_ROLE) or {}), emit=False
            )
            self.mask_edits = MaskEdits.from_dict(current.data(MASK_ROLE) or {})
            self._load_selected()

    def _cancel_view_job(self) -> None:
        if self._view_job in self._jobs:
            self._jobs[self._view_job][0].token.cancel()
        self._view_job = None
        self._full_loading = False

    def _load_selected(self, full_resolution: bool = False) -> None:
        item = self.library.currentItem()
        if item is None:
            return
        if self._export_job is not None:
            self._defer_controller_view()
            return
        self._debounce.stop()
        self._cancel_view_job()
        source = Path(item.data(PATH_ROLE))
        settings = self.controls.settings()
        local = self.local_controls.settings()
        edits = self.mask_edits
        reference = self.reference
        reference_path = self.reference_path
        is_video = source.suffix.lower() in VIDEO_EXTENSIONS
        seek_position = self._video_seek_position
        video_plan = self._video_plan
        cached_analysis = item.data(ANALYSIS_ROLE)
        cached_info = item.data(INFO_ROLE) or {}
        saved_analysis = self.preview.analysis if full_resolution and self.preview else None
        expected_signature = (
            self.preview.photo.stat_signature if full_resolution and self.preview else None
        )
        self._full_loading = full_resolution
        self._render_pending = True
        if not full_resolution:
            self.preview = None
            self.viewer.clear()
            self.warning_label.clear()
            self.image_title.setText(source.name)
            self.metadata_label.setText(_("Чтение медиафайла…"))
        self._status(
            _("Загрузка полного разрешения…") if full_resolution else _("Анализ медиафайла…")
        )

        def operation(token, progress):
            progress(10, _("Чтение медиафайла…"))
            info, actual = None, 0.0
            if is_video:
                from freedive_color.core.video_pipeline import probe_video, read_video_frame

                info = probe_video(source)
                rgb, actual = read_video_frame(
                    source,
                    seek_position,
                    max_edge=max(info.width, info.height) if full_resolution else 1440,
                )
                photo = Photo(
                    source,
                    rgb,
                    None,
                    info.width,
                    info.height,
                    info.bit_depth,
                    source.suffix[1:].upper(),
                    "SDR Rec.709 → sRGB",
                    list(getattr(info, "warnings", ())),
                    stat_signature=signature(source),
                )
            else:
                photo = load_photo(source, max_edge=None if full_resolution else 1440)
            if expected_signature and photo.stat_signature != expected_signature:
                raise StudioError(_("Оригинал изменился. Импортируйте его ещё раз."))
            token.check()
            digest = (
                cached_info.get("sha256")
                if cached_info.get("signature") == list(photo.stat_signature)
                else None
            ) or sha256_file(source)
            if cached_info.get("sha256") and cached_info["sha256"] != digest:
                raise StudioError(
                    _(
                        "Исходник изменился. Выберите «Файл → Повторно импортировать», "
                        "чтобы сбросить старые маски и анализ."
                    )
                )
            resolved_reference = reference
            if resolved_reference is None and reference_path is not None:
                try:
                    resolved_reference = create_image_reference(reference_path)
                except StudioError as exc:
                    photo.warnings.append(_("Эталон недоступен: ") + str(exc))
            resolved_plan = video_plan
            if (
                is_video
                and resolved_plan is None
                and isinstance(cached_info.get("shot_plan"), dict)
            ):
                from freedive_color.core.temporal import ShotPlan

                resolved_plan = ShotPlan.from_dict(cached_info["shot_plan"])
            if is_video and resolved_plan is not None:
                analysis = resolved_plan.analysis_at(float(info.video_start) + actual)
            elif saved_analysis:
                analysis = saved_analysis
            elif cached_analysis and cached_info.get("sha256") == digest:
                analysis = SceneAnalysis.from_dict(cached_analysis)
            else:
                analysis = analyze_scene(photo.rgb)
            progress(50, _("Создание предпросмотра…"))
            result, masks, warnings = render_rgb(
                photo.rgb,
                settings,
                analysis,
                local_settings=local,
                mask_edits=edits,
                reference=resolved_reference,
                check_cancelled=token.check,
            )
            return Preview(
                photo, analysis, result, masks, warnings, digest, resolved_reference, info, actual
            )

        def success(preview):
            self.preview = preview
            self.video_info = preview.video_info
            self.video_frame_seconds = preview.timestamp
            self._configure_video_ui()
            if self.reference is None and preview.reference is not None:
                self.reference = preview.reference
                self.local_controls.reference_label.setText(
                    _("Эталон: ") + self.reference.source_name
                )
            self.rendered_settings = settings
            self.rendered_local_settings = local
            self.rendered_mask_edits = edits
            self._render_pending = False
            self._full_loading = False
            self.viewer.set_images(
                to_qimage(preview.photo.rgb, preview.photo.alpha),
                to_qimage(preview.result, preview.photo.alpha),
                preview.photo.width,
                reset_transform=not full_resolution,
            )
            if full_resolution:
                self.viewer.one_to_one()
            photo = preview.photo
            thumb = QPixmap.fromImage(to_qimage(photo.rgb, photo.alpha)).scaled(
                560,
                320,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(thumb))
            item.setText(source.name + "\n" + _("Готово"))
            item.setData(ANALYSIS_ROLE, preview.analysis.to_dict())
            item.setData(
                INFO_ROLE,
                {
                    **cached_info,
                    **photo.metadata(),
                    "sha256": preview.source_sha256,
                    "signature": list(photo.stat_signature),
                    **({"video": self.video_info.to_dict()} if self.video_info else {}),
                },
            )
            self.image_title.setText(source.name)
            proxy_note = (
                _("полное разрешение")
                if photo.rgb.shape[1] == photo.width
                else _("быстрый просмотр")
            )
            self.metadata_label.setText(
                f"{photo.width} × {photo.height} px · {photo.bit_depth} бит · "
                f"{photo.profile_status} · {proxy_note}"
            )
            self.warning_label.setText(
                "\n".join(
                    dict.fromkeys(
                        photo.warnings
                        + list(preview.analysis.warnings)
                        + list(preview.local_warnings)
                    )
                )
            )
            self._update_mask_overlay()
            self.setWindowTitle(f"{source.name} — Freedive Color Studio")
            self._status(_("Перетащите линию для сравнения. Все регуляторы доступны справа."))
            self.preview_ready.emit()

        self._submit("view", operation, success)
        self._update_actions()

    def _settings_changed(self, settings: GradeSettings) -> None:
        if self.video_info is not None:
            self.stop_video_playback()
            self.video_preview_path = None
        item = self.library.currentItem()
        if item:
            item.setData(SETTINGS_ROLE, settings.to_dict())
        if self.preview is None:
            return
        self._cancel_view_job()
        self._render_pending = True
        self._debounce.start()
        self._update_actions()

    def _render(self) -> None:
        if self._export_job is not None:
            self._defer_controller_view()
            return
        if self.preview is None:
            return
        if (
            self.video_info is not None
            and abs(self.preview.timestamp - self._video_seek_position) > 0.001
        ):
            self._load_selected()
            return
        photo = self.preview.photo
        analysis = self.preview.analysis
        settings = self.controls.settings()
        local = self.local_controls.settings()
        edits = self.mask_edits
        reference = self.reference
        previous = self.preview
        self._status(_("Обновление цвета…"))

        def operation(token, _progress):
            return render_rgb(
                photo.rgb,
                settings,
                analysis,
                local_settings=local,
                mask_edits=edits,
                reference=reference,
                check_cancelled=token.check,
            )

        def success(processed):
            result, masks, warnings = processed
            self.preview = Preview(
                photo,
                analysis,
                result,
                masks,
                warnings,
                previous.source_sha256,
                previous.reference,
                previous.video_info,
                previous.timestamp,
            )
            self.rendered_settings = settings
            self.rendered_local_settings = local
            self.rendered_mask_edits = edits
            self._render_pending = False
            self.viewer.set_images(self.viewer.before, to_qimage(result, photo.alpha), photo.width)
            self.warning_label.setText(
                "\n".join(dict.fromkeys(photo.warnings + list(analysis.warnings) + list(warnings)))
            )
            self._update_mask_overlay()
            self._status(_("Предпросмотр обновлён"))
            self.preview_ready.emit()

        self._submit("view", operation, success)

    def _one_to_one(self) -> None:
        if self.preview is None:
            return
        if self._video_playing:
            self.stop_video_playback()
            self._video_seek_position = self.video_frame_seconds
            self._load_selected(full_resolution=True)
            return
        photo = self.preview.photo
        if photo.rgb.shape[:2] != (photo.height, photo.width):
            self._load_selected(full_resolution=True)
        else:
            self.viewer.one_to_one()

    def _local_settings_changed(self, settings: LocalSettings) -> None:
        item = self.library.currentItem()
        if item:
            item.setData(LOCAL_ROLE, settings.to_dict())
        self._settings_changed(self.controls.settings())

    def _mask_changed(self, edits: MaskEdits) -> None:
        if self._export_job is not None or self.preview is None:
            return
        self.mask_edits = edits
        item = self.library.currentItem()
        if item:
            item.setData(MASK_ROLE, edits.to_dict())
        self._settings_changed(self.controls.settings())

    def _add_stroke(self, values: dict) -> None:
        try:
            stroke = BrushStroke.from_dict(values)
            self._mask_changed(replace(self.mask_edits, strokes=(*self.mask_edits.strokes, stroke)))
            target_index = self.local_controls.overlay_combo.findData(stroke.target)
            self.local_controls.overlay_combo.setCurrentIndex(target_index)
        except StudioError as exc:
            self._status(str(exc), error=True)

    def _set_waterline(self, points) -> None:
        by_x = {round(float(x), 5): float(y) for x, y in points}
        line = sorted(by_x.items())
        if len(line) == 1:
            line = [(0.0, line[0][1]), (1.0, line[0][1])]
        if len(line) > 256:
            line = [line[round(index * (len(line) - 1) / 255)] for index in range(256)]
        try:
            self._mask_changed(replace(self.mask_edits, waterline=tuple(line)))
            self.local_controls.overlay_combo.setCurrentIndex(
                self.local_controls.overlay_combo.findData("underwater")
            )
        except StudioError as exc:
            self._status(str(exc), error=True)

    def _set_person_box(self, box) -> None:
        try:
            self._mask_changed(replace(self.mask_edits, person_box=tuple(box)))
        except StudioError as exc:
            self._status(str(exc), error=True)

    def undo_mask_edit(self) -> None:
        if self.mask_edits.strokes:
            self._mask_changed(replace(self.mask_edits, strokes=self.mask_edits.strokes[:-1]))
        elif self.mask_edits.waterline:
            self._mask_changed(replace(self.mask_edits, waterline=()))
        elif self.mask_edits.person_box:
            self._mask_changed(replace(self.mask_edits, person_box=None))

    def reset_mask_edits(self) -> None:
        self._mask_changed(MaskEdits())

    def reimport_selected(self) -> None:
        item = self.library.currentItem()
        if item and self._export_job is None:
            self.mask_edits = MaskEdits()
            item.setData(MASK_ROLE, self.mask_edits.to_dict())
            item.setData(ANALYSIS_ROLE, None)
            item.setData(INFO_ROLE, {})
            self._load_selected()

    def _update_mask_overlay(self, _mode=None) -> None:
        target = self.local_controls.overlay_combo.currentData()
        masks = self.preview.masks if self.preview else None
        self.viewer.set_mask_overlay(
            getattr(masks, target, None) if target != "none" else None, target
        )
        if masks:
            self.local_controls.confidence_label.setText(
                _("Уверенность маски кожи: ")
                + f"{masks.skin_confidence:.0%}\n"
                + "\n".join(masks.warnings)
            )

    def apply_to_selected(self) -> None:
        if self._export_job is not None:
            return
        items = self.library.selectedItems()
        for item in items:
            item.setData(SETTINGS_ROLE, self.controls.settings().to_dict())
            item.setData(LOCAL_ROLE, self.local_controls.settings().to_dict())
        self._status(_("Настройки применены к снимкам: ") + str(len(items)))

    def reference_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(self, _("Цветовой эталон"), "", OPEN_FILTER)
        if path:
            self.set_reference(Path(path))

    def set_reference(self, path: Path) -> None:
        if self._export_job is not None:
            return
        self._cancel_view_job()
        self._debounce.stop()
        self._render_pending = True

        def operation(token, _progress):
            token.check()
            return create_image_reference(path)

        def success(reference):
            self.reference, self.reference_path = reference, path.resolve()
            self._sync_controller_options()
            self.local_controls.reference_label.setText(_("Эталон: ") + path.name)
            self._local_settings_changed(self.local_controls.settings())

        self._submit("view", operation, success)
        self._update_actions()

    def clear_reference(self) -> None:
        self.reference, self.reference_path = None, None
        self._sync_controller_options()
        self.local_controls.reference_label.setText(_("Эталон не выбран"))
        self._local_settings_changed(self.local_controls.settings())

    def use_frame_reference(self) -> None:
        if self.preview is None or self._render_pending or self._export_job is not None:
            return
        self.stop_video_playback()
        preview = self.preview
        name = preview.photo.path.name
        if preview.video_info is not None:
            name += f" · {preview.timestamp:.2f} с"
        self._cancel_view_job()
        self._debounce.stop()
        self._render_pending = True

        def operation(token, _progress):
            token.check()
            return create_reference(
                preview.result,
                masks=preview.masks,
                source_name=name,
                source_sha256=preview.source_sha256,
            )

        def success(reference):
            self.reference, self.reference_path = reference, None
            self._sync_controller_options()
            self.local_controls.reference_label.setText(_("Эталон кадра: ") + name)
            self._local_settings_changed(self.local_controls.settings())

        self._submit("view", operation, success)
        self._update_actions()

    def _export_options(self) -> ExportOptions:
        return ExportOptions(
            self._photo_export_suffix
            if self.format_combo.currentData() == ".mp4"
            else self.format_combo.currentData(),
            getattr(self, "_photo_export_depth", 8)
            if self.format_combo.currentData() == ".mp4"
            else self.depth_combo.currentData(),
            self.quality_spin.value(),
            self.gps_check.isChecked(),
        )

    def _session_items(self) -> tuple[SessionItem, ...]:
        return self.controller.get_session().items

    def save_project(self, path: Path, overwrite: bool = False) -> None:
        self._sync_controller_options()
        self.controller.save_project_from_ui(path, overwrite=overwrite)
        self.session_path = Path(path).resolve()
        self._status(_("Проект сохранён: ") + str(path))

    def save_project_dialog(self, save_as=False) -> None:
        if self._export_job is not None:
            return
        path = self.session_path if not save_as else None
        if path is None:
            chosen, _selected = QFileDialog.getSaveFileName(
                self,
                _("Сохранить проект"),
                "freedive.fcsproj",
                "Freedive Project (*.fcsproj *.json)",
            )
            if not chosen:
                return
            path = Path(chosen)
            if not path.suffix:
                path = path.with_suffix(".fcsproj")
        try:
            self.save_project(path, overwrite=path == self.session_path)
        except StudioError as exc:
            self._status(str(exc), error=True)

    def open_project_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, _("Открыть проект"), "", "Freedive Project (*.fcsproj *.json)"
        )
        if path:
            try:
                self.load_project(Path(path))
            except StudioError as exc:
                self._status(str(exc), error=True)

    def load_project(self, path: Path) -> None:
        self._load_controller_project(path)

    def batch_export_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, _("Папка результатов серии"), str(self.output_directory or Path.home())
        )
        if path:
            self.start_batch_export(Path(path))

    def start_batch_export(self, directory: Path) -> None:
        if self._export_job is not None:
            return
        items, options, reference = self._session_items(), self._export_options(), self.reference
        self._cancel_view_job()
        self._debounce.stop()
        self._render_pending = False
        self.output_directory = Path(directory).resolve()
        self._sync_controller_options()
        temp_directory = self.temp_directory

        def operation(token, progress):
            return export_batch(
                items,
                directory,
                options,
                token=token,
                reference=reference,
                temp_directory=temp_directory,
                progress=lambda done, total, message: progress(
                    round(done / max(1, total) * 100), message
                ),
            )

        def success(result):
            self.last_batch = result
            statuses = {record.source: record for record in result.records}
            for row in range(self.library.count()):
                item = self.library.item(row)
                record = statuses.get(Path(item.data(PATH_ROLE)))
                if record:
                    caption = {
                        "completed": "Экспортировано",
                        "failed": "Ошибка",
                        "excluded": "Исключён",
                        "cancelled": "Отменён",
                        "not_started": "Не обработан",
                    }.get(record.status, record.status)
                    item.setText(record.source.name + "\n" + _(caption))
                    if record.error:
                        item.setToolTip(record.error)
            self._status(
                f"{_('Готово')}: {result.completed}; {_('ошибок')}: {result.failed}; "
                f"{_('исключено')}: {result.excluded}"
                + (_(" · Отменено") if result.cancelled else ""),
                error=bool(result.failed),
            )
            self.batch_completed.emit(result)

        media_ids = [
            self.controller.media_id_for_path(item.path) for item in items if item.included
        ]
        self._submit(
            "export",
            operation,
            success,
            media_ids=media_ids,
            snapshot={
                "directory": str(self.output_directory),
                "export_options": options.to_dict(),
                "temp_directory": str(temp_directory) if temp_directory else None,
                "excluded_paths": [str(item.path) for item in items if not item.included],
            },
        )
        self._update_actions()

    def _zoom_changed(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")
        if (
            self.preview is None
            or self._full_loading
            or self._export_job is not None
            or self._video_playing
        ):
            return
        photo = self.preview.photo
        if scale >= 0.95 and photo.rgb.shape[:2] != (photo.height, photo.width):
            # Queue out of a paint/resize callback; reaching 100% really loads source pixels.
            self._full_loading = True
            QTimer.singleShot(0, lambda: self._load_selected(full_resolution=True))

    def set_view_mode(self, mode: str) -> None:
        self.mode_buttons[mode].setChecked(True)
        self.viewer.set_mode(mode)

    def _format_changed(self) -> None:
        jpeg = self.format_combo.currentData() == ".jpg"
        video = self.format_combo.currentData() == ".mp4"
        if not video and self.format_combo.currentData():
            self._photo_export_suffix = self.format_combo.currentData()
        self.depth_combo.setEnabled(not jpeg and not video)
        self.quality_spin.setEnabled(jpeg)
        if jpeg or video:
            self.depth_combo.setCurrentIndex(0)
        self._depth_changed()
        if hasattr(self, "_syncing_controller"):
            self._sync_controller_options()

    def _depth_changed(self) -> None:
        limited_metadata = (
            self.format_combo.currentData() == ".tiff" and self.depth_combo.currentData() == 16
        )
        self.gps_check.setEnabled(not limited_metadata)
        if limited_metadata:
            self.gps_check.setChecked(False)

    def export_dialog(self) -> None:
        if self.preview is None or not self.export_button.isEnabled():
            return
        if self.video_info is not None:
            from freedive_color.integrations.ffmpeg import preferred_video_suffix

            source = self.preview.photo.path
            suffix = preferred_video_suffix(track.codec for track in self.video_info.audio_streams)
            formats = (
                "MOV H.264 (*.mov);;MP4 H.264 (*.mp4)"
                if suffix == ".mov"
                else "MP4 H.264 (*.mp4);;MOV H.264 (*.mov)"
            )
            path, _selected = QFileDialog.getSaveFileName(
                self,
                _("Экспорт видео"),
                str(source.with_name(source.stem + "_" + self.controls.settings().preset + suffix)),
                formats,
            )
            if path:
                destination = Path(path)
                self.start_video_export(
                    destination
                    if destination.suffix
                    else destination.with_suffix(
                        ".mov"
                        if "*.mov" in _selected
                        else (".mp4" if "*.mp4" in _selected else suffix)
                    )
                )
            return
        suffix = self.format_combo.currentData()
        source = self.preview.photo.path
        name = source.with_name(source.stem + "_" + self.controls.settings().preset + suffix)
        filters = {
            ".jpg": "JPEG (*.jpg *.jpeg)",
            ".png": "PNG (*.png)",
            ".tiff": "TIFF (*.tiff *.tif)",
        }
        path, _selected = QFileDialog.getSaveFileName(
            self, _("Экспорт фотографии"), str(name), filters[suffix]
        )
        if path:
            destination = Path(path)
            if not destination.suffix:
                destination = destination.with_suffix(suffix)
            self.start_export(destination)

    def start_export(self, destination: Path) -> None:
        if self.preview is None or self._export_job is not None:
            return
        if self.video_info is not None:
            self.start_video_export(destination)
            return
        self._debounce.stop()
        self._cancel_view_job()
        photo = self.preview.photo
        analysis = self.preview.analysis
        settings = self.controls.settings()
        depth, quality = self.depth_combo.currentData(), self.quality_spin.value()
        keep_gps = self.gps_check.isChecked()
        local, edits, reference = self.local_controls.settings(), self.mask_edits, self.reference
        self._status(_("Экспорт фотографии в исходном размере…"))

        def operation(token, progress):
            return export_photo(
                photo.path,
                destination,
                settings,
                analysis=analysis,
                bit_depth=depth,
                quality=quality,
                keep_gps=keep_gps,
                expected_signature=photo.stat_signature,
                local_settings=local,
                mask_edits=edits,
                reference=reference,
                token=token,
                progress=progress,
            )

        def success(report):
            self.last_export = report
            self._status(_("Сохранено: ") + str(destination))
            for row in range(self.library.count()):
                item = self.library.item(row)
                if Path(item.data(PATH_ROLE)) == photo.path:
                    item.setText(photo.path.name + "\n" + _("Экспортировано"))
                    break
            self.export_completed.emit(report)

        self._submit("export", operation, success)
        self._update_actions()

    def save_settings_dialog(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self, _("Сохранить настройки"), "freedive-preset.json", "JSON (*.json)"
        )
        if path:
            try:
                save_settings(
                    Path(path).with_suffix(".json"),
                    self.controls.settings(),
                    local_settings=self.local_controls.settings(),
                )
                self._status(_("Настройки сохранены"))
            except StudioError as exc:
                self._status(str(exc), error=True)

    def load_settings_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, _("Загрузить настройки"), "", "JSON (*.json)"
        )
        if path:
            try:
                data = read_json(path)
                if data.get("local_settings") is not None:
                    self.local_controls.set_settings(
                        LocalSettings.from_dict(data["local_settings"]), emit=False
                    )
                    item = self.library.currentItem()
                    if item:
                        item.setData(LOCAL_ROLE, self.local_controls.settings().to_dict())
                self.controls.set_settings(load_settings(path))
            except StudioError as exc:
                self._status(str(exc), error=True)

    def _submit(self, kind, operation, success, *, media_ids=None, snapshot=None) -> None:
        self._submit_controller_job(
            kind, operation, success, media_ids=media_ids, snapshot=snapshot
        )

    def _is_current(self, identifier: int) -> bool:
        return identifier in (self._view_job, self._export_job) and not self._closing

    def _succeeded(self, identifier: int, result) -> None:
        if self._is_current(identifier) and identifier in self._jobs:
            self._jobs[identifier][2](result)

    def _failed(self, identifier: int, message: str, cancelled: bool) -> None:
        if not self._is_current(identifier):
            return
        self._render_pending = False
        self._full_loading = False
        if identifier == self._view_job and self.preview and self.rendered_settings:
            self.controls.set_settings(self.rendered_settings, emit=False)
            if hasattr(self, "rendered_local_settings"):
                self.local_controls.set_settings(self.rendered_local_settings, emit=False)
                self.mask_edits = self.rendered_mask_edits
            if self.library.currentItem():
                self.library.currentItem().setData(SETTINGS_ROLE, self.rendered_settings.to_dict())
                self.library.currentItem().setData(
                    LOCAL_ROLE, self.local_controls.settings().to_dict()
                )
                self.library.currentItem().setData(MASK_ROLE, self.mask_edits.to_dict())
        self._status(message, error=not cancelled)
        if not cancelled:
            self.operation_failed.emit(message)

    def _progress(self, identifier: int, value: int, message: str) -> None:
        if self._is_current(identifier):
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
            self._status(message)

    def _finished(self, identifier: int) -> None:
        if identifier == self._view_job:
            self._view_job = None
        if identifier == self._export_job:
            self._export_job = None
        self._jobs.pop(identifier, None)
        if self._view_job is None and self._export_job is None:
            self.cancel_button.hide()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self._update_actions()

        self._resume_controller_view()

    def cancel_operation(self) -> None:
        was_waiting = self._debounce.isActive()
        self._debounce.stop()
        for identifier in (self._export_job, self._view_job):
            if identifier in self._jobs:
                self._jobs[identifier][0].token.cancel()
        if was_waiting and self._view_job is None:
            self._render_pending = False
            if self.rendered_settings:
                self.controls.set_settings(self.rendered_settings, emit=False)
                if hasattr(self, "rendered_local_settings"):
                    self.local_controls.set_settings(self.rendered_local_settings, emit=False)
                    self.mask_edits = self.rendered_mask_edits
                item = self.library.currentItem()
                if item:
                    item.setData(SETTINGS_ROLE, self.controls.settings().to_dict())
                    item.setData(LOCAL_ROLE, self.local_controls.settings().to_dict())
                    item.setData(MASK_ROLE, self.mask_edits.to_dict())
            self._update_actions()
        self._status(_("Отмена…"))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and self._export_job is None:
            event.acceptProposedAction()

    def choose_temp_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, _("Папка временных файлов"), str(self.temp_directory or Path.home())
        )
        if path:
            self.temp_directory = Path(path).resolve()
            self._sync_controller_options()
            self._status(_("Временные файлы: ") + str(self.temp_directory))

    def _configure_video_ui(self) -> None:
        info = self.video_info
        if info is not None:
            self.video_sound_note.setText(
                _("В окне — сравнение без звука. Звуковые дорожки сохранены в файле превью.")
                if info.audio_streams
                else _("В исходном видео нет звуковых дорожек.")
            )
            self.video_sound_button.setText(
                _("Открыть со звуком") if info.audio_streams else _("Открыть файл")
            )
        self.video_panel.setVisible(info is not None)
        if info is None:
            index = self.format_combo.findData(".mp4")
            if index >= 0:
                blocker = QSignalBlocker(self.format_combo)
                self.format_combo.removeItem(index)
                self.format_combo.setCurrentIndex(
                    self.format_combo.findData(self._photo_export_suffix)
                )
                del blocker
                self._format_changed()
                self.depth_combo.setCurrentIndex(
                    self.depth_combo.findData(getattr(self, "_photo_export_depth", 8))
                )
            self.format_combo.setEnabled(True)
            return
        if self.format_combo.currentData() != ".mp4":
            self._photo_export_depth = self.depth_combo.currentData()
        index = self.format_combo.findData(".mp4")
        if index < 0:
            self.format_combo.addItem("H.264 SDR · MP4 / MOV", ".mp4")
            index = self.format_combo.count() - 1
        self.format_combo.setCurrentIndex(index)
        self.format_combo.setEnabled(False)
        duration = max(0.1, float(info.duration))
        blocker = QSignalBlocker(self.video_timeline)
        self.video_timeline.setRange(0, round(duration * 1000))
        self.video_timeline.setValue(round(self.video_frame_seconds * 1000))
        del blocker
        self.video_start_spin.setRange(0, max(0, duration - 0.1))
        self.video_duration_spin.setMaximum(min(15, duration))
        self._preview_range_changed()
        self.video_time_label.setText(f"{self.video_frame_seconds:.2f} / {duration:.2f} с")

    def _preview_range_changed(self) -> None:
        if self.video_info is not None:
            self.video_duration_spin.setMaximum(
                min(15.0, max(0.1, float(self.video_info.duration) - self.video_start_spin.value()))
            )

    def seek_video(self, seconds: float) -> None:
        if self.video_info is None or self._export_job is not None:
            return
        self.stop_video_playback()
        self._video_seek_position = max(
            0, min(float(seconds), max(0, float(self.video_info.duration) - 0.001))
        )
        self.video_start_spin.setValue(self._video_seek_position)
        self._sync_controller_selection()
        self._load_selected()

    def _remember_video_plan(self, report: dict) -> None:
        if report.get("plan"):
            from freedive_color.core.temporal import ShotPlan

            source = Path(report.get("source", {}).get("path", "")).resolve()
            for row in range(self.library.count()):
                item = self.library.item(row)
                if Path(item.data(PATH_ROLE)).resolve() == source:
                    if item is self.library.currentItem():
                        self._video_plan = ShotPlan.from_dict(report["plan"])
                    item.setData(
                        INFO_ROLE, {**(item.data(INFO_ROLE) or {}), "shot_plan": report["plan"]}
                    )
                    break

    def create_video_preview(
        self, start_seconds: float, duration_seconds: float, destination: Path | None = None
    ) -> None:
        if self.video_info is None or self.preview is None or self._export_job is not None:
            return
        from freedive_color.core.video_pipeline import export_video

        self.stop_video_playback()
        self._cancel_view_job()
        self._debounce.stop()
        source = self.preview.photo.path
        settings, local, edits, reference = (
            self.controls.settings(),
            self.local_controls.settings(),
            self.mask_edits,
            self.reference,
        )
        info = self.video_info
        start_seconds = max(0.0, min(float(start_seconds), max(0.0, float(info.duration) - 0.001)))
        duration_seconds = max(
            0.1, min(15.0, float(duration_seconds), float(info.duration) - start_seconds)
        )
        temp_directory = self.temp_directory
        plan = self._video_plan or (self.library.currentItem().data(INFO_ROLE) or {}).get(
            "shot_plan"
        )
        if destination is None:
            base = temp_directory or Path(tempfile.gettempdir())
            base.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="freedive-proxy-", dir=base))
            self._proxy_directories.append(directory)
            from freedive_color.integrations.ffmpeg import preferred_video_suffix

            destination = directory / (
                "preview" + preferred_video_suffix(track.codec for track in info.audio_streams)
            )
        destination = Path(destination)
        self._render_pending = False

        def operation(token, progress):
            return export_video(
                source,
                destination,
                settings,
                local_settings=local,
                mask_edits=edits,
                reference=reference,
                token=token,
                progress=progress,
                preview_start=start_seconds,
                preview_seconds=duration_seconds,
                temp_directory=temp_directory,
                proxy_edge=1280,
                plan=plan,
            )

        def success(report):
            self.video_preview_path = destination
            actual = report.get("preview", {}).get("actual_source_start", {})
            self.video_preview_start = float(
                actual.get("seconds", float(info.video_start) + start_seconds)
            ) - float(info.video_start)
            self.video_preview_duration = duration_seconds
            self._remember_video_plan(report)
            self._status(
                _("Превью готово с ")
                + f"{self.video_preview_start:.2f} с. "
                + _("Можно сравнить движение или открыть файл со звуком.")
            )
            self.video_preview_completed.emit(report)

        self._status(_("Анализ сцен и создание превью…"))
        self._submit("export", operation, success)
        self._update_actions()

    def start_video_export(self, destination: Path) -> None:
        if self.video_info is None or self.preview is None or self._export_job is not None:
            return
        from freedive_color.core.video_pipeline import export_video

        self.stop_video_playback()
        self._cancel_view_job()
        self._debounce.stop()
        self._render_pending = False
        source = self.preview.photo.path
        settings, local, edits, reference = (
            self.controls.settings(),
            self.local_controls.settings(),
            self.mask_edits,
            self.reference,
        )
        temp_directory = self.temp_directory
        plan = self._video_plan or (self.library.currentItem().data(INFO_ROLE) or {}).get(
            "shot_plan"
        )
        keep_gps = self.gps_check.isChecked()

        def operation(token, progress):
            return export_video(
                source,
                destination,
                settings,
                local_settings=local,
                mask_edits=edits,
                reference=reference,
                token=token,
                progress=progress,
                temp_directory=temp_directory,
                plan=plan,
                keep_gps=keep_gps,
            )

        def success(report):
            self.last_export = report
            self._remember_video_plan(report)
            self._status(_("Видео сохранено: ") + str(destination))
            self.video_export_completed.emit(report)
            self.export_completed.emit(report)

        self._status(_("Потоковый экспорт видео…"))
        self._submit("export", operation, success)
        self._update_actions()

    def toggle_video_playback(self) -> None:
        if self._video_playing:
            self.stop_video_playback()
            self.seek_video(self.video_frame_seconds)
            return
        if self.video_info is None or self.preview is None or self._export_job is not None:
            return
        if self.video_preview_path is None:
            self.create_video_preview(
                self.video_start_spin.value(), self.video_duration_spin.value()
            )
            return
        self._video_playing = True
        player = VideoPlayback(
            self.preview.photo.path, self.video_preview_path, self.video_preview_start, self
        )
        self.video_playback = player
        source_media_id = self.controller.media_id_for_path(self.preview.photo.path)

        def frame_ready(before, after, seconds):
            try:
                if self.video_playback is player and self._video_playing and not self._closing:
                    self.video_frame_seconds = seconds
                    self._video_seek_position = seconds
                    self.controller.update_view_position(source_media_id, seconds)
                    self.viewer.set_mask_overlay(None)
                    self.viewer.set_images(
                        to_qimage(before), to_qimage(after), self.video_info.width
                    )
                    blocker = QSignalBlocker(self.video_timeline)
                    self.video_timeline.setValue(round(seconds * 1000))
                    del blocker
                    self.video_time_label.setText(
                        f"{seconds:.2f} / {float(self.video_info.duration):.2f} с"
                    )
            finally:
                player.acknowledged.set()

        def finished():
            if self.video_playback is player:
                self._video_playing = False
                self.video_playback = None
                self.video_play_button.setText(_("Воспроизвести"))
                self._update_actions()
                if not self._closing and self._export_job is None:
                    self.seek_video(self.video_frame_seconds)

        player.frame_ready.connect(frame_ready, Qt.ConnectionType.QueuedConnection)
        player.failed.connect(
            lambda error: self._status(error, error=True), Qt.ConnectionType.QueuedConnection
        )
        player.finished.connect(finished, Qt.ConnectionType.QueuedConnection)
        self.video_play_button.setText(_("Пауза"))
        self._update_actions()
        player.start()

    def stop_video_playback(self) -> None:
        player = self.video_playback
        self.video_playback = None
        self._video_playing = False
        if player is not None:
            player.stop()
            player.wait()
        if hasattr(self, "video_play_button"):
            self.video_play_button.setText(_("Воспроизвести"))

    def open_video_preview_external(self) -> None:
        if self.video_preview_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.video_preview_path)))

    def dropEvent(self, event) -> None:
        self.import_paths(
            [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        )
        event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.integration_mode and not self._explicit_quit:
            self.stop_video_playback()
            self.hide()
            event.ignore()
            return
        self._closing = True
        self.stop_video_playback()
        self._debounce.stop()
        for worker, _kind, _success in self._jobs.values():
            worker.token.cancel()
        self.controller.shutdown()
        self.pool.waitForDone()
        sources = [
            Path(self.library.item(index).data(PATH_ROLE)).resolve()
            for index in range(self.library.count())
        ]
        if self.reference_path:
            sources.append(Path(self.reference_path).resolve())
        for directory in self._proxy_directories:
            if not any(source.is_relative_to(directory.resolve()) for source in sources):
                shutil.rmtree(directory, ignore_errors=True)
        event.accept()
