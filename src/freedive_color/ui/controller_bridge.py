"""Keep widgets as views of the application controller, including remote edits."""

import json
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox, QStyle

from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import MaskEdits
from freedive_color.core.presets import GradeSettings
from freedive_color.core.reference import PaletteReference
from freedive_color.core.session import SessionItem, load_session

PATH = Qt.ItemDataRole.UserRole
GRADE = PATH + 1
LOCAL = PATH + 2
MASK = PATH + 3
ANALYSIS = PATH + 4
INFO = PATH + 5
MEDIA_ID = PATH + 6


class ControllerWindowMixin:
    def _attach_controller(self) -> None:
        self._syncing_controller = False
        self._controller_session_id = self.controller.session_id
        self._observed_remote_previews = set()
        self._deferred_controller_refresh = False
        self.controller.state_changed.connect(self._controller_changed)
        self.controller.job_changed.connect(self._controller_job_changed)
        self.library.itemChanged.connect(self._library_item_changed)
        self.quality_spin.valueChanged.connect(lambda _value: self._sync_controller_options())
        self.gps_check.toggled.connect(lambda _value: self._sync_controller_options())
        menu = self.menuBar().addMenu("Codex")
        connection = menu.addAction("Подключение к Codex…")
        connection.triggered.connect(self._show_codex_connection)
        copy = menu.addAction("Скопировать конфигурацию MCP")
        copy.triggered.connect(self._copy_mcp_configuration)
        quit_action = QAction("Выход", self)
        quit_action.triggered.connect(self.request_quit)
        menu.addAction(quit_action)
        if self.controller.records():
            self._controller_changed({"kind": "project"})

    def _controller_item(self, item) -> SessionItem:
        info = item.data(INFO) or {}
        return SessionItem(
            Path(item.data(PATH)),
            GradeSettings.from_dict(item.data(GRADE)),
            item.checkState() == Qt.CheckState.Checked,
            item.data(LOCAL) or {},
            item.data(MASK) or {},
            item.data(ANALYSIS),
            info,
            info.get("sha256"),
        )

    def _sync_controller_item(self, item) -> None:
        if item is None or self._syncing_controller:
            return
        self._syncing_controller = True
        try:
            record = self._controller_item(item)
            media_id = item.data(MEDIA_ID) or self.controller.media_id_for_path(record.path)
            if media_id is None:
                media_id = self.controller.add_item(record)
                item.setData(MEDIA_ID, media_id)
            else:
                self.controller.replace_item(media_id, record)
                item.setData(MEDIA_ID, media_id)
        finally:
            self._syncing_controller = False

    def _library_item_changed(self, item) -> None:
        # Metadata and checkboxes also belong to the project, rather than the QListWidget.
        if item.data(PATH) and item.data(GRADE):
            self._sync_controller_item(item)

    def _sync_controller_selection(self) -> None:
        if self._syncing_controller:
            return
        item = self.library.currentItem()
        if item is None:
            return
        self._sync_controller_item(item)
        self._syncing_controller = True
        try:
            self.controller.execute(
                "select_media",
                {
                    "session_id": self.controller.session_id,
                    "media_id": item.data(MEDIA_ID),
                    "time_seconds": self._video_seek_position,
                },
            )
        finally:
            self._syncing_controller = False

    def _sync_controller_options(self) -> None:
        if self._syncing_controller:
            return
        self._syncing_controller = True
        try:
            self.controller.update_session_options(
                reference_path=self.reference_path,
                reference_data=self.reference.to_dict() if self.reference else None,
                temp_directory=self.temp_directory,
                output_directory=self.output_directory,
                export_options=self._export_options(),
                selected_index=self.library.currentRow()
                if self.library.currentRow() >= 0
                else None,
            )
        finally:
            self._syncing_controller = False

    def _controller_changed(self, event) -> None:
        if self._syncing_controller or self._closing:
            return
        kind = event.get("kind", "") if isinstance(event, dict) else ""
        if kind.startswith("job"):
            return
        self._syncing_controller = True
        try:
            session = self.controller.get_session()
            records = self.controller.records()
            new_project = self._controller_session_id != self.controller.session_id
            previous = self.library.currentItem()
            old_path = previous.data(PATH) if previous else None
            old_processing = (
                (previous.data(GRADE), previous.data(LOCAL), previous.data(MASK))
                if previous
                else None
            )
            if new_project:
                self._cancel_view_job()
                self._debounce.stop()
                self.stop_video_playback()
                self.preview = None
                self.video_info = None
                self.video_preview_path = None
                self._video_plan = None
                self._render_pending = False
                self.viewer.clear()
            blocker = QSignalBlocker(self.library)
            if new_project:
                self.library.clear()
            by_id = {
                self.library.item(i).data(MEDIA_ID): self.library.item(i)
                for i in range(self.library.count())
            }
            for record in records:
                item = by_id.get(record.media_id)
                if item is None:
                    item = QListWidgetItem(record.item.path.name)
                    item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                    self.library.addItem(item)
                source = record.item
                item.setData(PATH, str(source.path))
                item.setData(MEDIA_ID, record.media_id)
                item.setData(GRADE, source.settings.to_dict())
                item.setData(LOCAL, source.local_settings or LocalSettings().to_dict())
                item.setData(MASK, source.mask_edits or MaskEdits().to_dict())
                item.setData(ANALYSIS, source.analysis)
                item.setData(
                    INFO,
                    {
                        **source.media_info,
                        **({"sha256": source.source_sha256} if source.source_sha256 else {}),
                    },
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if source.included else Qt.CheckState.Unchecked
                )
                item.setToolTip(str(source.path))
            self._controller_session_id = self.controller.session_id
            self.reference_path = session.reference_path
            self.reference = (
                PaletteReference.from_dict(session.reference_data)
                if session.reference_data
                else None
            )
            self.local_controls.reference_label.setText(
                "Эталон: " + self.reference.source_name if self.reference else "Эталон не выбран"
            )
            self.temp_directory = session.temp_directory
            self.output_directory = session.output_directory
            self.session_path = getattr(self.controller, "project_path", self.session_path)
            options = session.export_options
            suffix = {".jpeg": ".jpg", ".tif": ".tiff"}.get(options.suffix, options.suffix)
            if not self.video_info:
                self.format_combo.setCurrentIndex(self.format_combo.findData(suffix))
                self.depth_combo.setCurrentIndex(self.depth_combo.findData(options.bit_depth))
            self.quality_spin.setValue(options.quality)
            self.gps_check.setChecked(options.keep_gps and self.gps_check.isEnabled())
            index = session.selected_index
            if index is not None:
                self.library.setCurrentRow(index)
            elif self.library.count() and self.library.currentRow() < 0:
                self.library.setCurrentRow(0)
            del blocker
            self.library_count.setText("Файлов: " + str(self.library.count()))
            current = self.library.currentItem()
            current_path = current.data(PATH) if current else None
            processing = (
                (current.data(GRADE), current.data(LOCAL), current.data(MASK)) if current else None
            )
            if current:
                self.controls.set_settings(GradeSettings.from_dict(current.data(GRADE)), emit=False)
                self.local_controls.set_settings(
                    LocalSettings.from_dict(current.data(LOCAL)), emit=False
                )
                self.mask_edits = MaskEdits.from_dict(current.data(MASK))
                selected_time = getattr(self.controller, "time_seconds", 0.0)
                selection_changed = new_project or current_path != old_path or kind == "selection"
                if selection_changed:
                    self.stop_video_playback()
                    self.video_info = None
                    self.video_preview_path = None
                    self._video_plan = None
                    self._video_seek_position = selected_time
                    self._load_selected()
                elif processing != old_processing or "reference" in kind:
                    self.stop_video_playback()
                    self.video_preview_path = None
                    self._cancel_view_job()
                    self._render_pending = True
                    if self._export_job is not None:
                        self._defer_controller_view()
                    elif self.preview is not None:
                        self._debounce.start()
                    else:
                        self._load_selected()
            self._update_actions()
        finally:
            self._syncing_controller = False

    def _defer_controller_view(self) -> None:
        """Keep an export's frozen inputs, then refresh the current project selection."""
        self._deferred_controller_refresh = True
        self._debounce.stop()
        self._render_pending = True
        self.preview = None
        self.video_preview_path = None
        self.viewer.clear()
        self._update_actions()

    def _resume_controller_view(self) -> None:
        if self._deferred_controller_refresh and self._export_job is None and not self._closing:
            self._deferred_controller_refresh = False
            # An older export/preview callback may just have returned; none of its
            # view caches may become the newly selected file's playback state.
            self.preview = None
            self.video_info = None
            self.video_preview_path = None
            self._video_plan = None
            self._load_selected()

    def _controller_job_changed(self, job_id: str) -> None:
        if self._closing:
            return
        job = self.controller.get_job(job_id)
        # Only an explicitly current-revision preview may enter the moving comparison.
        if (
            job.get("kind") == "render_preview"
            and job.get("status") == "succeeded"
            and not job.get("stale_preview")
            and job_id not in self._observed_remote_previews
        ):
            self._observed_remote_previews.add(job_id)
            self._accept_remote_preview(job)
        if self._view_job is None and self._export_job is None:
            state = job.get("status")
            self._status(job.get("message") or job.get("stage") or str(state))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(job.get("progress", 0)))
            self.cancel_button.setVisible(state in {"queued", "running"})

    def _accept_remote_preview(self, job: dict) -> None:
        # Adapted to the controller's public result schema below; no media reaches MCP.
        result = job.get("result") or {}
        records = result.get("results", []) if isinstance(result, dict) else []
        if not records and isinstance(result, dict):
            records = [result]
        current = self.library.currentItem()
        if current is None:
            return
        media_id = current.data(MEDIA_ID)
        record = self.controller.read_record(media_id)
        for item in records:
            if item.get("media_id", media_id) != media_id:
                continue
            if item.get("revision", record.revision) != record.revision or item.get("stale"):
                continue
            path = item.get("path") or item.get("output_path")
            if not path and isinstance(item.get("output"), dict):
                path = item["output"].get("path")
            if path and Path(path).is_file() and Path(path).suffix.lower() in {".mp4", ".mov"}:
                self.video_preview_path = Path(path)
                preview = item.get("preview") or item.get("recipe", {}).get("preview", {})
                actual = preview.get("actual_source_start", {})
                self.video_preview_start = float(actual.get("seconds", 0.0)) - (
                    float(self.video_info.video_start) if self.video_info else 0.0
                )
                self.video_preview_duration = float(
                    item.get("duration_seconds", preview.get("requested_seconds", 10.0))
                )

    def _submit_controller_job(
        self, kind, operation, success, *, media_ids=None, snapshot=None
    ) -> None:
        self._counter += 1
        identifier = self._counter
        current = self.library.currentItem()
        if media_ids is None:
            media_ids = [current.data(MEDIA_ID)] if current and current.data(MEDIA_ID) else []
        if kind == "view":
            self._view_job = identifier
        else:
            self._export_job = identifier
        handle = SimpleNamespace(job_id=None)

        def cancel():
            if handle.job_id:
                self.controller.execute(
                    "cancel_job",
                    {
                        "session_id": self.controller.session_id,
                        "job_id": handle.job_id,
                    },
                )

        handle.token = SimpleNamespace(cancel=cancel)
        self._jobs[identifier] = (handle, kind, success)
        handle.job_id = self.controller.submit_ui_job(
            kind,
            operation,
            media_ids=media_ids,
            on_success=lambda result: self._succeeded(identifier, result),
            on_error=lambda message, cancelled: self._failed(identifier, message, cancelled),
            on_progress=lambda value, message: self._progress(identifier, value, message),
            on_finished=lambda: self._finished(identifier),
            snapshot=snapshot,
        )
        self.progress_bar.setRange(0, 0)
        self.cancel_button.show()

    def _load_controller_project(self, path: Path) -> None:
        self.controller.replace_session(load_session(path), project_path=path)

    def _mcp_configuration_text(self) -> str:
        from freedive_color.control.client import mcp_configuration

        config = mcp_configuration()
        # A JSON string is also a TOML basic string, including Windows backslashes.
        return (
            "[mcp_servers.freedive_color]\ncommand = "
            + json.dumps(config["command"], ensure_ascii=False)
            + "\nargs = "
            + json.dumps(config.get("args", []), ensure_ascii=False)
            + "\n"
        )

    def _copy_mcp_configuration(self) -> None:
        QApplication.clipboard().setText(self._mcp_configuration_text())
        self._status("Конфигурация MCP скопирована. Добавьте её в настройки Codex.")

    def _show_codex_connection(self) -> None:
        message = (
            "Локальное подключение MCP · API 1\n"
            f"Проект: {self.controller.session_id}\n"
            "Общее состояние окна и чата. Обработка остаётся на этом компьютере.\n"
            "После подключения закрытие окна оставляет очередь работать в фоне.\n\n"
            + self._mcp_configuration_text()
            + "\nГлобальные настройки Codex автоматически не меняются."
        )
        QMessageBox.information(self, "Подключение к Codex", message)

    def request_quit(self) -> None:
        self._explicit_quit = True
        self.stop_video_playback()
        self._debounce.stop()
        self._cancel_view_job()
        runtime = getattr(self, "_control_runtime", None)
        if runtime is not None:
            runtime.request_quit()
        else:
            self.close()
