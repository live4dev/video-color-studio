"""GUI-owned authenticated local server and independent desktop lifecycle."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from PySide6.QtCore import QEvent, QLockFile, QObject, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .protocol import (
    API_VERSION,
    MAX_MESSAGE_BYTES,
    ControlPaths,
    LineDecoder,
    ProtocolError,
    encode_json,
    error_response,
    private_directory,
    read_connection,
    validate_request,
    write_private,
)


class InstanceLease:
    """An age-independent Qt lock; Qt removes it only when its process is stale."""

    def __init__(self, paths: ControlPaths):
        self.paths = paths
        private_directory(paths.directory)
        self.lock = QLockFile(str(paths.lock))
        self.lock.setStaleLockTime(0)

    def acquire(self) -> bool:
        acquired = self.lock.tryLock(0)
        if acquired and os.name != "nt":
            self.paths.lock.chmod(0o600)
        return acquired

    def release(self):
        self.lock.unlock()


class ApplicationRuntime(QObject):
    connection_changed = Signal(int)

    def __init__(
        self,
        app,
        controller,
        window,
        *,
        integration_mode=False,
        paths: ControlPaths | None = None,
        lease: InstanceLease | None = None,
    ):
        super().__init__(app)
        self.app, self.controller, self.window = app, controller, window
        self.paths = paths or ControlPaths.default()
        self.lease = lease or InstanceLease(self.paths)
        if not self.lease.lock.isLocked() and not self.lease.acquire():
            raise ProtocolError("ALREADY_RUNNING", "Приложение уже запущено.")
        self.instance_id = secrets.token_hex(16)
        self.token = secrets.token_urlsafe(32)
        self.endpoint = self.paths.endpoint()
        self._clients: dict[QLocalSocket, LineDecoder] = {}
        self._closing = False
        self._closed = False
        self.window.integration_mode = integration_mode
        self.window._control_runtime = self
        self.app.installEventFilter(self)
        self.app.setQuitOnLastWindowClosed(False)
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.server.setMaxPendingConnections(8)
        # Only a process that owns the lifetime lock can remove a stale endpoint.
        QLocalServer.removeServer(self.endpoint)
        if not self.server.listen(self.endpoint):
            self.lease.release()
            raise ProtocolError(
                "CHANNEL_UNAVAILABLE", "Не удалось открыть локальный канал приложения."
            )
        if os.name != "nt":
            Path(self.server.fullServerName()).chmod(0o600)
        try:
            write_private(self.paths.token, (self.token + "\n").encode("ascii"))
            write_private(
                self.paths.registry,
                json.dumps(
                    {
                        "api_version": API_VERSION,
                        "endpoint": self.endpoint,
                        "pid": os.getpid(),
                        "instance_id": self.instance_id,
                    }
                ).encode("utf-8"),
            )
        except Exception:
            self.server.close()
            self.lease.release()
            raise
        self.server.newConnection.connect(self._accept)
        self._quit_timer = QTimer(self)
        self._quit_timer.setInterval(50)
        self._quit_timer.timeout.connect(self._finish_quit)
        self._create_tray()
        self.app.aboutToQuit.connect(self.close)

    def _create_tray(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor("#126c8a"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("white"))
        painter.drawText(pixmap.rect(), 0x84, "F")
        painter.end()
        self.tray = QSystemTrayIcon(QIcon(pixmap), self)
        self.tray.setToolTip("Freedive Color Studio")
        self.menu = QMenu()
        self.open_action = QAction("Открыть", self.menu)
        self.quit_action = QAction("Выход", self.menu)
        self.open_action.triggered.connect(self.show_window)
        self.quit_action.triggered.connect(self.request_quit)
        self.menu.addAction(self.open_action)
        self.menu.addAction(self.quit_action)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _tray_activated(self, reason):
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_window()

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def eventFilter(self, watched, event):
        if watched is self.app and event.type() == QEvent.Type.Quit and not self._closing:
            self.request_quit()
            return True
        if watched is self.window and event.type() == QEvent.Type.Close and not self._closing:
            event.ignore()
            if self.window.integration_mode:
                if hasattr(self.window, "stop_video_playback"):
                    self.window.stop_video_playback()
                self.window.hide()
            else:
                self.request_quit()
            return True
        return super().eventFilter(watched, event)

    def _accept(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            if len(self._clients) >= 16 or self._closing:
                socket.abort()
                socket.deleteLater()
                continue
            socket.setReadBufferSize(MAX_MESSAGE_BYTES + 1)
            self._clients[socket] = LineDecoder()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._disconnect(s))
            # A client that stalls in the middle of a line cannot occupy a slot forever.
            deadline = QTimer(socket)
            deadline.setSingleShot(True)
            deadline.setInterval(15000)
            deadline.timeout.connect(lambda s=socket: s.disconnectFromServer())
            deadline.start()
            socket._control_deadline = deadline
            self.connection_changed.emit(len(self._clients))
            if socket.bytesAvailable():
                self._read(socket)

    def _disconnect(self, socket):
        self._clients.pop(socket, None)
        socket.deleteLater()
        self.connection_changed.emit(len(self._clients))

    def _send(self, socket, value):
        try:
            data = encode_json(value)
        except ProtocolError as error:
            data = encode_json(error_response(value.get("id"), error))
        # Bound pending replies too, rather than allowing an unread peer to grow memory.
        if socket.bytesToWrite() + len(data) > MAX_MESSAGE_BYTES * 2:
            socket.abort()
            return
        socket.write(data)
        socket.flush()

    def _read(self, socket):
        decoder = self._clients.get(socket)
        if decoder is None:
            return
        try:
            while socket.bytesAvailable():
                chunk = bytes(socket.read(min(65536, socket.bytesAvailable())))
                for request in decoder.feed(chunk):
                    identifier = request.get("id")
                    if type(identifier) not in (str, int):
                        identifier = None
                    try:
                        identifier, method, params = validate_request(request, self.token)
                        result = self._dispatch(method, params)
                        self._send(socket, {"jsonrpc": "2.0", "id": identifier, "result": result})
                    except Exception as error:
                        self._send(socket, error_response(identifier, error))
                        if getattr(error, "code", "") in ("UNAUTHORIZED", "API_VERSION_MISMATCH"):
                            socket.disconnectFromServer()
                            return
        except ProtocolError as error:
            self._send(socket, error_response(None, error))
            socket.disconnectFromServer()

    def _dispatch(self, method, params):
        if self._closing:
            raise ProtocolError("APP_SHUTTING_DOWN", "Приложение завершает работу.")
        result = self.controller.execute(method, params)
        if method == "connect_session":
            self.window.integration_mode = True
            self.show_window()
            result = {
                **result,
                "connection": {
                    "api_version": API_VERSION,
                    "instance_id": self.instance_id,
                    "window_visible": True,
                },
            }
        elif method == "show_window":
            if params["visible"]:
                self.show_window()
            else:
                self.window.hide()
        return result

    def request_quit(self):
        if self._closing:
            return
        self._closing = True
        self.window.request_quit()
        self.controller.shutdown()
        self._quit_timer.start()
        self._finish_quit()

    def _finish_quit(self):
        if not self.controller.is_idle():
            return
        pool = getattr(self.window, "pool", None)
        if pool is not None and pool.activeThreadCount():
            return
        self._quit_timer.stop()
        self.close()
        self.window.close()
        self.app.quit()

    def close(self):
        """Remove only this instance's registry while its lifetime lease is held."""
        if self._closed:
            return
        self._closed = True
        self.server.close()
        for socket in list(self._clients):
            socket.abort()
        try:
            registry, _token = read_connection(self.paths)
            if registry.get("instance_id") == self.instance_id:
                self.paths.registry.unlink(missing_ok=True)
                self.paths.token.unlink(missing_ok=True)
        except (OSError, ValueError, ProtocolError):
            pass
        self.tray.hide()
        self.lease.release()
