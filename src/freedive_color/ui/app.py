import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .. import __version__
from ..control.client import ApplicationClient
from ..control.protocol import ControlPaths, RemoteControlError
from ..control.runtime import ApplicationRuntime, InstanceLease
from .theme import apply_theme


def run(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    integration_mode = "--integration" in arguments
    arguments = list(arguments)
    connection_paths = ControlPaths.default()
    if "--control-dir" in arguments:
        index = arguments.index("--control-dir")
        if index + 1 >= len(arguments):
            print("Не указан каталог локального подключения.", file=sys.stderr)
            return 2
        connection_paths = ControlPaths(Path(arguments[index + 1]).expanduser().absolute())
        del arguments[index : index + 2]
    paths = [
        Path(value).expanduser().absolute() for value in arguments if not value.startswith("-")
    ]
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Freedive Color Studio")
    app.setOrganizationName("Freedive Color Studio")
    app.setApplicationVersion(__version__)
    lease = InstanceLease(connection_paths)
    if not lease.acquire():
        client = ApplicationClient(paths=connection_paths, timeout=1, launch=False)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                connected = client.call("connect_session", {})
                if paths:
                    client.call(
                        "import_media",
                        {
                            "session_id": connected["session_id"],
                            "paths": [str(path) for path in paths],
                        },
                    )
                return 0
            except RemoteControlError as error:
                if error.code != "APP_NOT_RUNNING":
                    print(error.message, file=sys.stderr)
                    return 1
                time.sleep(0.05)
        print("Запущенный экземпляр приложения не отвечает.", file=sys.stderr)
        return 1
    runtime = None
    try:
        from ..control.controller import ProjectController
        from .main_window import MainWindow

        apply_theme(app)
        controller = ProjectController(
            journal_path=connection_paths.directory / "jobs.json",
            temp_root=connection_paths.directory / "previews",
        )
        window = MainWindow(controller=controller, integration_mode=integration_mode)
        runtime = ApplicationRuntime(
            app,
            controller,
            window,
            integration_mode=integration_mode,
            paths=connection_paths,
            lease=lease,
        )
        if not integration_mode:
            window.show()
        if paths:
            QTimer.singleShot(0, lambda: window.import_paths(paths))
        return app.exec()
    finally:
        if runtime is not None:
            runtime.close()
        else:
            lease.release()
