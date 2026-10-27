import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from freedive_color.core.errors import Cancelled, StudioError
from freedive_color.core.image_pipeline import CancellationToken

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str, bool)
    progress = Signal(int, int, str)
    finished = Signal(int)


class Worker(QRunnable):
    def __init__(self, identifier: int, operation: Callable, token: CancellationToken):
        super().__init__()
        self.identifier, self.operation, self.token = identifier, operation, token
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            self.token.check()
            result = self.operation(self.token, self._progress)
            self.signals.succeeded.emit(self.identifier, result)
        except Cancelled as exc:
            self.signals.failed.emit(self.identifier, str(exc), True)
        except StudioError as exc:
            self.signals.failed.emit(self.identifier, str(exc), False)
        except Exception:
            logger.exception("Background operation failed")
            self.signals.failed.emit(
                self.identifier,
                "Не удалось выполнить обработку. Попробуйте другой снимок или папку.",
                False,
            )
        finally:
            self.signals.finished.emit(self.identifier)

    def _progress(self, value: int, message: str) -> None:
        self.token.check()
        self.signals.progress.emit(self.identifier, value, message)
