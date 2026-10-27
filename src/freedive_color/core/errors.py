class StudioError(Exception):
    """An actionable error suitable for displaying to the user."""


class Cancelled(StudioError):
    def __init__(self) -> None:
        super().__init__("Обработка отменена. Оригинал не изменён.")
