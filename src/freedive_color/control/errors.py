"""Structured local control errors shared by Qt IPC and the MCP adapter."""

from freedive_color.core.errors import StudioError


class ControlError(StudioError):
    def __init__(self, code: str, message: str, hint: str = "", details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details if details is not None else {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }
