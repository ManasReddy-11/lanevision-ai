from __future__ import annotations


class AIModelError(RuntimeError):
    """Raised when model loading or inference fails."""


class AIUnavailableError(AIModelError):
    """Raised when the AI model cannot be used because it is missing or disabled."""
