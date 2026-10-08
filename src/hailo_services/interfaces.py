"""Structural interfaces for injected backends and shared chat values."""

from __future__ import annotations

from threading import Event
from typing import Any, Callable, Protocol

from hailo_services.schemas import ChatRequest

ChatResult = str | dict[str, Any]
ChatEmitter = Callable[[ChatResult], None]


class ChatBackend(Protocol):
    """Contract implemented by resident Hailo, LiteRT and test chat backends.

    Methods are synchronous and run on a dedicated owner thread. A backend must
    keep its model resident between calls and release resources in ``close``.
    """

    def start(self) -> None:
        """Load resident models on the owner thread.

        Returns:
            None. The backend is ready after successful initialization.

        Raises:
            RuntimeError: Native model initialization fails.
            OSError: A required model artifact cannot be read.
        """
        ...

    def chat(
        self, request: ChatRequest, emit: ChatEmitter | None = None, cancelled: Event | None = None
    ) -> ChatResult:
        """Generate text or a validated assistant tool-call message.

        Args:
            request: Validated chat request with request-local metrics.
            emit: Optional callback for text chunks or validated tool messages.
            cancelled: Optional event requesting generation cancellation.

        Returns:
            Final text or an assistant message containing validated tool calls.

        Raises:
            ValueError: The request, model choice or generated tool call is invalid.
            RuntimeError: The native backend cannot complete inference.
        """
        ...

    def close(self) -> None:
        """Release resident models and devices on their owner thread.

        Returns:
            None. Owned native resources are released.

        Raises:
            RuntimeError: Native cleanup fails, unless the backend logs the failure.
        """
        ...
