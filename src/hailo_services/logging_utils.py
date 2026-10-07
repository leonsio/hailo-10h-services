"""Logging helpers that keep inline media payloads out of diagnostics."""

from __future__ import annotations

import logging
import re

_INLINE_DATA_URL = re.compile(
    r"data:(?P<mime>[-\w.+/]+);base64,(?P<payload>[A-Za-z0-9+/=_-]+)",
    re.IGNORECASE,
)


def redact_inline_data(text: str) -> str:
    """Replace inline base64 data URLs with compact size metadata.

    Args:
        text: Fully formatted log message.

    Returns:
        Sanitized message without the encoded media payload.
    """

    def replace(match: re.Match[str]) -> str:
        """Render compact metadata for one matched inline data URL.

        Args:
            match: Regular-expression match containing MIME type and base64 payload.

        Returns:
            Redacted data URL preserving MIME type and approximate payload size.
        """
        payload = match.group("payload")
        padding = len(payload) - len(payload.rstrip("="))
        approx_bytes = max(0, (len(payload) * 3) // 4 - padding)
        return (
            f"data:{match.group('mime')};base64,"
            f"<redacted chars={len(payload)} approx_bytes={approx_bytes}>"
        )

    return _INLINE_DATA_URL.sub(replace, text)


class InlineDataRedactionFilter(logging.Filter):
    """Redact inline base64 media before a record reaches a log handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize one formatted record in place.

        Args:
            record: Record about to be emitted by a configured handler.

        Returns:
            Always ``True`` so the sanitized record remains visible.
        """
        message = record.getMessage()
        sanitized = redact_inline_data(message)
        if sanitized != message:
            record.msg = sanitized
            record.args = ()
        return True


def install_inline_data_redaction() -> None:
    """Install one inline-media redaction filter on every root handler.

    Returns:
        None: Existing root handlers are updated in place.
    """
    for handler in logging.getLogger().handlers:
        if not any(isinstance(item, InlineDataRedactionFilter) for item in handler.filters):
            handler.addFilter(InlineDataRedactionFilter())
