import logging

from hailo_services.diagnostics.logging_utils import (
    InlineDataRedactionFilter,
    install_inline_data_redaction,
    redact_inline_data,
)


def test_inline_data_url_is_replaced_with_size_metadata():
    text = 'body={"image":"data:image/jpeg;base64,QUJDRA=="}'
    sanitized = redact_inline_data(text)
    assert "QUJDRA==" not in sanitized
    assert "data:image/jpeg;base64,<redacted chars=8 approx_bytes=4>" in sanitized


def test_log_filter_sanitizes_formatted_arguments():
    record = logging.LogRecord(
        "hailo_services.test",
        logging.DEBUG,
        __file__,
        1,
        "request=%s",
        ("data:image/png;base64,QUJD",),
        None,
    )
    assert InlineDataRedactionFilter().filter(record)
    assert record.args == ()
    assert "QUJD" not in record.getMessage()
    assert "approx_bytes=3" in record.getMessage()


def test_install_is_idempotent(monkeypatch):
    handler = logging.StreamHandler()
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [handler])
    install_inline_data_redaction()
    install_inline_data_redaction()
    assert sum(isinstance(item, InlineDataRedactionFilter) for item in handler.filters) == 1
