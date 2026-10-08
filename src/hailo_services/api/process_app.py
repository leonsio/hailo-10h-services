"""Production application composition using process-isolated resident backends."""

from hailo_services.api import app as app_module
from hailo_services.runtime.process_runtime import (
    ProcessRuntime,
    ProcessSpeechRuntime,
    ProcessVisionRuntime,
)


def create_process_app(settings):
    """Create the normal API application with process-backed runtime classes.

    The protocol/application layer is intentionally unchanged. The three
    runtime constructors are injected into ``create_app`` while it composes the
    service, preserving all HTTP, MCP, Wyoming, MQTT and ZMQ contracts.

    Returns:
        FastAPI: Application using process-isolated resident inference backends.
    """
    return app_module.create_app(
        settings,
        runtime_factory=ProcessRuntime,
        speech_factory=ProcessSpeechRuntime,
        vision_factory=ProcessVisionRuntime,
    )
