"""Production application composition using process-isolated resident backends."""

from . import app as app_module
from .process_workers import ProcessRuntime, ProcessSpeechRuntime, ProcessVisionRuntime


def create_process_app(settings):
    """Create the normal API application with process-backed runtime classes.

    The protocol/application layer is intentionally unchanged. Only the three
    runtime constructors are substituted before ``create_app`` composes the
    service, preserving all HTTP, MCP, Wyoming, MQTT and ZMQ contracts.
    """
    app_module.Runtime = ProcessRuntime
    app_module.SpeechRuntime = ProcessSpeechRuntime
    app_module.VisionRuntime = ProcessVisionRuntime
    return app_module.create_app(settings)
