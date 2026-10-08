"""Public process-runtime aliases for the production gateway."""

from hailo_services.runtime.workers.worker import (
    ProcessRuntime,
    ProcessSpeechRuntime,
    ProcessVisionRuntime,
)

__all__ = ["ProcessRuntime", "ProcessSpeechRuntime", "ProcessVisionRuntime"]
