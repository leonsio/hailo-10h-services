"""Backend-independent service exceptions."""


class BusyError(RuntimeError):
    """Signal unavailable models or a full owner-thread inference queue."""


class LiteRTInferenceError(RuntimeError):
    """Wrap native LiteRT failures with configured context information."""
