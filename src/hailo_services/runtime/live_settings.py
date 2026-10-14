"""Gateway-owned settings that do not require rebuilding native resources."""

LIVE_FIELDS = frozenset({"request_timeout", "piper_max_input_chars", "ha_wait_messages", "vision_confidence", "vision_max_detections"})

LIVE_PATHS = frozenset({f"settings.{name}" for name in LIVE_FIELDS} | {"vision.confidence", "vision.max_detections"})


class LiveSettings:
    """Share a picklable configuration view across gateway components."""

    def __init__(self, base):
        """Initialize an immutable base with replaceable live overrides.

        Returns:
            None: Initializes the view.
        """
        self.base = base
        self.overrides = {}

    def __getattr__(self, name):
        """Resolve overrides first and safely support spawn deserialization.

        Returns:
            Any: Effective configuration value.
        """
        state = object.__getattribute__(self, "__dict__")
        if name in state.get("overrides", {}):
            return state["overrides"][name]
        base = state.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def apply(self, effective):
        """Atomically replace the gateway-only runtime overrides.

        Returns:
            None: Updates live values.
        """
        self.overrides = {name: getattr(effective, name) for name in LIVE_FIELDS}
