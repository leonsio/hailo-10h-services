"""Fix Home Assistant measurement semantics used by deterministic state routing."""

from __future__ import annotations

from functools import wraps

from . import ha_state_routing as _state


def install():
    """Do not mistake a climate target setpoint for a current measurement."""
    if getattr(_state, "_measurement_semantics_fixed", False):
        return

    original_measurement_value = _state._measurement_value

    @wraps(original_measurement_value)
    def measurement_value(entity, kind: str):
        if kind == "temperature" and entity.get("domain") == "climate":
            attrs = entity.get("attributes", {})
            value = attrs.get("current_temperature")
            if value in {None, ""}:
                return None
            unit = attrs.get("unit_of_measurement") or attrs.get("unit") or ""
            return f"{_state._format_number(value)}{(' ' + str(unit)) if unit else ''}"
        return original_measurement_value(entity, kind)

    _state._measurement_value = measurement_value
    _state._measurement_semantics_fixed = True
