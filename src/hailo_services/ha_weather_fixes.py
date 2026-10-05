"""Harden weather and ambient-temperature routing semantics."""

from __future__ import annotations

from functools import wraps

from . import ha_weather_routing as _weather


def install():
    """Keep measurement reads separate from climate targets and weather summaries."""
    if getattr(_weather, "_weather_semantics_fixed", False):
        return

    original_live_value = _weather._live_value
    original_weather_query = _weather._weather_query

    @wraps(original_live_value)
    def live_value(entity, kind: str, *, generic_temperature: bool = False):
        # In Home Assistant climate entities, ``temperature`` is the target
        # setpoint.  A current ambient reading exists only when
        # ``current_temperature`` is actually present.  This guard is needed
        # here as well because the weather router imported its measurement
        # helper before the global measurement-semantics patch was installed.
        if kind == "temperature" and entity.get("domain") == "climate":
            attrs = entity.get("attributes", {})
            if attrs.get("current_temperature") in {None, ""}:
                return None
        return original_live_value(
            entity,
            kind,
            generic_temperature=generic_temperature,
        )

    @wraps(original_weather_query)
    def weather_query(text: str) -> bool:
        # "Wie warm / welche Temperatur ... draußen?" is a measurement
        # question, not a request for a complete weather summary.  Keeping the
        # two intents separate avoids fetching humidity/dew point unnecessarily.
        if _weather._ambient_temperature_query(text):
            return False
        return original_weather_query(text)

    _weather._live_value = live_value
    _weather._weather_query = weather_query
    _weather._weather_semantics_fixed = True
