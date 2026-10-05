from hailo_services import ha_state_routing as state


def test_climate_target_temperature_is_not_reported_as_current_temperature():
    entity = {
        "name": "ASMOKE Pit thermostat",
        "domain": "climate",
        "state": "off",
        "area": "Garten",
        "attributes": {"current_temperature": "", "temperature": "110"},
    }
    assert state._measurement_value(entity, "temperature") is None


def test_climate_current_temperature_is_reported_when_present():
    entity = {
        "name": "Wohnzimmer Thermostat",
        "domain": "climate",
        "state": "off",
        "area": "Wohnzimmer",
        "attributes": {
            "current_temperature": "21.5",
            "temperature": "19",
            "unit_of_measurement": "°C",
        },
    }
    assert state._measurement_value(entity, "temperature") == "21,5 °C"
