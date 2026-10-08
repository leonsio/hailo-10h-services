from hailo_services.assistants.ha import ha_state_routing as state


def test_climate_target_temperature_is_not_reported_as_current_temperature():
    entity = {
        "name": "Room Thermostat",
        "domain": "climate",
        "state": "off",
        "area": "Living Room",
        "attributes": {"current_temperature": "", "temperature": "24"},
    }
    assert state._measurement_value(entity, "temperature") is None


def test_climate_current_temperature_is_reported_when_present():
    entity = {
        "name": "Room Thermostat",
        "domain": "climate",
        "state": "off",
        "area": "Living Room",
        "attributes": {
            "current_temperature": "21.5",
            "temperature": "19",
            "unit_of_measurement": "°C",
        },
    }
    assert state._measurement_value(entity, "temperature") == "21,5 °C"
