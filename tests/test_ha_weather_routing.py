from hailo_services import ha_weather_routing


ENTITIES = [
    {"name": "ASMOKE Grill temperature 1", "domain": "sensor", "area": "Garten"},
    {"name": "ASMOKE Pit thermostat", "domain": "climate", "area": "Garten"},
    {"name": "Deye M200G4 Temperature", "domain": "sensor", "area": "Garten"},
    {"name": "Wetterstation Frostpunkt", "domain": "sensor", "area": "Garten"},
    {"name": "Wetterstation Rel. Luftfeuchte", "domain": "sensor", "area": "Garten"},
    {"name": "Wetterstation Temperatur", "domain": "sensor", "area": "Garten"},
]


def test_generic_garden_temperature_is_a_read_measurement():
    assert ha_weather_routing._ambient_temperature_query("Welche Temperatur ist im Garten?")
    assert not ha_weather_routing._ambient_temperature_query(
        "Stelle das Thermostat im Garten auf 22 Grad"
    )


def test_outdoor_temperature_is_not_expanded_to_full_weather_summary():
    assert ha_weather_routing._ambient_temperature_query("Welche Temperatur ist draußen?")
    assert not ha_weather_routing._weather_query("Welche Temperatur ist draußen?")


def test_garden_temperature_prefers_weather_station_over_grill_and_device_temperature():
    sources = ha_weather_routing._temperature_sources(
        ENTITIES,
        "Welche Temperatur ist im Garten?",
    )
    assert sources == [
        {"name": "Wetterstation Temperatur", "domain": "sensor", "area": "Garten"}
    ]


def test_weather_query_uses_only_weather_station_entities():
    assert ha_weather_routing._weather_query("Wie ist das Wetter draußen?")
    sources = ha_weather_routing._weather_sources(ENTITIES, "Wie ist das Wetter draußen?")
    assert [item["name"] for item in sources] == [
        "Wetterstation Temperatur",
        "Wetterstation Rel. Luftfeuchte",
        "Wetterstation Frostpunkt",
    ]


def test_off_climate_is_not_accepted_as_current_generic_temperature():
    entities = [
        {
            "name": "ASMOKE Pit thermostat",
            "domain": "climate",
            "state": "off",
            "area": "Garten",
            "attributes": {"current_temperature": "31.0", "unit_of_measurement": "°C"},
        }
    ]
    response = ha_weather_routing._temperature_response(
        entities,
        "Welche Temperatur ist im Garten?",
    )
    assert response == "Aktuell habe ich keine aktuellen Temperaturdaten für Garten."
    assert "31" not in response


def test_climate_target_setpoint_is_never_used_as_current_temperature():
    entities = [
        {
            "name": "ASMOKE Pit thermostat",
            "domain": "climate",
            "state": "heat",
            "area": "Garten",
            "attributes": {
                "current_temperature": "",
                "temperature": "110",
                "unit_of_measurement": "°C",
            },
        }
    ]
    response = ha_weather_routing._temperature_response(
        entities,
        "Welche Temperatur ist im Garten?",
    )
    assert response == "Aktuell habe ich keine aktuellen Temperaturdaten für Garten."
    assert "110" not in response


def test_weather_response_combines_live_station_values_without_llm():
    entities = [
        {
            "name": "Wetterstation Temperatur",
            "domain": "sensor",
            "state": "12.4",
            "area": "Garten",
            "attributes": {"unit_of_measurement": "°C"},
        },
        {
            "name": "Wetterstation Rel. Luftfeuchte",
            "domain": "sensor",
            "state": "81",
            "area": "Garten",
            "attributes": {"unit_of_measurement": "%"},
        },
        {
            "name": "Wetterstation Frostpunkt",
            "domain": "sensor",
            "state": "9.2",
            "area": "Garten",
            "attributes": {"unit_of_measurement": "°C"},
        },
    ]
    response = ha_weather_routing._weather_response(entities)
    assert "12,4 °C" in response
    assert "81 %" in response
    assert "9,2 °C" in response


def test_unavailable_weather_data_reports_no_current_data():
    entities = [
        {
            "name": "Wetterstation Temperatur",
            "domain": "sensor",
            "state": "unavailable",
            "area": "Garten",
            "attributes": {"unit_of_measurement": "°C"},
        }
    ]
    assert ha_weather_routing._weather_response(entities) == (
        "Aktuell sind die Wetterdaten in Home Assistant nicht verfügbar."
    )
