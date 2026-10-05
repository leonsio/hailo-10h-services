from hailo_services import ha_weather_routing

GENERIC_ENTITIES = [
    {"name": "Outdoor Temperature", "domain": "sensor", "area": "Garden"},
    {"name": "Outdoor Humidity", "domain": "sensor", "area": "Garden"},
    {"name": "Inverter Temperature", "domain": "sensor", "area": "Garden"},
    {"name": "Grill Probe Temperature", "domain": "sensor", "area": "Garden"},
    {"name": "Room Thermostat", "domain": "climate", "area": "Living Room"},
]


def test_generic_temperature_question_is_read_only():
    assert ha_weather_routing._ambient_temperature_query("Welche Temperatur ist im Garten?")
    assert not ha_weather_routing._ambient_temperature_query(
        "Stelle das Thermostat im Garten auf 22 Grad"
    )


def test_outdoor_temperature_is_not_expanded_to_weather_summary():
    assert ha_weather_routing._ambient_temperature_query("Welche Temperatur ist draußen?")
    assert not ha_weather_routing._weather_query("Welche Temperatur ist draußen?")


def test_static_ranking_uses_generic_environment_semantics():
    sources = ha_weather_routing._temperature_sources(
        GENERIC_ENTITIES,
        "Welche Temperatur ist im Garden?",
    )
    assert sources[0]["name"] == "Outdoor Temperature"
    assert all("Inverter" not in item["name"] for item in sources[:1])


def test_initial_temperature_lookup_uses_area_and_domains_not_device_names():
    arguments = ha_weather_routing._initial_live_arguments(
        GENERIC_ENTITIES,
        "Welche Temperatur ist im Garden?",
        kind="temperature",
    )
    assert arguments == {
        "domain": ["sensor", "climate", "weather"],
        "area": "Garden",
    }


def test_climate_target_setpoint_is_never_current_temperature():
    entity = {
        "name": "Room Thermostat",
        "domain": "climate",
        "state": "heat",
        "area": "Living Room",
        "attributes": {
            "current_temperature": "",
            "temperature": "24",
            "unit_of_measurement": "°C",
        },
    }
    assert ha_weather_routing._temperature_value(entity) is None


def test_climate_current_temperature_can_be_used_even_when_control_is_off():
    entity = {
        "name": "Room Thermostat",
        "domain": "climate",
        "state": "off",
        "area": "Living Room",
        "attributes": {
            "current_temperature": "21.5",
            "temperature": "18",
            "unit_of_measurement": "°C",
        },
    }
    response = ha_weather_routing._temperature_response(
        [entity],
        "Welche Temperatur ist im Living Room?",
    )
    assert response == "Die aktuelle Temperatur beträgt 21,5 °C."
    assert "18" not in response


def test_temperature_prefers_ambient_sensor_over_process_sensor():
    entities = [
        {
            "name": "Outdoor Temperature",
            "domain": "sensor",
            "state": "12.4",
            "area": "Garden",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
        {
            "name": "Inverter Temperature",
            "domain": "sensor",
            "state": "46.2",
            "area": "Garden",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
        {
            "name": "Grill Probe Temperature",
            "domain": "sensor",
            "state": "82.0",
            "area": "Garden",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
    ]
    response = ha_weather_routing._temperature_response(
        entities,
        "Welche Temperatur ist im Garden?",
    )
    assert response == "Die aktuelle Temperatur beträgt 12,4 °C."


def test_weather_entity_is_preferred_when_available():
    entities = [
        {
            "name": "Local Weather",
            "domain": "weather",
            "state": "cloudy",
            "area": "",
            "attributes": {"temperature": "12.4", "humidity": "81", "temperature_unit": "°C"},
        },
        {
            "name": "Room Temperature",
            "domain": "sensor",
            "state": "22.0",
            "area": "Living Room",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
    ]
    response = ha_weather_routing._weather_response(entities, "Wie ist das Wetter draußen?")
    assert "bewölkt" in response
    assert "12,4 °C" in response
    assert "81 %" in response
    assert "22" not in response


def test_weather_sensor_group_uses_device_classes_generically():
    entities = [
        {
            "name": "Outdoor Temperature",
            "domain": "sensor",
            "state": "11.8",
            "area": "Garden",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
        {
            "name": "Outdoor Humidity",
            "domain": "sensor",
            "state": "77",
            "area": "Garden",
            "attributes": {"device_class": "humidity", "unit_of_measurement": "%"},
        },
        {
            "name": "Dew Point",
            "domain": "sensor",
            "state": "7.9",
            "area": "Garden",
            "attributes": {"device_class": "dew_point", "unit_of_measurement": "°C"},
        },
        {
            "name": "Room Temperature",
            "domain": "sensor",
            "state": "22.1",
            "area": "Living Room",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
    ]
    response = ha_weather_routing._weather_response(entities, "Wie ist das Wetter draußen?")
    assert "11,8 °C" in response
    assert "77 %" in response
    assert "7,9 °C" in response
    assert "22,1" not in response


def test_unavailable_environmental_data_reports_no_current_data():
    entities = [
        {
            "name": "Outdoor Temperature",
            "domain": "sensor",
            "state": "unavailable",
            "area": "Garden",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        }
    ]
    response = ha_weather_routing._weather_response(entities, "Wie ist das Wetter draußen?")
    assert response == "Aktuell sind die Wetterdaten in Home Assistant nicht verfügbar."


def test_multiple_weather_sources_remain_an_llm_decision():
    entities = [
        {"name": name, "domain": "weather", "state": "sunny", "area": area,
         "attributes": {"temperature": temperature, "temperature_unit": "°C"}}
        for name, area, temperature in [("North weather", "North", 12), ("South weather", "South", 18)]
    ]
    assert ha_weather_routing._weather_response(entities, "Wie ist das Wetter draußen?") is None


def test_unavailable_weather_source_does_not_hide_available_environment_sensor():
    entities = [
        {"name": "Weather", "domain": "weather", "state": "unavailable", "area": "Garden",
         "attributes": {"temperature": 99, "temperature_unit": "°C"}},
        {"name": "Ambient temperature", "domain": "sensor", "state": "12", "area": "Garden",
         "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"}},
    ]
    answer = ha_weather_routing._weather_response(entities, "Wie ist das Wetter draußen?")
    assert "12 °C" in answer
    assert "99" not in answer


def test_weather_initial_lookup_keeps_sensors_even_with_provider():
    arguments = ha_weather_routing._initial_live_arguments(
        [{"name": "Weather", "domain": "weather", "area": ""}],
        "Wie ist das Wetter draußen?", kind="weather",
    )
    assert arguments == {"domain": ["weather", "sensor"]}


def test_ambiguous_weather_prompt_preserves_provider_measurements():
    from hailo_services.schemas import ChatRequest

    request = ChatRequest(messages=[{"role": "user", "content": "Wie ist das Wetter?"}])
    entities = [
        {"name": "North", "domain": "weather", "state": "cloudy", "area": "",
         "attributes": {"temperature": 12, "humidity": 70, "temperature_unit": "°C"}},
        {"name": "South", "domain": "weather", "state": "sunny", "area": "",
         "attributes": {"temperature": 18, "humidity": 55, "temperature_unit": "°C"}},
    ]
    compact = ha_weather_routing._compact_environment_decision(request, entities, "Wetter?", "weather")
    prompt = compact.messages[-1]["content"]
    assert '"temperature": 12' in prompt
    assert '"temperature": 18' in prompt
    assert '"humidity": 70' in prompt
    assert compact.tools is None


def test_same_area_ambiguous_measurements_do_not_use_catalogue_order():
    entities = [
        {"name": name, "domain": "sensor", "state": value, "area": "Garden",
         "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"}}
        for name, value in [("Outdoor A", "12"), ("Outdoor B", "18")]
    ]
    for ordered in (entities, list(reversed(entities))):
        assert ha_weather_routing._weather_response(ordered, "Wie ist das Wetter draußen?") is None
