import os
from dataclasses import dataclass
from pathlib import Path

import yaml

VLM_MODEL = "Qwen2-VL-2B-Instruct"
STT_MODEL = "whisper-base"
LLM_MODEL = "gemma-4-E2B-it"
HA_ASSIST_MODEL = "HA-Assist"


@dataclass(frozen=True)
class Settings:
    model_catalog: str = ""
    model_store: str = "/usr/local/hailo/resources/models/hailo10h"
    model_release: str = "auto"
    vlm_release: str = "auto"
    hailo_llm_release: str = "auto"
    vlm_max_input_tokens: int = 2048
    hailo_llm_max_input_tokens: int = 2048
    vlm_enabled: bool = True
    whisper_enabled: bool = True
    minilm_enabled: bool = True
    hailo_llm_enabled: bool = False
    hailo_llm_model: str = "Qwen2.5-1.5B-Instruct"
    litert_enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8090
    wyoming_host: str = "0.0.0.0"
    wyoming_port: int = 10300
    api_key: str = ""
    vlm_hef: str = VLM_MODEL
    whisper_hef: str = "Whisper-Base"
    language: str = "de"
    service_language: str = "de"
    ha_assist_enabled: bool = True
    ha_assist_text_model: str = LLM_MODEL
    ha_assist_vision_model: str = VLM_MODEL
    ha_assist_fuzzy_enabled: bool = True
    ha_assist_fuzzy_threshold: float = 90.0
    ha_assist_fuzzy_margin: float = 8.0
    ha_wait_messages: bool = True
    queue_size: int = 8
    request_timeout: float = 180
    max_body: int = 16 * 1024 * 1024
    max_audio_seconds: int = 120
    debug_log: bool = False
    litert_model_path: str = ""
    litert_max_num_tokens: int = 16384
    litert_max_input_tokens: int = 4096
    minilm_hef_path: str = ""
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_prefix: str = "hailo10h"
    mqtt_tls: bool = False
    mcp_no_auth_networks: str = "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,fc00::/7"
    mcp_hosts: str = "localhost:*,127.0.0.1:*"

    def __post_init__(self):
        if HA_ASSIST_MODEL in {self.ha_assist_text_model, self.ha_assist_vision_model}:
            raise ValueError("HA-Assist cannot route to itself")
        if not self.ha_assist_text_model.strip() or not self.ha_assist_vision_model.strip():
            raise ValueError("HA-Assist target model IDs must not be empty")
        if not 80 <= self.ha_assist_fuzzy_threshold <= 100 or not 0 < self.ha_assist_fuzzy_margin <= 100:
            raise ValueError("Invalid HA-Assist fuzzy threshold or margin")
        for name in ("vlm_max_input_tokens", "hailo_llm_max_input_tokens"):
            if not 1 <= getattr(self, name) <= 2048:
                raise ValueError(f"{name} must be between 1 and the compiled HEF limit of 2048")

    @property
    def vlm_model(self):
        return Path(self.vlm_hef).stem

    @property
    def stt_model(self):
        return Path(self.whisper_hef).stem.lower()

    @property
    def hailo_llm_model_id(self):
        name = Path(self.hailo_llm_model).name
        return name[:-4] if name.endswith(".hef") else name

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        config_path = os.getenv("HAILO_CONFIG")
        if config_path:
            path = Path(config_path)
        else:
            path = Path("/etc/hailo-10h-services.yaml")
        if config_path or path.exists():
            with path.open(encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
            if not isinstance(document, dict) or set(document) - {"settings", "models"}:
                raise ValueError("YAML config must contain only settings and models mappings")
            values = document.get("settings", {}) or {}
            if not isinstance(values, dict) or set(values) - set(vars(defaults)):
                raise ValueError("Unknown setting in YAML configuration")
            values = dict(values)
            selections = document.get("models", {}) or {}
            roles = {
                "vlm": ("vlm_enabled", "vlm_hef"),
                "whisper": ("whisper_enabled", "whisper_hef"),
                "minilm": ("minilm_enabled", None),
                "hailo_llm": ("hailo_llm_enabled", "hailo_llm_model"),
                "gemma": ("litert_enabled", None),
            }
            if not isinstance(selections, dict) or set(selections) - set(roles):
                raise ValueError("Unknown model role in YAML configuration")
            for role, selection in selections.items():
                if not isinstance(selection, dict) or set(selection) - {"enabled", "model", "path", "max_input_tokens", "release"}:
                    raise ValueError(f"Invalid models.{role} configuration")
                limit_key = {"vlm": "vlm_max_input_tokens", "gemma": "litert_max_input_tokens",
                             "hailo_llm": "hailo_llm_max_input_tokens"}.get(role)
                if "max_input_tokens" in selection:
                    if limit_key is None:
                        raise ValueError(f"models.{role} does not allow max_input_tokens")
                    values[limit_key] = selection["max_input_tokens"]
                if "release" in selection:
                    release_key = {"vlm": "vlm_release", "hailo_llm": "hailo_llm_release"}.get(role)
                    if release_key is None:
                        raise ValueError(f"models.{role} does not allow release")
                    values[release_key] = selection["release"]
                enabled, model_key = roles[role]
                if "enabled" in selection:
                    values[enabled] = selection["enabled"]
                if "model" in selection:
                    if model_key is None:
                        raise ValueError(f"models.{role} does not allow model selection")
                    values[model_key] = selection["model"]
                if "path" in selection:
                    path_key = {"gemma": "litert_model_path", "minilm": "minilm_hef_path",
                                "vlm": "vlm_hef", "whisper": "whisper_hef"}.get(role)
                    if path_key is None:
                        raise ValueError(f"models.{role} does not allow a path override")
                    values[path_key] = selection["path"]
            # Explicit false wins over a legacy model path.
            if selections.get("gemma", {}).get("enabled") is False:
                values["litert_model_path"] = ""
        for key, value in vars(defaults).items():
            raw = os.getenv("HAILO_" + key.upper())
            if raw is not None:
                if isinstance(value, bool):
                    values[key] = raw.lower() in {"true", "yes", "1"}
                else:
                    values[key] = type(value)(raw)
        for key, value in values.items():
            default = getattr(defaults, key)
            if isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid = type(value) is type(default)
            if not valid:
                raise ValueError(f"Invalid type for setting {key}: expected {type(default).__name__}")
        config = cls(**values)
        if config.service_language not in {"de", "en", "ru"}:
            raise ValueError("Service language must be de, en or ru")
        if config.queue_size < 1 or config.max_body < 1024 or config.request_timeout <= 0:
            raise ValueError("Invalid queue size, body limit or timeout")
        if not 0 <= config.wyoming_port <= 65535 or not 1 <= config.port <= 65535:
            raise ValueError("Invalid listener port")
        if config.max_audio_seconds < 1:
            raise ValueError("Audio duration limit must be positive")
        if not 2048 <= config.litert_max_num_tokens <= 131072:
            raise ValueError("LiteRT context must be between 2048 and 131072 tokens")
        if not 512 <= config.litert_max_input_tokens <= 131072:
            raise ValueError("LiteRT input limit must be between 512 and 131072 tokens")
        return config
