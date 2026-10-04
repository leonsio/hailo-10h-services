import os
from dataclasses import dataclass

VLM_MODEL = "Qwen2-VL-2B-Instruct"
STT_MODEL = "whisper-base"
LLM_MODEL = "gemma-4-E2B-it"


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8090
    wyoming_host: str = "0.0.0.0"
    wyoming_port: int = 10300
    api_key: str = ""
    vlm_hef: str = VLM_MODEL
    whisper_hef: str = "Whisper-Base"
    language: str = "de"
    queue_size: int = 8
    request_timeout: float = 180
    max_body: int = 16 * 1024 * 1024
    max_audio_seconds: int = 120
    debug_log: bool = False
    litert_model_path: str = ""
    litert_max_num_tokens: int = 16384
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_prefix: str = "hailo10h"
    mqtt_tls: bool = False
    mcp_no_auth_networks: str = "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,fc00::/7"
    mcp_hosts: str = "localhost:*,127.0.0.1:*"

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for key, value in vars(defaults).items():
            raw = os.getenv("HAILO_" + key.upper())
            if raw is not None:
                if isinstance(value, bool):
                    values[key] = raw.lower() in {"true", "yes", "1"}
                else:
                    values[key] = type(value)(raw)
        config = cls(**values)
        if config.queue_size < 1 or config.max_body < 1024 or config.request_timeout <= 0:
            raise ValueError("Invalid queue size, body limit or timeout")
        if not 0 <= config.wyoming_port <= 65535 or not 1 <= config.port <= 65535:
            raise ValueError("Invalid listener port")
        if config.max_audio_seconds < 1:
            raise ValueError("Audio duration limit must be positive")
        if not 2048 <= config.litert_max_num_tokens <= 131072:
            raise ValueError("LiteRT context must be between 2048 and 131072 tokens")
        return config
