"""Validated settings with YAML input and environment overrides."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

VLM_MODEL = "Qwen2-VL-2B-Instruct"
STT_MODEL = "whisper-base"
LLM_MODEL = "gemma-4-E2B-it"
HA_ASSIST_MODEL = "HA-Assist"
VISION_MODEL = "yolov11m"


@dataclass(frozen=True)
class Settings:
    """Immutable typed service settings loaded from YAML and environment overrides.

    Attributes:
        model_catalog (str): Optional YAML catalogue path; empty uses the bundled catalogue.
        model_store (str): Directory for cached HEF, tokenizer and model artifacts.
        model_release (str): Global model-zoo release override; auto selects a compatible release.
        vlm_release (str): VLM-specific release override taking precedence over the global selection.
        hailo_llm_release (str): Hailo LLM release override taking precedence over the global selection.
        vision_release (str): Object-detector release override taking precedence over the global selection.
        vlm_max_input_tokens (int): Maximum VLM input tokens, including conservative visual/template reserves.
        hailo_llm_max_input_tokens (int): Maximum native Hailo LLM input tokens, bounded by the compiled HEF.
        vlm_enabled (bool): Whether to load the resident Hailo vision-language model.
        whisper_enabled (bool): Whether to load the resident Hailo speech-recognition model.
        minilm_enabled (bool): Whether to load the resident semantic encoder for HA retrieval.
        hailo_llm_enabled (bool): Whether to load a native resident Hailo text-only LLM.
        hailo_llm_model (str): Catalogue ID or local HEF path of the native text model.
        vision_enabled (bool): Whether the resident object-detection service is enabled.
        vision_model (str): Catalogue ID or local HEF path of the selected object detector.
        vision_zmq_enabled (bool): Whether to expose the Frigate-compatible detector bridge.
        vision_zmq_endpoint (str): ZeroMQ REP endpoint for model probes and RGB detector tensors.
        vision_confidence (float): Default minimum detection confidence for object results.
        vision_iou_threshold (float): Validated overlap threshold reserved for detector configuration; current native decoders do not apply host IoU suppression.
        vision_max_detections (int): Maximum returned object-detection rows for HTTP requests.
        vision_queue_size (int): Maximum pending requests per chat/speech backend queue.
        vision_scheduler_priority (int): Hailo scheduler priority for the resident object detector.
        litert_enabled (bool): Whether to initialize the CPU LiteRT-LM backend.
        host (str): HTTP listener bind address.
        port (int): HTTP API and UI listener port.
        wyoming_host (str): Wyoming speech listener bind address.
        wyoming_port (int): Wyoming listener port; zero disables the listener.
        api_key (str): Bearer credential for protected API requests; empty disables key checks.
        vlm_hef (str): Catalogue ID or local HEF path of the vision-language model.
        whisper_hef (str): Catalogue ID or local HEF path of the speech model.
        language (str): Default Whisper transcription language.
        service_language (str): Fallback language for HA routing, generated replies and notifications.
        ha_assist_enabled (bool): Whether to expose the virtual HA-Assist model.
        ha_assist_text_model (str): Enabled text backend selected for HA-Assist generative text fallback.
        ha_assist_vision_model (str): Enabled VLM backend selected for HA-Assist image requests.
        ha_assist_fuzzy_enabled (bool): Whether conservative catalogue spelling repair is enabled.
        ha_assist_fuzzy_threshold (float): Minimum character similarity score for a catalogue slot repair.
        ha_assist_fuzzy_margin (float): Minimum separation from the runner-up slot candidate.
        ha_wait_messages (bool): Whether HA-Assist streams localized pre-inference wait notifications.
        queue_size (int): Queue size. Default: 8.
        request_timeout (float): Client inference deadline in seconds; native work remains shielded.
        max_body (int): Maximum encoded request or media payload size in bytes.
        max_audio_seconds (int): Maximum accepted audio-recording duration in seconds.
        debug_log (bool): Whether to log detailed prompts, retrieval traces and protocol events.
        litert_model_path (str): Local LiteRT model path; a non-empty path also enables the backend.
        litert_max_num_tokens (int): Total native LiteRT context capacity; larger values increase RAM use.
        litert_max_input_tokens (int): Maximum LiteRT input tokens, including tools and template headroom.
        minilm_hef_path (str): Optional local MiniLM encoder HEF path.
        mqtt_host (str): MQTT broker hostname; empty disables the bridge.
        mqtt_port (int): MQTT broker port.
        mqtt_username (str): Optional MQTT authentication username.
        mqtt_password (str): Optional MQTT authentication password.
        mqtt_prefix (str): Topic prefix for status, request and correlated response messages.
        mqtt_tls (bool): Whether to verify and encrypt broker connections using system trust.
        mcp_no_auth_networks (str): CIDR ranges whose actual socket peers may access MCP without a key.
        mcp_hosts (str): Comma-separated allowed MCP Host patterns for transport validation.
    """

    model_catalog: str = ""
    model_store: str = "/usr/local/hailo/resources/models/hailo10h"
    model_release: str = "auto"
    vlm_release: str = "auto"
    hailo_llm_release: str = "auto"
    vision_release: str = "auto"
    vlm_max_input_tokens: int = 2048
    hailo_llm_max_input_tokens: int = 2048
    vlm_enabled: bool = True
    whisper_enabled: bool = True
    minilm_enabled: bool = True
    hailo_llm_enabled: bool = False
    hailo_llm_model: str = "Qwen2.5-1.5B-Instruct"
    vision_enabled: bool = False
    vision_model: str = VISION_MODEL
    vision_zmq_enabled: bool = True
    vision_zmq_endpoint: str = "tcp://127.0.0.1:5555"
    vision_confidence: float = 0.4
    vision_iou_threshold: float = 0.45
    vision_max_detections: int = 20
    vision_queue_size: int = 16
    vision_scheduler_priority: int = 1
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
    mcp_no_auth_networks: str = (
        "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,fc00::/7"
    )
    mcp_hosts: str = "localhost:*,127.0.0.1:*"

    def __post_init__(self):
        """Validate model routes, input limits and detector configuration.

        Returns:
            None: Accepts a valid immutable settings instance.

        Raises:
            ValueError: HA-Assist cannot route to itself.
        """
        if HA_ASSIST_MODEL in {self.ha_assist_text_model, self.ha_assist_vision_model}:
            raise ValueError("HA-Assist cannot route to itself")
        if not self.ha_assist_text_model.strip() or not self.ha_assist_vision_model.strip():
            raise ValueError("HA-Assist target model IDs must not be empty")
        if (
            not 80 <= self.ha_assist_fuzzy_threshold <= 100
            or not 0 < self.ha_assist_fuzzy_margin <= 100
        ):
            raise ValueError("Invalid HA-Assist fuzzy threshold or margin")
        for name in ("vlm_max_input_tokens", "hailo_llm_max_input_tokens"):
            if not 1 <= getattr(self, name) <= 2048:
                raise ValueError(f"{name} must be between 1 and the compiled HEF limit of 2048")
        if not self.vision_model.strip():
            raise ValueError("Vision model must not be empty")
        if not 0 <= self.vision_confidence <= 1 or not 0 < self.vision_iou_threshold <= 1:
            raise ValueError("Vision confidence/IoU thresholds must be between 0 and 1")
        if not 1 <= self.vision_max_detections <= 100:
            raise ValueError("vision_max_detections must be between 1 and 100")
        if self.vision_queue_size < 1:
            raise ValueError("vision_queue_size must be positive")
        if not 0 <= self.vision_scheduler_priority <= 255:
            raise ValueError("vision_scheduler_priority must be between 0 and 255")
        if self.vision_zmq_enabled and not (
            self.vision_zmq_endpoint.startswith("tcp://")
            or self.vision_zmq_endpoint.startswith("ipc://")
        ):
            raise ValueError("vision_zmq_endpoint must use tcp:// or ipc://")

    @property
    def vlm_model(self):
        """Derive the public VLM identifier from its configured HEF selection.

        Returns:
            str: Selected VLM model identifier.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return Path(self.vlm_hef).stem

    @property
    def stt_model(self):
        """Derive the public Whisper identifier from its configured HEF selection.

        Returns:
            str: Selected speech model identifier.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return Path(self.whisper_hef).stem.lower()

    @property
    def hailo_llm_model_id(self):
        """Derive the public Hailo LLM identifier from its configured selection.

        Returns:
            str: Selected text model identifier.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        name = Path(self.hailo_llm_model).name
        return name[:-4] if name.endswith(".hef") else name

    @property
    def vision_model_id(self):
        """Derive the public object detector identifier from its configured selection.

        Returns:
            str: Selected detector model identifier.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        name = Path(self.vision_model).name
        return name[:-4] if name.endswith(".hef") else name

    @classmethod
    def from_env(cls):
        """Load typed settings from YAML and environment overrides.

        Returns:
            Settings: Validated immutable service settings.

        Raises:
            ValueError: Invalid vision.zmq configuration.
        """
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
            if not isinstance(document, dict) or set(document) - {"settings", "models", "vision"}:
                raise ValueError(
                    "YAML config must contain only settings, models and vision mappings"
                )
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
                # Kept for backwards compatibility. New configurations should use
                # the dedicated top-level vision: block below.
                "vision": ("vision_enabled", "vision_model"),
                "gemma": ("litert_enabled", None),
            }
            if not isinstance(selections, dict) or set(selections) - set(roles):
                raise ValueError("Unknown model role in YAML configuration")
            for role, selection in selections.items():
                if not isinstance(selection, dict) or set(selection) - {
                    "enabled",
                    "model",
                    "path",
                    "max_input_tokens",
                    "release",
                }:
                    raise ValueError(f"Invalid models.{role} configuration")
                limit_key = {
                    "vlm": "vlm_max_input_tokens",
                    "gemma": "litert_max_input_tokens",
                    "hailo_llm": "hailo_llm_max_input_tokens",
                }.get(role)
                if "max_input_tokens" in selection:
                    if limit_key is None:
                        raise ValueError(f"models.{role} does not allow max_input_tokens")
                    values[limit_key] = selection["max_input_tokens"]
                if "release" in selection:
                    release_key = {
                        "vlm": "vlm_release",
                        "hailo_llm": "hailo_llm_release",
                        "vision": "vision_release",
                    }.get(role)
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
                    path_key = {
                        "gemma": "litert_model_path",
                        "minilm": "minilm_hef_path",
                        "vlm": "vlm_hef",
                        "whisper": "whisper_hef",
                        "vision": "vision_model",
                    }.get(role)
                    if path_key is None:
                        raise ValueError(f"models.{role} does not allow a path override")
                    values[path_key] = selection["path"]
            # Explicit false wins over a legacy model path.
            if selections.get("gemma", {}).get("enabled") is False:
                values["litert_model_path"] = ""

            vision = document.get("vision", {}) or {}
            if vision and "vision" in selections:
                raise ValueError(
                    "Configure vision either in top-level vision or legacy models.vision, not both"
                )
            allowed_vision = {
                "enabled",
                "model",
                "path",
                "release",
                "confidence",
                "iou_threshold",
                "max_detections",
                "queue_size",
                "scheduler_priority",
                "zmq",
            }
            if not isinstance(vision, dict) or set(vision) - allowed_vision:
                raise ValueError("Invalid top-level vision configuration")
            if "model" in vision and "path" in vision:
                raise ValueError("vision.model and vision.path are mutually exclusive")
            vision_mapping = {
                "enabled": "vision_enabled",
                "model": "vision_model",
                "path": "vision_model",
                "release": "vision_release",
                "confidence": "vision_confidence",
                "iou_threshold": "vision_iou_threshold",
                "max_detections": "vision_max_detections",
                "queue_size": "vision_queue_size",
                "scheduler_priority": "vision_scheduler_priority",
            }
            for key, target in vision_mapping.items():
                if key in vision:
                    values[target] = vision[key]
            zmq = vision.get("zmq", {}) or {}
            if not isinstance(zmq, dict) or set(zmq) - {"enabled", "endpoint"}:
                raise ValueError("Invalid vision.zmq configuration")
            if "enabled" in zmq:
                values["vision_zmq_enabled"] = zmq["enabled"]
            if "endpoint" in zmq:
                values["vision_zmq_endpoint"] = zmq["endpoint"]
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
                raise ValueError(
                    f"Invalid type for setting {key}: expected {type(default).__name__}"
                )
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
