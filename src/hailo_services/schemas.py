"""Pydantic request models and input validation for public APIs."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from hailo_services.config import FRIGATE_ASSIST_MODEL, VLM_MODEL
from hailo_services.shared.i18n import SUPPORTED_LANGUAGES

CatalogueText = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^\r\n]+$")]


class HAEntity(BaseModel):
    """Describe one entity explicitly exposed by the requesting HA client.

    Attributes:
        name: Canonical client tool target name.
        domain: Home Assistant integration domain.
        area: Canonical room name, or empty for an unassigned entity.
        entity_id: Optional stable identifier, retained for validation and diagnostics.
        aliases: Configured HA names, not generated spelling variants.
        area_aliases: Configured room aliases.
        floor: Optional floor name.
        device_class: Optional HA device class.
        capabilities: Supported writable properties; None means not supplied.
    """

    model_config = ConfigDict(extra="forbid")
    name: CatalogueText
    domain: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    area: str = Field(default="", max_length=256, pattern=r"^[^\r\n]*$")
    entity_id: str | None = Field(
        default=None, max_length=256, pattern=r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$"
    )
    aliases: list[CatalogueText] = Field(default_factory=list, max_length=32)
    area_aliases: list[CatalogueText] = Field(default_factory=list, max_length=32)
    floor: CatalogueText | None = None
    device_class: CatalogueText | None = None
    capabilities: (
        list[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]] | None
    ) = Field(default=None, max_length=32)


class HAContext(BaseModel):
    """Carry an optional versioned exposed catalogue without cached live values.

    Attributes:
        version: Caller-supplied catalogue revision; content remains authoritative.
        entities: Only entities exposed to this Assist conversation.
    """

    model_config = ConfigDict(extra="forbid")
    version: str | None = Field(default=None, max_length=128)
    entities: list[HAEntity] = Field(max_length=4096)


class StreamOptions(BaseModel):
    """Validate Frigate's optional OpenAI usage-stream request."""

    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatRequest(BaseModel):
    """Validate OpenAI chat input, tool policy and generation limits.

    Attributes:
        model (str): Native model exposing tokenization/template methods, or a catalogue identifier. Default: VLM_MODEL.
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        max_tokens (int): Maximum generated tokens reserved for the response.
        max_input_tokens (int | None): Configured maximum input-token count before generation.
        temperature (float): Temperature.
        top_p (float | None): Top p.
        seed (int): Seed.
        stream (bool): Stream. Default: False.
        stream_options (StreamOptions | None): Usage-stream options for Frigate-Assist only.
        user (str | None): User. Default: None.
        language (str | None): Language code; None uses the configured or detected language.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.
        tool_choice (str | dict[str, Any] | None): Tool choice. Default: None.
        ha_context (HAContext | None): Optional exposed metadata used exclusively by HA-Assist.
        parallel_tool_calls (bool): Whether multi-target output may expand into parallel function calls. Default: True.
    """

    model_config = ConfigDict(extra="forbid")
    _request_id: str = PrivateAttr(default="-")
    _metrics: dict[str, Any] = PrivateAttr(default_factory=dict)
    model: str = VLM_MODEL
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    max_input_tokens: int | None = Field(default=None, ge=1, le=131072)
    temperature: float = Field(default=0.1, ge=0, le=1)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None
    language: str | None = Field(
        default=None, pattern=rf"^({'|'.join(SUPPORTED_LANGUAGES)})(?:-[A-Za-z]{{2}})?$"
    )
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    ha_context: HAContext | None = None

    @model_validator(mode="after")
    def check_stream_options(self):
        """Keep Frigate compatibility options isolated from native chat models.

        Returns:
            ChatRequest: Validated request.

        Raises:
            ValueError: Stream options were supplied for a different model.
        """
        if self.stream_options is not None and self.model != FRIGATE_ASSIST_MODEL:
            raise ValueError("stream_options is supported only by Frigate-Assist")
        return self

    @field_validator("tools")
    @classmethod
    def check_tools(cls, tools):
        """Validate OpenAI function definitions before accepting a chat request.

        Args:
            tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

        Returns:
            list[dict[str, Any]] | None: Original validated tools.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        from hailo_services.shared.tool_calling import validate_tools

        validate_tools(tools or [])
        return tools

    @field_validator("max_input_tokens", mode="before")
    @classmethod
    def check_max_input_tokens(cls, value):
        # Local OpenAI LLM renders custom request-body values as template text.
        # Accept a decimal string such as "4096", but reject bools/floats.
        """Accept integer input limits and decimal template strings.

        Args:
            value (Any): Input value inspected or normalized by this helper.

        Returns:
            int | None: Parsed integer limit, or None.

        Raises:
            ValueError: max_input_tokens must be an integer.
        """
        if isinstance(value, bool):
            raise ValueError("max_input_tokens must be an integer")
        if isinstance(value, str):
            if not value.isdecimal():
                raise ValueError("max_input_tokens must be an integer")
            return int(value)
        if value is not None and not isinstance(value, int):
            raise ValueError("max_input_tokens must be an integer")
        return value

    @field_validator("tool_choice")
    @classmethod
    def check_tool_choice(cls, choice):
        """Validate automatic, disabled, required or forced function selection.

        Args:
            choice (str | dict[str, Any] | None): OpenAI automatic, disabled, required or forced function choice.

        Returns:
            str | dict[str, Any] | None: Original validated choice.

        Raises:
            ValueError: Unsupported tool_choice.
        """
        if choice is None or choice in ("auto", "none", "required"):
            return choice
        if (
            isinstance(choice, dict)
            and choice.get("type") == "function"
            and isinstance(choice.get("function"), dict)
            and isinstance(choice["function"].get("name"), str)
        ):
            return choice
        raise ValueError("Unsupported tool_choice")

    @field_validator("messages")
    @classmethod
    def check_messages(cls, messages):
        """Validate roles, content parts and historical function-call shapes.

        Args:
            messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

        Returns:
            list[dict[str, Any]]: Original validated message history.

        Raises:
            ValueError: Only text and base64 image_url parts are supported.
        """
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Only system/user/assistant/tool messages are supported")
            content = message.get("content")
            if message.get("tool_calls") and message["role"] != "assistant":
                raise ValueError("Only assistant messages may contain tool_calls")
            if message.get("role") == "assistant" and message.get("tool_calls"):
                from hailo_services.shared.tool_calling import validate_history_calls

                validate_history_calls(message["tool_calls"])
                if content is None:
                    continue
            if message.get("role") == "tool":
                if not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]:
                    raise ValueError("Tool messages require tool_call_id")
            if isinstance(content, str):
                continue
            if not isinstance(content, list) or not content:
                raise ValueError("Message content must be text or a non-empty content list")
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("Content parts must be objects")
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    continue
                image = item.get("image_url")
                if (
                    item.get("type") == "image_url"
                    and isinstance(image, dict)
                    and isinstance(image.get("url"), str)
                ):
                    continue
                raise ValueError("Only text and base64 image_url parts are supported")
        return messages


class TranscribeRequest(BaseModel):
    """Validate the JSON audio-transcription envelope.

    Attributes:
        audio_base64 (str): Base64-encoded recording or audio data URL.
        language (str | None): Language code; None uses the configured or detected language.
    """

    model_config = ConfigDict(extra="forbid")
    audio_base64: str
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")


class VisionDetectRequest(BaseModel):
    """OpenAI-style JSON envelope for resident object detection.

    Attributes:
        model (str | None): Native model exposing tokenization/template methods, or a catalogue identifier. Default: None.
        image (str): Image.
        confidence (float | None): Detection score threshold; None uses service settings.
        max_detections (int | None): Max detections.
    """

    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    image: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    max_detections: int | None = Field(default=None, ge=1, le=100)


class SpeechRequest(BaseModel):
    """Validate CPU speech synthesis input and supported audio formats.

    Attributes:
        model (str): Public speech backend identifier.
        input (str): Text to speak, bounded again by the configured service limit.
        voice (str | None): Installed voice ID; omission uses the configured default.
        language (str | None): Optional expected language; must match the selected voice.
        response_format (str): WAV at the voice rate or mono PCM16LE at 24 kHz.
        speed (float): Speaking speed, mapped to inverse Piper length_scale.
    """

    model_config = ConfigDict(extra="forbid")
    model: Literal["piper"] = "piper"
    input: str = Field(min_length=1, max_length=16384)
    voice: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}(?:[-_][A-Z]{2})?$")
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0, allow_inf_nan=False)

    @field_validator("input")
    @classmethod
    def check_text(cls, value):
        """Reject blank text before allocating synthesis work.

        Args:
            value (str): Text supplied by the client.

        Returns:
            str: Original nonblank text.
        """
        if not value.strip():
            raise ValueError("Speech input must not be blank")
        return value
