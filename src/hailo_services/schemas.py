from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import VLM_MODEL


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = VLM_MODEL
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.1, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    stream: bool = False
    user: str | None = None
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True

    @field_validator("tools")
    @classmethod
    def check_tools(cls, tools):
        from .tool_calling import validate_tools

        validate_tools(tools or [])
        return tools

    @field_validator("tool_choice")
    @classmethod
    def check_tool_choice(cls, choice):
        if choice is None or choice in ("auto", "none", "required"):
            return choice
        if (isinstance(choice, dict) and choice.get("type") == "function"
                and isinstance(choice.get("function"), dict)
                and isinstance(choice["function"].get("name"), str)):
            return choice
        raise ValueError("Unsupported tool_choice")

    @field_validator("messages")
    @classmethod
    def check_messages(cls, messages):
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Only system/user/assistant/tool messages are supported")
            content = message.get("content")
            if message.get("tool_calls") and message["role"] != "assistant":
                raise ValueError("Only assistant messages may contain tool_calls")
            if message.get("role") == "assistant" and message.get("tool_calls"):
                from .tool_calling import validate_history_calls

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
    model_config = ConfigDict(extra="forbid")
    audio_base64: str
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
