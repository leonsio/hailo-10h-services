from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import VLM_MODEL


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["Qwen2-VL-2B-Instruct"] = VLM_MODEL
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.1, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def check_messages(cls, messages):
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Only system/user/assistant messages are supported")
            content = message.get("content")
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
