"""Compatibility imports; new code should use chat_hailo_llm."""

from .chat_hailo_llm import limit_request, model_prompt, tool_response

__all__ = ["limit_request", "model_prompt", "tool_response"]
