"""Compatibility imports; new code should use chat_hailo_vlm."""

from .chat_hailo_vlm import limit_request, model_prompt, render_prompt, tool_response

__all__ = ["limit_request", "model_prompt", "render_prompt", "tool_response"]
