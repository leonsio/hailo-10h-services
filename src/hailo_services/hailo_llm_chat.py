"""Native Hailo LLM text prompt adapter, sharing budget and tool validation."""

import re

from . import vlm_chat
from .tool_calling import has_tool_context


def model_prompt(request):
    if any(part.get("type") == "image_url" for message in request.messages
           for part in (message.get("content") if isinstance(message.get("content"), list) else [])):
        raise ValueError(f"{request.model} is a text-only Hailo LLM; select an enabled VLM for images")
    # Reuse the compact, model-independent function contract/history validation,
    # then convert to the string content required by genai.LLM (no vision parts).
    return [{"role": message["role"], "content": "\n".join(
        part["text"] for part in message["content"]
    )} for message in vlm_chat.model_prompt(request)]


def limit_request(model, request, configured_limit, context_length, *, debug=False,
                  template_options=None):
    if not callable(getattr(model, "prompt_template", None)):
        raise ValueError("LLM input budgeting requires HailoRT LLM.prompt_template; upgrade HailoRT")
    trimmed, prompt = vlm_chat.limit_request(
        model, request, configured_limit, context_length, debug=debug,
        prompt_builder=model_prompt, model_kind="LLM", template_options=template_options,
    )
    # LLM.generate accepts a raw string. Send exactly the template we measured,
    # including enable_thinking=False, without a second native template render.
    return trimmed, vlm_chat.render_prompt(
        model, prompt, model_kind="LLM", template_options=template_options,
    )


def tool_response(text, request):
    if has_tool_context(request):
        # Hailo's function-calling HEFs may wrap otherwise valid JSON in XML.
        wrapped = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", text, re.DOTALL)
        if wrapped:
            text = wrapped.group(1)
    return vlm_chat.tool_response(text, request)
