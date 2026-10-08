"""Native Hailo LLM text prompt adapter, sharing budget and tool validation."""

import re

from hailo_services.chat import chat_common
from hailo_services.shared.tool_calling import has_tool_context


def model_prompt(request):
    """Build text-only Hailo LLM messages using the shared tool contract.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        list[dict[str, str]]: Native text messages with no image placeholders.

    Raises:
        ValueError: Images are present or tool-call history/selection is invalid.
    """
    if any(
        part.get("type") == "image_url"
        for message in request.messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    ):
        raise ValueError(
            f"{request.model} is a text-only Hailo LLM; select an enabled VLM for images"
        )
    # Reuse the compact, model-independent function contract/history validation,
    # then convert to the string content required by genai.LLM (no vision parts).
    return [
        {"role": message["role"], "content": "\n".join(part["text"] for part in message["content"])}
        for message in chat_common.model_prompt(request)
    ]


def limit_request(
    model, request, configured_limit, context_length, *, debug=False, template_options=None
):
    """Budget and render the exact raw string sent to native LLM generation.

    Args:
        model (Any): Native model exposing tokenization/template methods, or a catalogue identifier.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        configured_limit (int): Configured upper bound for input tokens.
        context_length (int): Compiled native model context length in tokens.
        debug (bool): Whether prompt and budget details are logged.
        template_options (dict[str, Any] | None): Catalogue options controlling native template rendering.

    Returns:
        tuple[ChatRequest, str]: Trimmed request and its measured rendered prompt.

    Raises:
        ValueError: Native prompt templating/tokenization is unavailable or invalid.
        InputBudgetError: Required text and tool context exceed the effective input budget.
    """
    if not callable(getattr(model, "prompt_template", None)):
        raise ValueError(
            "LLM input budgeting requires HailoRT LLM.prompt_template; upgrade HailoRT"
        )
    budget = chat_common.budget_request(
        model,
        request,
        configured_limit,
        context_length,
        debug=debug,
        prompt_builder=model_prompt,
        model_kind="LLM",
        template_options=template_options,
    )
    # LLM.generate accepts a raw string. Send exactly the template we measured,
    # including enable_thinking=False, without a second native template render.
    return budget.request, budget.rendered


def tool_response(text, request):
    """Normalize native tool JSON and validate client-owned function calls.

    Args:
        text (str): Text to parse, normalize, match or render.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        ChatResult: Plain text or validated assistant tool-call message.

    Raises:
        ValueError: Native function output cannot be parsed or validated.
    """
    if has_tool_context(request):
        # Hailo's function-calling HEFs may wrap otherwise valid JSON in XML.
        wrapped = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", text, re.DOTALL)
        if wrapped:
            text = wrapped.group(1)
    return chat_common.tool_response(text, request)
