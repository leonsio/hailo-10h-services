"""Compile Frigate prompts without carrying its large generic instruction envelope."""

import copy
import json
import re

from hailo_services.config import LLM_MODEL
from hailo_services.shared.i18n import catalogue, detect_language, language_code, lexicon, translate
from hailo_services.shared.tool_calling import native_messages

TOOL_HINTS = catalogue("en")["frigate"]["tool_hints"]


def has_images(request):
    """Detect incoming image parts without importing Home Assistant routing.

    Args:
        request: Client chat request.

    Returns:
        bool: Whether an incoming message contains an image.
    """
    return any(
        part.get("type") == "image_url"
        for message in request.messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    )


def uses_images(request):
    """Choose vision for a fresh image or an explicit reference to a previous frame.

    Args:
        request: Incoming Frigate request, including historical messages.

    Returns:
        bool: Whether this turn needs image observation rather than text reasoning.
    """
    users = [m for m in request.messages if m.get("role") == "user"]
    if not users or not has_images(request):
        return False
    latest = users[-1]
    if has_images(request.model_copy(update={"messages": [latest]})):
        return True
    question = text_content(latest)
    # An attached event identifies a different frame; never reuse an unrelated live image.
    if "[attached_event:" in question or re.search(
        r"last (?:seen|detected)|zuletzt (?:gesehen|erkannt)", question, re.I
    ):
        return False
    return bool(
        re.search(
            lexicon("frigate_prompt.pattern.62.12"),
            question,
            re.I,
        )
    )


def text_content(message):
    """Read text parts without including image URLs.

    Args:
        message: OpenAI message.

    Returns:
        str: Textual message content.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content or [] if part.get("type") == "text")


def camera_catalogue(messages):
    """Extract only camera/zone mappings explicitly supplied by Frigate.

    Args:
        messages: Original client messages.

    Returns:
        dict[str, str]: Exact camera IDs mapped to friendly names.
    """
    cameras = {}
    for message in messages:
        if message.get("role") != "system":
            continue
        for line in text_content(message).splitlines():
            match = re.match(r"\s*-\s*(.+?)\s*\(ID:\s*([^,()]+)(?:,.*)?\)\s*$", line)
            if match:
                cameras[match[2].strip()] = match[1].strip()
    return cameras


def server_time(messages):
    """Extract Frigate's authoritative local clock, never the proxy clock.

    Args:
        messages: Original client messages.

    Returns:
        str: Supplied time string, or empty when absent.
    """
    for message in reversed(messages):
        if message.get("role") == "system":
            match = re.search(lexicon("frigate_prompt.pattern.117.16"), text_content(message))
            if match:
                return match[1].strip()
    return ""


def compact_data(value, max_events, depth=0):
    """Bound result payloads while preserving errors, local timestamps and omission markers.

    Args:
        value: Decoded tool data.
        max_events: Maximum records per list.
        depth: Current nesting depth.

    Returns:
        object: Compact data with explicit markers for omitted material.
    """
    if depth > 6:
        return {"_omitted": "nested data exceeds depth limit"}
    if isinstance(value, list):
        kept = [compact_data(item, max_events, depth + 1) for item in value[:max_events]]
        if len(value) > max_events:
            kept.append({"_omitted_records": len(value) - max_events, "_total_records": len(value)})
        return kept
    if isinstance(value, dict):
        noisy = {
            "thumbnail",
            "thumbnail_path",
            "box",
            "region",
            "pixels",
            "embedding",
            "_image_url",
        }
        return {
            key: compact_data(item, max_events, depth + 1)
            for key, item in value.items()
            if key not in noisy
        }
    if isinstance(value, str) and len(value) > 600:
        return value[:600] + " [text omitted]"
    return value


def compact_tool(tool, language=None):
    """Remove prose from a tool schema without weakening validation constraints.

    Args:
        tool: Client-provided tool declaration.
        language: Optional request language for model-facing resource text.

    Returns:
        dict: Independent compact declaration.
    """
    result = copy.deepcopy(tool)
    function = result["function"]
    function["description"] = catalogue(language_code(language))["frigate"]["tool_hints"].get(
        function["name"], function.get("description", "")[:160]
    )

    def clean(node):
        """Strip schema annotations while retaining all semantic constraints.

        Args:
            node: Nested JSON Schema node.

        Returns:
            None: Modifies the copied node in place.
        """
        if isinstance(node, dict):
            for key in ("description", "title", "examples"):
                node.pop(key, None)
            # Property names and enum/default data are not schema annotations.
            # A parameter called "description" must remain a permitted field.
            for key in (
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
            ):
                mapping = node.get(key)
                if isinstance(mapping, dict):
                    for schema in mapping.values():
                        clean(schema)
            for key in (
                "items",
                "additionalProperties",
                "unevaluatedProperties",
                "unevaluatedItems",
                "contains",
                "propertyNames",
                "not",
                "if",
                "then",
                "else",
                "allOf",
                "anyOf",
                "oneOf",
                "prefixItems",
            ):
                clean(node.get(key))
        elif isinstance(node, list):
            for item in node:
                clean(item)

    clean(function.get("parameters", {}))
    return result


def compile_request(request, settings, tools, images, context=None):
    """Retain the active tool round for text; turn vision into a focused observation task.

    Args:
        request: Original Frigate request.
        settings: Service settings.
        tools: Preselected tool declarations.
        images: Whether this turn requires image observation.
        context: Deterministically resolved request facts, without changing user constraints.

    Returns:
        ChatRequest: Compact native-backend request sharing request metrics.

    Raises:
        ValueError: Required text/constraints cannot fit without changing their meaning.
    """
    cameras = camera_catalogue(request.messages)
    clock = server_time(request.messages)
    # Validate IDs/dependencies even when vision later removes tool history.
    text_history = [
        {**message, "content": text_content(message)}
        if isinstance(message.get("content"), list)
        else message
        for message in request.messages
    ]
    native_messages(text_history)
    if images:
        questions = []
        for message in request.messages:
            if message.get("role") == "user":
                question = text_content(message).strip()
                if question:
                    questions.append(question)
        # The latest real question and Frigate's subsequent frame caption matter;
        # retain the previous question when the last user message is its frame.
        image_indexes = [
            i
            for i, m in enumerate(request.messages)
            if isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
        ]
        latest_user = max(i for i, m in enumerate(request.messages) if m.get("role") == "user")
        last_image = image_indexes[-1]
        preceding = [
            text_content(m).strip()
            for m in request.messages[:last_image]
            if m.get("role") == "user" and text_content(m).strip()
        ]
        question = (
            questions[-1] if questions else translate("frigate.prompt.1", language=request.language)
        )
        caption = request.messages[last_image]
        synthetic_frame = text_content(caption).startswith(
            "Here is the current live image from camera "
        )
        if latest_user == last_image and preceding and synthetic_frame:
            question = preceding[-1] + "\n" + question
        # Frigate's synthetic English caption must not determine the answer language.
        human_question = (
            preceding[-1]
            if synthetic_frame and preceding and latest_user == last_image
            else question
        )
        fallback = (
            "en"
            if re.search(r"\b(describe|show|what|image|visible)\b", human_question, re.I)
            else settings.service_language
        )
        language = language_code(request.language or detect_language(human_question, fallback))
        object.__setattr__(request, "_response_language", language)
        if synthetic_frame:
            caption_text = text_content(caption)
            match = re.fullmatch(lexicon("frigate_prompt.pattern.297.16"), caption_text)
            if match:
                question = question.replace(
                    caption_text,
                    translate("frigate.live_caption", language=language, camera=match[1]),
                )
        # Keep the latest image-bearing message (all frames in that message),
        # not old images from earlier questions. Never invent a fresh frame.
        parts = [
            part
            for part in request.messages[last_image]["content"]
            if part.get("type") == "image_url"
        ]
        # Preserve explicit non-chat system constraints (e.g. a description JSON contract).
        custom = [
            text_content(m)
            for m in request.messages
            if m.get("role") == "system" and "helpful assistant for Frigate" not in text_content(m)
        ]
        task = "\n".join([*custom, question])
        if len(task) > settings.frigate_assist_vision_chars:
            raise ValueError(
                "Frigate-Assist vision task exceeds frigate_assist_vision_chars; shorten the description prompt or use a separate description model"
            )
        system = translate("frigate.vision_system", language=language)
        # Historical images are explicitly labelled, never passed off as a fresh live view.
        if last_image < latest_user:
            task = translate("frigate.prompt.4", language=request.language) + task
        omitted_images = sum(
            len([p for p in request.messages[i]["content"] if p.get("type") == "image_url"])
            for i in image_indexes[:-1]
        )
        if omitted_images:
            task = (
                translate("frigate.omitted_images", language=language, count=omitted_images) + task
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": task}, *parts]},
        ]
        choice = request.tool_choice if tools else None
        if tools:
            system = system.replace(translate("frigate.no_tools", language=language), "")
            system += translate("frigate.prompt.2", language=request.language)
            messages[0]["content"] = system
    else:
        system = translate("frigate.prompt.0", language=request.language)
        if settings.frigate_text_model != LLM_MODEL and (
            tools or any(m.get("role") == "tool" or m.get("tool_calls") for m in request.messages)
        ):
            # Native Hailo tool adapters parse JSON for every tool-aware round,
            # including a final summary after completed client calls.
            system += translate("frigate.prompt.3", language=request.language)
        if clock:
            system += "\nServer local time: " + clock
        if cameras:
            system += "\nCameras: " + json.dumps(cameras, ensure_ascii=False, separators=(",", ":"))
        # Retain the compact camera lines, including zones, to preserve zone IDs.
        for m in request.messages:
            if m.get("role") == "system":
                system += "".join(
                    "\n" + line.strip()
                    for line in text_content(m).splitlines()
                    if re.match(r"\s*-.*\(ID:", line)
                )
        start = max(i for i, m in enumerate(request.messages) if m.get("role") == "user")
        messages = [{"role": "system", "content": system}]
        # Preserve the preceding complete exchange, including compact results for references.
        previous_users = [
            i for i, m in enumerate(request.messages[:start]) if m.get("role") == "user"
        ]
        if previous_users:
            prior = copy.deepcopy(text_history[previous_users[-1] : start])
            for message in prior:
                if message.get("role") == "tool":
                    try:
                        data = json.loads(message.get("content", ""))
                    except (ValueError, TypeError):
                        data = message.get("content", "")
                    message["content"] = json.dumps(
                        compact_data(data, settings.frigate_assist_max_events),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
            if len(json.dumps(prior, ensure_ascii=False)) <= min(
                4000, settings.frigate_assist_text_chars // 3
            ):
                messages.extend(prior)
            else:
                messages[0]["content"] += (
                    " Earlier exchange omitted for size; ask for the relevant event or detail if the question depends on it."
                )
        for original in text_history[start:]:
            message = copy.deepcopy(original)
            if message.get("role") == "tool":
                content = message.get("content", "")
                try:
                    data = json.loads(content) if isinstance(content, str) else content
                except (ValueError, TypeError):
                    data = content
                message["content"] = json.dumps(
                    compact_data(data, settings.frigate_assist_max_events),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            messages.append(message)
        choice = request.tool_choice
        if not tools and choice not in ("required",) and not isinstance(choice, dict):
            choice = None
    if context:
        messages[0]["content"] += translate(
            "frigate.resolved_facts", language=request.language
        ) + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    prepared = request.model_copy(
        update={
            "messages": messages,
            "tools": [
                compact_tool(t, getattr(request, "_response_language", None) or request.language)
                for t in tools
            ],
            "tool_choice": choice,
        }
    )
    size = len(json.dumps({"messages": messages, "tools": prepared.tools}, ensure_ascii=False))
    if not images and size > settings.frigate_assist_text_chars:
        raise ValueError(
            "Frigate-Assist required context exceeds frigate_assist_text_chars; narrow the question/time range"
        )
    request._metrics["frigate_prompt"] = {
        "messages_before": len(request.messages),
        "messages_after": len(messages),
        "tools_before": len(request.tools or []),
        "tools_after": len(prepared.tools or []),
        "prepared_chars": size,
        "max_events": settings.frigate_assist_max_events,
    }
    return prepared
