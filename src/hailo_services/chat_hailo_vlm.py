"""Hailo VLM adapter retaining visual placeholders in shared chat prompts."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .chat_common import limit_request as _limit_request
from .chat_common import model_prompt, render_prompt, tool_response

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _sequence_image_limits() -> dict[str, int]:
    """Return bundled multi-image VLM limits that can be sampled safely.

    Single-image models are intentionally excluded: collapsing a genuinely
    multi-image request to one frame changes its semantics too much, so the
    backend keeps rejecting those requests with its normal model-limit error.
    """
    path = Path(__file__).with_name("model_catalog.yaml")
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    models = document.get("models", {}) if isinstance(document, dict) else {}
    limits: dict[str, int] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict) or entry.get("kind") != "vlm":
            continue
        limit = entry.get("max_images")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 1:
            limits[str(name)] = limit
    return limits


def _image_positions(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return message/part coordinates for OpenAI image_url parts."""
    positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                positions.append((message_index, part_index))
    return positions


def _uniform_sample(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    """Select evenly spaced items while preserving the first and last frame."""
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[-1]]
    last = len(items) - 1
    denominator = count - 1
    return [items[(index * last + denominator // 2) // denominator] for index in range(count)]


def _limit_sequence_images(request, max_images: int):
    """Fit an overflowing image sequence to a multi-frame VLM.

    Images from the active user turn take precedence over historical images.
    When the active turn itself overflows, frames are sampled uniformly so a
    Frigate sequence retains its beginning, intermediate motion and final state.
    """
    positions = _image_positions(request.messages)
    if len(positions) <= max_images:
        return request

    last_user = max(
        (index for index, message in enumerate(request.messages) if message.get("role") == "user"),
        default=-1,
    )
    active = [position for position in positions if position[0] >= last_user]
    if len(active) >= max_images:
        selected = _uniform_sample(active, max_images)
    else:
        historical = [position for position in positions if position not in active]
        selected = historical[-(max_images - len(active)) :] + active

    selected_set = set(selected)
    messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(request.messages):
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        filtered = [
            part
            for part_index, part in enumerate(content)
            if not (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and (message_index, part_index) not in selected_set
            )
        ]
        messages.append({**message, "content": filtered})

    ordinals = [index for index, position in enumerate(positions) if position in selected_set]
    _LOG.info(
        "vlm_image_overflow request_id=%s model=%s images=%d max_images=%d kept=%d "
        "dropped=%d selected_indices=%s",
        getattr(request, "_request_id", "-"),
        request.model,
        len(positions),
        max_images,
        len(selected),
        len(positions) - len(selected),
        ordinals,
    )
    return request.model_copy(update={"messages": messages})


def limit_request(
    model,
    request,
    configured_limit,
    context_length,
    *,
    debug=False,
    prompt_builder=model_prompt,
    model_kind="VLM",
    template_options=None,
):
    """Apply multi-frame overflow handling before shared VLM token budgeting."""
    max_images = _sequence_image_limits().get(request.model)
    if max_images is not None:
        request = _limit_sequence_images(request, max_images)
    return _limit_request(
        model,
        request,
        configured_limit,
        context_length,
        debug=debug,
        prompt_builder=prompt_builder,
        model_kind=model_kind,
        template_options=template_options,
    )


__all__ = ["limit_request", "model_prompt", "render_prompt", "tool_response"]
