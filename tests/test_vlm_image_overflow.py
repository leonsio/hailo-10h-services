import pytest

from hailo_services.chat_hailo_vlm import limit_request
from hailo_services.schemas import ChatRequest


class BudgetVLM:
    def tokenize(self, _text):
        # Mirrors the Frigate failure shape: text itself is small, while the
        # conservative image reserve dominates the input budget.
        return range(100)

    def max_context_capacity(self):
        return 2048


def _request(image_count):
    content = [{"type": "text", "text": "Analyze the sequence of images."}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"https://example.invalid/frame-{index}.jpg"},
        }
        for index in range(image_count)
    )
    return ChatRequest(
        model="Qwen2-VL-2B-Instruct",
        messages=[{"role": "user", "content": content}],
        max_tokens=256,
    )


def _frame_numbers(request):
    values = []
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                values.append(int(part["image_url"]["url"].rsplit("-", 1)[1].split(".")[0]))
    return values


@pytest.mark.parametrize(
    "image_count, expected",
    [
        (5, [0, 1, 3, 4]),
        (10, [0, 3, 6, 9]),
    ],
)
def test_qwen2_frigate_sequences_are_sampled_before_budgeting(image_count, expected, caplog):
    with caplog.at_level("INFO"):
        trimmed, prompt = limit_request(BudgetVLM(), _request(image_count), 2048, 2048)

    assert _frame_numbers(trimmed) == expected
    assert sum(
        part.get("type") == "image"
        for message in prompt
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ) == 4
    assert "vlm_image_overflow" in caplog.text
    assert f"images={image_count}" in caplog.text
    assert "max_images=4" in caplog.text


def test_qwen2_four_images_are_left_unchanged(caplog):
    request = _request(4)
    with caplog.at_level("INFO"):
        trimmed, _prompt = limit_request(BudgetVLM(), request, 2048, 2048)

    assert _frame_numbers(trimmed) == [0, 1, 2, 3]
    assert "vlm_image_overflow" not in caplog.text


def test_single_image_vlm_is_not_silently_collapsed():
    request = _request(2).model_copy(update={"model": "Qwen3-VL-2B-Instruct"})
    trimmed, prompt = limit_request(BudgetVLM(), request, 2048, 2048)

    # Qwen3's catalogue limit is one image. The adapter deliberately leaves
    # this semantic mismatch intact so backend_hailo keeps returning its clear
    # "supports at most 1 image" error instead of silently choosing a frame.
    assert _frame_numbers(trimmed) == [0, 1]
    assert sum(
        part.get("type") == "image"
        for message in prompt
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ) == 2
