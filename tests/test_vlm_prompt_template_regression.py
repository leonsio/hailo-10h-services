from hailo_services.vlm_chat import render_prompt


class QwenTemplateModel:
    def prompt_template(self):
        # Reduced form of the Qwen3-VL template fields that caused the
        # production failure: assistant.tool_calls and add_vision_id.
        return """
{%- set image_count = namespace(value=0) %}
{%- for message in messages %}
{%- if message.role == "user" %}
<|im_start|>user\n
{%- for content in message.content %}
{%- if content.type == "image" or "image" in content or "image_url" in content %}
{%- set image_count.value = image_count.value + 1 %}
{%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
<|vision_start|><|image_pad|><|vision_end|>
{%- elif "text" in content %}{{ content.text }}{% endif -%}
{%- endfor %}<|im_end|>\n
{%- elif message.role == "assistant" %}
<|im_start|>assistant\n
{%- for content in message.content %}
{%- if "text" in content %}{{ content.text }}{% endif -%}
{%- endfor %}
{%- if message.tool_calls %}unexpected-tool-call{% endif -%}
<|im_end|>\n
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif -%}
"""


def test_qwen3_budget_template_accepts_assistant_chat_history_without_tool_calls():
    prompt = [
        {"role": "user", "content": [{"type": "text", "text": "wieviele sind 2+2"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "2 + 2 sind 4."}]},
        {"role": "user", "content": [{"type": "text", "text": "was ist die Hauptstadt von Frankreich?"}]},
    ]

    rendered = render_prompt(QwenTemplateModel(), prompt)

    assert "2 + 2 sind 4." in rendered
    assert "Hauptstadt von Frankreich" in rendered
    assert "unexpected-tool-call" not in rendered
    # Budget rendering must not modify the model-bound prompt.
    assert "tool_calls" not in prompt[1]


def test_qwen_budget_template_defines_optional_add_vision_id_for_images():
    prompt = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Was ist auf dem Bild?"},
        ],
    }]

    rendered = render_prompt(QwenTemplateModel(), prompt)

    assert "<|vision_start|><|image_pad|><|vision_end|>" in rendered
    assert "Picture 1:" not in rendered
    assert "Was ist auf dem Bild?" in rendered
