"""Regression guards for explicit backend ownership and documented source contracts."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hailo_services import chat_common, chat_hailo_llm, chat_hailo_vlm
from hailo_services.backend_hailo import HailoBackend
from hailo_services.backend_litert import LiteRTLMBackend
from hailo_services.schemas import ChatRequest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "hailo_services"


def test_package_import_does_not_load_or_patch_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hailo_services; "
            "assert 'hailo_services.runtime' not in sys.modules; "
            "assert 'hailo_platform' not in sys.modules; "
            "assert 'litert_lm' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    before = (HailoBackend.select_tools, LiteRTLMBackend.chat)
    importlib.reload(importlib.import_module("hailo_services"))
    assert before == (HailoBackend.select_tools, LiteRTLMBackend.chat)
    assert HailoBackend.select_tools.__module__ == "hailo_services.backend_hailo"
    assert LiteRTLMBackend.chat.__module__ == "hailo_services.backend_litert"


def test_llm_generation_uses_the_exact_prompt_counted_once():
    rendered = []
    tokenized = []

    def template():
        rendered.append(True)
        return "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}assistant:"

    model = SimpleNamespace(
        prompt_template=template,
        tokenize=lambda text: tokenized.append(text) or text.split(),
    )
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=32)
    trimmed, prompt = chat_hailo_llm.limit_request(model, request, 1024, 2048)
    assert trimmed.messages == request.messages
    assert prompt == tokenized[-1]
    assert len(rendered) == 1
    assert len(tokenized) == 1
    assert chat_hailo_vlm.model_prompt is chat_common.model_prompt


def test_compatibility_imports_use_canonical_implementations():
    from hailo_services import hailo_llm_chat, litert_optimizations, vlm_chat
    from hailo_services.diagnostics_litert import instrument_engine
    from hailo_services.ha_action_verification import successful_action_followup

    assert hailo_llm_chat.limit_request is chat_hailo_llm.limit_request
    assert vlm_chat.model_prompt is chat_hailo_vlm.model_prompt
    assert litert_optimizations.instrument_engine is instrument_engine
    assert litert_optimizations.successful_action_followup is successful_action_followup


def test_production_docstrings_cover_definitions_and_return_contracts():
    missing = []
    for path in sorted(SOURCE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(f"{path.name}: module")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                missing.append(f"{path.name}:{node.lineno}: {node.name}")
            elif not isinstance(node, ast.ClassDef) and not ("Returns:" in doc or "Yields:" in doc):
                missing.append(f"{path.name}:{node.lineno}: {node.name}: return contract")
    assert not missing, "\n".join(missing)
