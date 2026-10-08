"""Regression guards for explicit backend ownership and documented source contracts."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hailo_services.chat import chat_common, chat_hailo_llm, chat_hailo_vlm
from hailo_services.chat.backend_hailo import HailoBackend
from hailo_services.chat.backend_litert import LiteRTLMBackend
from hailo_services.schemas import ChatRequest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "hailo_services"


def test_package_import_does_not_load_or_patch_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hailo_services; "
            "assert 'hailo_services.runtime.runtime' not in sys.modules; "
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
    assert HailoBackend.select_tools.__module__ == "hailo_services.chat.backend_hailo"
    assert LiteRTLMBackend.chat.__module__ == "hailo_services.chat.backend_litert"


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


def test_removed_compatibility_modules_are_not_importable():
    for name in ("vlm_chat", "hailo_llm_chat", "litert_optimizations"):
        assert not (SOURCE / f"{name}.py").exists()
        assert importlib.util.find_spec(f"hailo_services.{name}") is None


def test_production_docstrings_cover_definitions_and_return_contracts():
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        # IPC proxy methods retain the documented backend contracts; Ruff
        # enforces their explicit per-file docstring exemptions.
        if path.relative_to(SOURCE).parts[:2] == ("runtime", "workers"):
            continue
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


def test_assistants_do_not_import_each_others_domain_modules():
    for domain, other in (("ha", "frigate"), ("frigate", "ha")):
        for path in (SOURCE / "assistants" / domain).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            assert not any(
                name and name.startswith(f"hailo_services.assistants.{other}.") for name in imports
            ), path
