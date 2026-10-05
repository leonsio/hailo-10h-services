"""Exercise the standalone benchmark over real HTTP without model hardware."""

import importlib.util
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-text.py"
spec = importlib.util.spec_from_file_location("benchmark_text", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@contextmanager
def server(*, fail_call=None, fail_status=500, llm_available=True):
    calls, headers = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            headers.append(self.headers.get("Authorization"))
            if self.path == "/ui/config":
                self.reply(200, {"llm_model": "gemma-test", "vlm_model": "qwen-test"})
            else:
                models = ["qwen-test"] + (["gemma-test"] if llm_available else [])
                self.reply(200, {"data": [{"id": value} for value in models]})

        def do_POST(self):
            headers.append(self.headers.get("Authorization"))
            assert self.path == "/v1/chat/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            if len(calls) == fail_call:
                self.reply(fail_status, {"error": "test failure"})
                return
            metrics = {"input_tokens": 123, "input_tokens_source": "native",
                       "output_tokens": 7, "output_tokens_source": "native",
                       "processing_ms": 75, "inference_ms": 50, "request_id": f"test-{len(calls)}"}
            if payload["model"] == "gemma-test":
                metrics.update(ttft_ms=10, ttft_source="native", decode_tokens_per_second=14)
            self.reply(200, {"model": payload["model"],
                             "choices": [{"message": {"content": "<script>alert(1)</script>\nAntwort"},
                                          "finish_reason": "stop"}], "metrics": metrics,
                             "usage": {"prompt_tokens": 123, "completion_tokens": 7}})

    value = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=value.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{value.server_port}", calls, headers
    finally:
        value.shutdown()
        value.server_close()
        thread.join()


def test_complete_comparison_uses_parameter_key_and_preserves_real_metrics(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HAILO_API_KEY", "wrong-environment-key")
    with server() as (url, calls, headers):
        assert benchmark.main(["--url", url + "/v1/", "--api-key", "parameter-secret",
                               "--output-dir", str(tmp_path)]) == 0
    assert len(calls) == 20
    assert set(headers) == {"Bearer parameter-secret"}
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["status"] == "abgeschlossen"
    assert len(report["cases"]) == 10
    for index, case in enumerate(report["cases"]):
        pair = calls[index * 2:index * 2 + 2]
        assert pair[0]["messages"] == pair[1]["messages"]
        assert len(pair[0]["messages"]) == 1
        assert pair[0]["messages"][0]["content"] == case["prompt"]
        assert {value["model"] for value in pair} == {"qwen-test", "gemma-test"}
        assert pair[0]["model"] == ("gemma-test" if index % 2 == 0 else "qwen-test")
        for result in case["results"].values():
            assert result["client_total_ms"] >= 0
            assert result["metrics"]["input_tokens"] == 123
            assert result["request"]["temperature"] == 0
            assert result["request"]["max_tokens"] == 256
            assert result["request"]["stream"] is False
            assert "image" not in str(result["request"]["messages"])
        assert case["results"]["llm"]["metrics"]["ttft_ms"] == 10
        assert "ttft_ms" not in case["results"]["vlm"]["metrics"]
    output = capsys.readouterr().out
    assert "LLM · gemma-test" in output and "VLM · qwen-test" in output
    assert "nicht verfügbar" in output
    rendered = (tmp_path / "comparison.html").read_text()
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered
    assert "parameter-secret" not in rendered + output + (tmp_path / "results.json").read_text()
    assert report["summary"]["llm"]["successful"] == 10


def test_http_error_does_not_discard_completed_results(tmp_path, capsys):
    with server(fail_call=3) as (url, calls, _):
        assert benchmark.main(["--url", url, "--api-key", "test", "--output-dir", str(tmp_path)]) == 1
    assert len(calls) == 20
    report = json.loads((tmp_path / "results.json").read_text())
    failed = report["cases"][1]["results"]["vlm"]
    assert not failed["ok"]
    assert "HTTP 500" in failed["error"]
    assert failed["client_total_ms"] >= 0
    assert failed["metrics"] == {}
    assert report["summary"]["vlm"]["failed"] == 1
    assert report["summary"]["llm"]["successful"] == 10


def test_unavailable_llm_stops_before_inference(tmp_path, capsys):
    with server(llm_available=False) as (url, calls, _):
        assert benchmark.main(["--url", url, "--api-key", "test", "--output-dir", str(tmp_path)]) == 2
    assert calls == []
    assert "nicht verfügbar" in capsys.readouterr().err


def test_native_timeout_stops_subsequent_measurements_and_keeps_partial_report(tmp_path, capsys):
    with server(fail_call=3, fail_status=504) as (url, calls, _):
        assert benchmark.main(["--url", url, "--api-key", "test", "--output-dir", str(tmp_path)]) == 1
    assert len(calls) == 3
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["status"] == "abgebrochen nach Timeout"
    assert report["cases"][1]["results"]["vlm"]["timed_out"]
    rendered = (tmp_path / "comparison.html").read_text()
    assert "HTTP 504" in rendered and "nicht ausgeführt" in rendered


def test_warmup_is_recorded_but_excluded_from_comparison(tmp_path, capsys):
    with server() as (url, calls, _):
        assert benchmark.main(["--url", url, "--api-key", "test", "--warmup",
                               "--output-dir", str(tmp_path)]) == 0
    assert len(calls) == 22
    report = json.loads((tmp_path / "results.json").read_text())
    assert len(report["warmup"]) == 2
    assert len(report["cases"]) == 10
    assert report["summary"]["vlm"]["completed"] == 10


@pytest.mark.parametrize("argument,value", [("--max-tokens", "0"), ("--temperature", "nan"),
                                           ("--timeout", "-1"), ("--seed", "-1"),
                                           ("--url", "https://user:secret@example.com")])
def test_invalid_parameters_rejected_before_requests(argument, value):
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args([argument, value])
    assert exc.value.code == 2
