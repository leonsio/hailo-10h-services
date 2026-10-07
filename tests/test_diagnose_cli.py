"""Standalone diagnostic-client protocol and history checks."""

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose_ha.py"
spec = importlib.util.spec_from_file_location("diagnose_ha", SCRIPT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_history_replacement_preserves_catalogue_and_drops_later_tools():
    payload = json.loads((ROOT / "scripts/examples/ha-diagnosis-request.json").read_text())
    payload["messages"] += [{"role": "assistant", "tool_calls": []}, {"role": "tool"}]
    replaced = client.prepare_request(payload, "Neuer Auftrag")
    assert len(replaced["messages"]) == 2
    assert len(payload["messages"]) == 4
    assert replaced["messages"][0] == payload["messages"][0]
    assert replaced["tools"] == payload["tools"]
    assert replaced["messages"][-1]["content"] == "Neuer Auftrag"
    assert client.prepare_request(payload)["messages"] == payload["messages"]


def test_only_diagnosis_urls():
    for base in ("http://localhost:8090", "http://localhost:8090/v1/"):
        assert client.endpoint_url(base) == "http://localhost:8090/v1/ha-assist/diagnose"
    try:
        client.endpoint_url("http://localhost:8090/v1/chat/completions")
    except ValueError:
        pass
    else:
        raise AssertionError("A generation URL must be rejected")


def test_http_replay_authentication_errors_and_saved_evidence(tmp_path):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append((self.path, self.headers.get("Authorization"), body))
            if self.headers.get("Authorization") != "Bearer test-key":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"detail":"Invalid API key"}')
                return
            result = {
                "object": "ha_assist.diagnosis",
                "generative_calls": 0,
                "tools_executed": 0,
                "prepared_request": body,
                "metrics": {"ha_route": {"would_inference_calls": 1}},
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        output = tmp_path / "result.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--url",
            f"http://127.0.0.1:{server.server_port}",
            "--request",
            str(ROOT / "scripts/examples/ha-diagnosis-request.json"),
            "--prompt",
            "Wohnzimer auf 90%",
            "--output",
            str(output),
            "--json",
        ]
        successful = subprocess.run(
            command + ["--api-key", "test-key"], capture_output=True, text=True
        )
        assert successful.returncode == 0, successful.stderr
        assert json.loads(successful.stdout) == json.loads(output.read_text())
        path, auth, request = captured[-1]
        assert path == "/v1/ha-assist/diagnose" and auth == "Bearer test-key"
        assert request["model"] == "HA-Assist" and request["stream"] is False
        assert request["messages"][-1]["content"] == "Wohnzimer auf 90%"
        failed = subprocess.run(command + ["--api-key", "wrong"], capture_output=True, text=True)
        assert failed.returncode == 1 and "HTTP 401" in failed.stderr
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
