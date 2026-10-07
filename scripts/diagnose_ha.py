#!/usr/bin/env python3
"""Replay a Home Assistant request through the non-generating diagnosis endpoint."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirects(HTTPRedirectHandler):
    """Keep authenticated request data on the explicitly selected endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Reject redirection rather than forwarding credentials or POST data.

        Args:
            req: Original request.
            fp: Response stream.
            code: HTTP status.
            msg: HTTP status message.
            headers: Response headers.
            newurl: Redirect destination.

        Returns:
            None: Let urllib report the redirect as an HTTP error.
        """
        return None


def prepare_request(payload, prompt=None):
    """Copy a complete HA envelope and optionally replace its last user turn.

    Args:
        payload: Parsed OpenAI request, including catalogue and tool schemas.
        prompt: Optional replacement user text; drops subsequent tool history.

    Returns:
        dict: Independent request targeting HA-Assist with streaming disabled.

    Raises:
        ValueError: Messages or tool schemas are missing or invalid.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object")
    result = copy.deepcopy(payload)
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Request needs a non-empty messages array")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("Each message must be an object")
    if not isinstance(result.get("tools"), list) or not result["tools"]:
        raise ValueError("Use a complete HA request including tools and static catalogue")
    if prompt is not None:
        indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]
        if not indices:
            raise ValueError("Cannot replace prompt: request has no user message")
        last = indices[-1]
        result["messages"] = messages[: last + 1]
        result["messages"][last]["content"] = prompt
    result.update(model="HA-Assist", stream=False)
    return result


def endpoint_url(url):
    """Accept a service root, /v1 base or the full diagnosis URL.

    Args:
        url: HTTP(S) service address, optionally including a reverse-proxy prefix.

    Returns:
        str: Full non-generating diagnosis endpoint URL.

    Raises:
        ValueError: URL is invalid or selects a different API endpoint.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("URL must start with http:// or https://")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use a URL without credentials, query or fragment")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1/ha-assist/diagnose"):
        if "/v1/" in path:
            raise ValueError("Use the service root, /v1 base or diagnosis endpoint")
        path += "/ha-assist/diagnose" if path.endswith("/v1") else "/v1/ha-assist/diagnose"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def main():
    """Run one diagnosis and print its evidence or the complete JSON response.

    Returns:
        int: Zero on success; one for request, network or response errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Service URL, e.g. http://HOST:8090")
    parser.add_argument("--api-key", default=os.environ.get("HAILO_API_KEY", ""))
    parser.add_argument("--request", required=True, type=Path, help="Complete HA request JSON")
    parser.add_argument("--prompt", help="Replace last user message; drop subsequent history")
    parser.add_argument("--timeout", type=float, default=60, help="HTTP timeout in seconds")
    parser.add_argument("--output", type=Path, help="Save complete response JSON")
    parser.add_argument("--json", action="store_true", help="Print complete response JSON")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        payload = prepare_request(json.loads(args.request.read_text(encoding="utf-8")), args.prompt)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if args.api_key:
            headers["Authorization"] = "Bearer " + args.api_key
        request = Request(
            endpoint_url(args.url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with build_opener(NoRedirects()).open(request, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000
        if not isinstance(result, dict) or result.get("object") != "ha_assist.diagnosis":
            raise ValueError("Server did not return an HA diagnosis")
        if result.get("generative_calls") != 0 or result.get("tools_executed") != 0:
            raise ValueError("Response does not confirm zero generation and zero tool execution")
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        if args.json:
            print(rendered)
        else:
            print(f"HTTP round trip: {elapsed:.1f} ms; preparation: {result.get('duration_ms')} ms")
            metrics = result.get("metrics", {})
            for label, value in (
                ("Route", metrics.get("ha_route")),
                ("Proposed response / tool calls", result.get("proposed_response")),
                ("Plan / candidates", metrics.get("ha_plan")),
                ("Intent", metrics.get("ha_intent")),
                ("Stages (ms)", metrics.get("ha_stages_ms")),
                ("Prompt stage", result.get("prompt_stage")),
                ("Prepared model request", result.get("prepared_request")),
            ):
                print(f"\n{label}:\n{json.dumps(value, ensure_ascii=False, indent=2)}")
        return 0
    except HTTPError as exc:
        print(
            f"HTTP {exc.code}: {exc.read(8192).decode('utf-8', errors='replace')}", file=sys.stderr
        )
    except (OSError, ValueError, URLError) as exc:
        print(f"Diagnosis failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
