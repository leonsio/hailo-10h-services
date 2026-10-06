#!/usr/bin/env python3
"""Compare resident Qwen VLMs on plain text and compact HA intent extraction.

This benchmark deliberately sends no OpenAI tools and no Home Assistant
``Static Context:`` envelope. It therefore measures the selected resident VLM
instead of deterministic HA routing or generic function calling.

Run once with Qwen2-VL resident and once with Qwen3-VL resident. Each run writes
model-specific JSON/CSV files. ``--compare`` compares two saved JSON reports.
Python 3.10+, stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

KNOWN_MODELS = ("Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct")
DEFAULT_OUTPUT_DIR = "qwen-text-intent-benchmark"


class Scenario:
    def __init__(self, id, group, prompt, expected, *, title=None):
        self.id = id
        self.group = group
        self.prompt = prompt
        self.expected = expected
        self.title = title or id

    def as_dict(self):
        return vars(self)


GENERAL_SYSTEM = (
    "Isolierter Sprachmodell-Benchmark. Folge der Benutzeranweisung exakt. "
    "Antworte nur mit der verlangten kurzen Antwort, ohne Erklärung, Markdown oder Präambel."
)

INTENTS = (
    "LIGHT_BRIGHTNESS",
    "LIGHT_ON",
    "LIGHT_OFF",
    "LIGHT_COLOR",
    "CLIMATE_TEMPERATURE",
    "COVER_POSITION",
    "COVER_CLOSE",
    "VACUUM_START",
    "SWITCH_ON",
    "STATE_READ_TEMPERATURE",
)
INTENT_SYSTEM = (
    "Isolierter Home-Assistant-Intent-Benchmark. Es werden keine echten Geräte gesteuert.\n"
    "Klassifiziere nur den letzten Benutzertext und extrahiere die darin ausdrücklich genannten Slots.\n"
    "Antworte mit GENAU EINER Zeile im Format:\n"
    "INTENT|TARGET_TYPE|TARGET|VALUE\n"
    "Erlaubte INTENTs: " + ", ".join(INTENTS) + ".\n"
    "TARGET_TYPE ist nur area oder name. TARGET muss exakt aus dem Benutzertext übernommen werden; "
    "nichts erfinden, ergänzen oder umformulieren. VALUE ist nur ein ausdrücklich genannter Wert; "
    "wenn keiner benötigt wird, verwende -. Keine Erklärung und kein Markdown."
)

SCENARIOS = [
    Scenario("text_math_2_plus_2", "text", "Was ist 2 + 2? Antworte ausschließlich mit der Zahl.", "4", title="2 + 2"),
    Scenario("text_capital_france", "text", "Was ist die Hauptstadt von Frankreich? Antworte nur mit dem Stadtnamen.", "Paris", title="Hauptstadt Frankreich"),
    Scenario("text_math_17_times_6", "text", "Was ist 17 mal 6? Antworte ausschließlich mit der Zahl.", "102", title="17 × 6"),
    Scenario("text_next_weekday", "text", "Welcher Wochentag kommt nach Montag? Antworte nur mit dem Wochentag.", "Dienstag", title="Wochentag"),
    Scenario("text_larger_number", "text", "Welche Zahl ist größer: 17 oder 23? Antworte nur mit der Zahl.", "23", title="Zahlenvergleich"),
    Scenario("text_translate_house", "text", "Übersetze das englische Wort house ins Deutsche. Antworte nur mit einem Wort.", "Haus", title="Einfache Übersetzung"),
    Scenario("intent_light_brightness", "intent", "Schalte das Licht im Wohnzimmer auf 70 Prozent.", "LIGHT_BRIGHTNESS|area|Wohnzimmer|70", title="Licht Helligkeit"),
    Scenario("intent_light_on", "intent", "Schalte das Licht in der Küche an.", "LIGHT_ON|area|Küche|-", title="Licht einschalten"),
    Scenario("intent_light_off_name", "intent", "Schalte die Nachttischlampe aus.", "LIGHT_OFF|name|Nachttischlampe|-", title="Licht nach Name aus"),
    Scenario("intent_light_color", "intent", "Stelle das Licht im Wohnzimmer auf rot.", "LIGHT_COLOR|area|Wohnzimmer|rot", title="Lichtfarbe"),
    Scenario("intent_climate_temperature", "intent", "Stelle die Temperatur im Schlafzimmer auf 19,5 Grad.", "CLIMATE_TEMPERATURE|area|Schlafzimmer|19,5", title="Solltemperatur"),
    Scenario("intent_cover_position", "intent", "Fahre den Rollladen im Dachgeschoss auf 40 Prozent.", "COVER_POSITION|area|Dachgeschoss|40", title="Rollladen Position"),
    Scenario("intent_cover_close", "intent", "Schließe den Rollladen.", "COVER_CLOSE|name|Rollladen|-", title="Rollladen schließen"),
    Scenario("intent_vacuum_start", "intent", "Starte Deebot mini.", "VACUUM_START|name|Deebot mini|-", title="Staubsauger starten"),
    Scenario("intent_switch_on", "intent", "Schalte die Kaffeemaschine an.", "SWITCH_ON|name|Kaffeemaschine|-", title="Schalter einschalten"),
    Scenario("intent_state_temperature", "intent", "Wie warm ist es im Wohnzimmer?", "STATE_READ_TEMPERATURE|area|Wohnzimmer|-", title="Temperatur abfragen"),
]


class HttpFailure(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        self.payload = None
        try:
            self.payload = json.loads(body)
        except (TypeError, ValueError):
            pass
        super().__init__(f"HTTP {status}: {body}")

    @property
    def message(self):
        value = self.payload.get("error") if isinstance(self.payload, dict) else None
        if isinstance(value, dict):
            return str(value.get("message") or value)
        if value is not None:
            return str(value)
        return self.body.strip() or f"HTTP {self.status}"


class Client:
    def __init__(self, url, key, timeout, context):
        self.url = url
        self.key = key
        self.timeout = timeout
        self.context = context

    def json(self, path, payload=None):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise HttpFailure(exc.code, exc.read(16384).decode(errors="replace")) from exc


def normalize_url(value):
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--url muss eine vollständige http://- oder https://-Adresse sein")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("--url darf keine Zugangsdaten, Query-Parameter oder Fragmente enthalten")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def metric(response, name, usage=None):
    metrics = response.get("metrics") or {}
    if name in metrics:
        return metrics[name]
    if usage:
        return (response.get("usage") or {}).get(usage)
    return None


def message_text(response):
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    value = message.get("content") if isinstance(message, dict) else ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(
            str(part.get("text", "")) for part in value if isinstance(part, dict)
        ).strip()
    return ""


def _strip_single_fence(text):
    value = text.strip()
    match = re.fullmatch(r"```(?:text|txt)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _norm_word(value):
    return " ".join(value.strip().casefold().split())


def _norm_numeric(value):
    value = value.strip().replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return str(int(number)) if number.is_integer() else (f"{number:.12g}")


def validate_text(answer, expected):
    value = _strip_single_fence(answer)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"correct": False, "reason": "expected exactly one answer line", "answer": answer}
    correct = _norm_word(lines[0]).rstrip(".!") == _norm_word(expected).rstrip(".!")
    return {"correct": correct, "reason": "ok" if correct else "wrong answer", "answer": answer}


def parse_intent_line(answer):
    value = _strip_single_fence(answer)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "expected exactly one intent line"
    parts = [part.strip() for part in lines[0].split("|")]
    if len(parts) != 4:
        return None, "intent output must have four pipe-separated fields"
    return parts, None


def validate_intent(answer, expected):
    actual, error = parse_intent_line(answer)
    if error:
        return {"correct": False, "reason": error, "answer": answer}
    wanted = [part.strip() for part in expected.split("|")]
    intent_ok = actual[0] == wanted[0]
    target_type_ok = actual[1] == wanted[1]
    target_ok = _norm_word(actual[2]) == _norm_word(wanted[2])
    if wanted[3] == "-":
        value_ok = actual[3] == "-"
    elif _norm_numeric(wanted[3]) is not None:
        value_ok = _norm_numeric(actual[3]) == _norm_numeric(wanted[3])
    else:
        value_ok = _norm_word(actual[3]) == _norm_word(wanted[3])
    checks = {
        "intent": intent_ok,
        "target_type": target_type_ok,
        "target": target_ok,
        "value": value_ok,
    }
    correct = all(checks.values())
    wrong = [name for name, ok in checks.items() if not ok]
    return {
        "correct": correct,
        "reason": "ok" if correct else "wrong " + ",".join(wrong),
        "answer": answer,
        "parsed": actual,
        "expected": wanted,
        "checks": checks,
    }


def validate(scenario, response):
    answer = message_text(response)
    if scenario.group == "intent":
        return validate_intent(answer, scenario.expected)
    return validate_text(answer, scenario.expected)


def payload_for(scenario, model, max_tokens, temperature, seed):
    system = INTENT_SYSTEM if scenario.group == "intent" else GENERAL_SYSTEM
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": scenario.prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "language": "de",
        "stream": False,
    }


def available_model_ids(client):
    data = client.json("/v1/models")
    return {
        item.get("id") for item in data.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def resolve_model(client, requested):
    available = available_model_ids(client)
    if requested != "auto":
        if requested not in available:
            raise RuntimeError(f"{requested!r} nicht in /v1/models; verfügbar: {sorted(available)}")
        return requested
    resident = [model for model in KNOWN_MODELS if model in available]
    if len(resident) == 1:
        return resident[0]
    if not resident:
        raise RuntimeError(f"Kein Qwen2/Qwen3-VL in /v1/models; verfügbar: {sorted(available)}")
    raise RuntimeError("Mehrere Qwen-VLMs verfügbar; --model explizit setzen: " + ", ".join(resident))


def run_one(client, scenario, model, args, repeat):
    payload = payload_for(scenario, model, args.max_tokens, args.temperature, args.seed + repeat - 1)
    result = {
        "scenario": scenario.id,
        "title": scenario.title,
        "group": scenario.group,
        "expected": scenario.expected,
        "repeat": repeat,
        "request": payload,
        "requested_at": now(),
    }
    started = time.perf_counter()
    try:
        response = client.json("/v1/chat/completions", payload)
        validation = validate(scenario, response)
        result.update(
            response=response,
            returned_model=response.get("model"),
            answer=message_text(response),
            validation=validation,
            input_tokens=metric(response, "input_tokens", "prompt_tokens"),
            input_budget_tokens=metric(response, "input_budget_tokens"),
            output_tokens=metric(response, "output_tokens", "completion_tokens"),
            ttft_ms=metric(response, "ttft_ms"),
            inference_ms=metric(response, "inference_ms"),
            processing_ms=metric(response, "processing_ms"),
            http_status=200,
        )
        result["ok"] = response.get("model") == model and validation["correct"]
        if response.get("model") != model:
            result["validation"] = {
                "correct": False,
                "reason": f"server returned model {response.get('model')!r}",
                "answer": result["answer"],
            }
    except HttpFailure as exc:
        result.update(
            ok=False,
            http_status=exc.status,
            server_error=exc.message,
            error=str(exc),
            answer="",
            validation={"correct": False, "reason": f"HTTP {exc.status}: {exc.message}"},
            input_tokens=None,
            input_budget_tokens=None,
            output_tokens=None,
            ttft_ms=None,
            inference_ms=None,
            processing_ms=None,
        )
    except Exception as exc:
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            answer="",
            validation={"correct": False, "reason": f"client error: {type(exc).__name__}"},
            input_tokens=None,
            input_budget_tokens=None,
            output_tokens=None,
            ttft_ms=None,
            inference_ms=None,
            processing_ms=None,
        )
    result.update(client_total_ms=(time.perf_counter() - started) * 1000, responded_at=now())
    return result


def _numbers(rows, key):
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def _avg(rows, key):
    values = _numbers(rows, key)
    return statistics.fmean(values) if values else None


def _median(rows, key):
    values = _numbers(rows, key)
    return statistics.median(values) if values else None


def summarize(results):
    summary = {}
    for group in ("text", "intent", "all"):
        rows = results if group == "all" else [row for row in results if row["group"] == group]
        passed = sum(bool(row.get("ok")) for row in rows)
        total = len(rows)
        summary[group] = {
            "runs": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": (passed / total) if total else None,
            "avg_client_total_ms": _avg(rows, "client_total_ms"),
            "median_client_total_ms": _median(rows, "client_total_ms"),
            "avg_inference_ms": _avg(rows, "inference_ms"),
            "avg_ttft_ms": _avg(rows, "ttft_ms"),
            "avg_input_tokens": _avg(rows, "input_tokens"),
            "avg_input_budget_tokens": _avg(rows, "input_budget_tokens"),
            "avg_output_tokens": _avg(rows, "output_tokens"),
            "failure_reasons": dict(Counter(
                row.get("validation", {}).get("reason", "unknown")
                for row in rows if not row.get("ok")
            )),
        }
    return summary


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def detail(result):
    answer = " ".join(str(result.get("answer") or result.get("server_error") or "").split())
    return answer if len(answer) <= 160 else answer[:157] + "..."


def write_csv(path, results):
    fields = [
        "scenario", "title", "group", "repeat", "expected", "answer", "http_status", "ok",
        "client_total_ms", "processing_ms", "inference_ms", "ttft_ms", "input_tokens",
        "input_budget_tokens", "output_tokens", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            row["reason"] = result.get("validation", {}).get("reason")
            writer.writerow(row)


def format_ms(value):
    return "-" if not isinstance(value, (int, float)) else f"{value:.1f}"


def print_summary(model, summary):
    print(f"\nZusammenfassung {model}:")
    print("Bereich      korrekt      Accuracy   Ø gesamt ms   Ø inference ms   Ø TTFT ms")
    for group in ("text", "intent", "all"):
        data = summary[group]
        accuracy = "-" if data["accuracy"] is None else f"{data['accuracy'] * 100:6.1f}%"
        print(
            f"{group:10} {data['passed']:3}/{data['runs']:<3}     {accuracy:>8}   "
            f"{format_ms(data['avg_client_total_ms']):>11}   "
            f"{format_ms(data['avg_inference_ms']):>14}   "
            f"{format_ms(data['avg_ttft_ms']):>9}"
        )


def load_report(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model" not in data or "summary" not in data:
        raise ValueError(f"Keine gültige Benchmark-Datei: {path}")
    return data


def compare_reports(first, second):
    reports = [first, second]
    print("Modellvergleich:")
    print("Modell                    Bereich   Accuracy   Ø gesamt ms   Ø inference ms   Ø TTFT ms   Ø out tok")
    for report in reports:
        for group in ("text", "intent", "all"):
            data = report["summary"][group]
            accuracy = "-" if data["accuracy"] is None else f"{data['accuracy'] * 100:6.1f}%"
            out_tokens = data.get("avg_output_tokens")
            out_text = "-" if not isinstance(out_tokens, (int, float)) else f"{out_tokens:.1f}"
            print(
                f"{report['model'][:24]:24} {group:8} {accuracy:>8}   "
                f"{format_ms(data.get('avg_client_total_ms')):>11}   "
                f"{format_ms(data.get('avg_inference_ms')):>14}   "
                f"{format_ms(data.get('avg_ttft_ms')):>9}   {out_text:>9}"
            )
    print("\nFehler pro Modell:")
    for report in reports:
        failures = [row for row in report.get("results", []) if not row.get("ok")]
        print(f"- {report['model']}: {len(failures)} Fehler")
        for row in failures:
            print(
                f"  {row.get('scenario')}: {row.get('validation', {}).get('reason')} | "
                f"{detail(row)}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen2/Qwen3 text and Home-Assistant intent benchmark without tool calling"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key", default=None, help="überschreibt HAILO_API_KEY")
    parser.add_argument("--model", default="auto", help="auto, Qwen2-VL-2B-Instruct oder Qwen3-VL-2B-Instruct")
    parser.add_argument("--groups", choices=("all", "text", "intent"), default="all")
    parser.add_argument("--tasks", default="all", help="all oder kommaseparierte Scenario-IDs")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("QWEN2_JSON", "QWEN3_JSON"))
    args = parser.parse_args(argv)

    if args.compare:
        return args
    try:
        args.url = normalize_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats muss zwischen 1 und 20 liegen")
    if not 1 <= args.max_tokens <= 256:
        parser.error("--max-tokens muss zwischen 1 und 256 liegen")
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        parser.error("--temperature muss endlich und >0 sein (Hailo VLM-Anforderung)")
    if args.seed < 0 or args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("ungültige Laufzeitparameter")

    selected = SCENARIOS if args.groups == "all" else [item for item in SCENARIOS if item.group == args.groups]
    if args.tasks != "all":
        ids = [value.strip() for value in args.tasks.split(",") if value.strip()]
        known = {item.id for item in SCENARIOS}
        unknown = [value for value in ids if value not in known]
        if not ids or unknown:
            parser.error("unbekannte --tasks: " + ", ".join(unknown or ids))
        selected = [item for item in selected if item.id in ids]
        if not selected:
            parser.error("--tasks und --groups ergeben keine Tests")
    args.selected = selected
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        try:
            compare_reports(load_report(args.compare[0]), load_report(args.compare[1]))
        except Exception as exc:
            print(f"Vergleich fehlgeschlagen: {exc}", file=sys.stderr)
            return 2
        return 0

    key = args.api_key if args.api_key is not None else os.getenv("HAILO_API_KEY")
    context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    client = Client(args.url, key, args.timeout, context)
    try:
        model = resolve_model(client, args.model)
    except Exception as exc:
        print(f"Preflight fehlgeschlagen: {exc}", file=sys.stderr)
        return 2

    print(f"Modell: {model}")
    print("Kein Tool Calling, kein Home-Assistant Static-Context-Envelope.")
    print(f"Tests: {len(args.selected)} Szenarien × {args.repeats} Lauf/Läufe; max_tokens={args.max_tokens}")
    print("Intent-Format: INTENT|TARGET_TYPE|TARGET|VALUE\n")

    results = []
    for scenario in args.selected:
        for repeat in range(1, args.repeats + 1):
            result = run_one(client, scenario, model, args, repeat)
            results.append(result)
            status = "PASS" if result.get("ok") else "FAIL"
            reason = result.get("validation", {}).get("reason", "unknown")
            print(
                f"{scenario.id:28} run={repeat:2} {status:4} "
                f"total={result['client_total_ms']:8.1f} ms "
                f"infer={format_ms(result.get('inference_ms')):>8} "
                f"ttft={format_ms(result.get('ttft_ms')):>8} "
                f"in={str(result.get('input_tokens') if result.get('input_tokens') is not None else '-'):>4} "
                f"out={str(result.get('output_tokens') if result.get('output_tokens') is not None else '-'):>4} "
                f"{reason} | {detail(result)}"
            )

    summary = summarize(results)
    report = {
        "created_at": now(),
        "service_url": args.url,
        "model": model,
        "groups": args.groups,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "scenarios": [item.as_dict() for item in args.selected],
        "summary": summary,
        "results": results,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = model_slug(model)
    json_path = output / f"{stem}.json"
    csv_path = output / f"{stem}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, results)
    print_summary(model, summary)
    print(f"\nErgebnisse: {json_path} und {csv_path}")
    print("Nach dem Lauf mit dem zweiten VLM vergleichen mit:")
    print(
        "python3 scripts/benchmark-qwen-text-intent.py --compare "
        f"{args.output_dir}/Qwen2-VL-2B-Instruct.json "
        f"{args.output_dir}/Qwen3-VL-2B-Instruct.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
