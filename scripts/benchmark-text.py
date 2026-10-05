#!/usr/bin/env python3
"""Compare ten independent German text tasks on the service's LLM and VLM.

Python 3.10+, standard library only. Run --help for URL, authentication and
generation options. Saves a side-by-side HTML report and complete JSON results.
"""

import argparse
import html
import json
import math
import os
import shutil
import ssl
import sys
import textwrap
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

QUESTIONS = [
    ("1 · sehr einfach", "Kopfrechnen", "Was ist 7 + 5? Antworte nur mit der Zahl.", "12"),
    ("2 · einfach", "Übersetzung",
     "Übersetze ins Deutsche, ohne Erklärung: 'The window is open. Please turn off the light.'",
     "Das Fenster ist offen. Bitte schalte das Licht aus."),
    ("3 · einfach", "Begriffe erklären",
     "Erkläre den Unterschied zwischen Leistung in kW und Energie in kWh in genau drei kurzen Sätzen. "
     "Verwende als Beispiel ein Gerät mit 2 kW, das 3 Stunden läuft.",
     "Leistung versus Energie; 2 kW × 3 h = 6 kWh."),
    ("4 · mittel", "Strukturierte Extraktion",
     "Gib ausschließlich ein gültiges JSON-Objekt mit den Schlüsseln raum, temperatur_c, "
     "licht_an und fenster_offen aus. Daten: Im Wohnzimmer sind es 21,5 Grad Celsius. "
     "Das Licht ist ausgeschaltet und das Fenster ist offen. Zahlen und Booleans als JSON-Werte.",
     '{"raum":"Wohnzimmer","temperatur_c":21.5,"licht_an":false,"fenster_offen":true}'),
    ("5 · mittel", "Mehrstufiges Rechnen",
     "Ein Gerät verbraucht 4,2 kWh. Davon stammen 2,7 kWh aus eigener PV. Der übrige Strom "
     "kostet 0,35 Euro je kWh. Berechne Netzbezug und Kosten vor Rundung sowie auf Cent gerundet. "
     "Zeige den Rechenweg in höchstens vier Zeilen.",
     "Netzbezug 1,5 kWh; Kosten 0,525 Euro, gerundet 0,53 Euro."),
    ("6 · mittel bis schwer", "Zusammenfassen",
     "Fasse den folgenden Bericht in genau drei Stichpunkten zusammen. Nenne Ursache, Lösung und "
     "die noch offene Prüfung. Erfinde keine Informationen. Bericht: Am Montag antwortete der "
     "lokale Sprachdienst bei langen Anfragen nicht mehr. Die Bildanalyse arbeitete weiterhin. "
     "Die Untersuchung zeigte, dass der Textkontext das konfigurierte Tokenlimit überschritt "
     "und der Speicherverbrauch stark stieg. Am Dienstag wurde die Anzahl der Eingabetokens "
     "begrenzt und ältere vollständige Gesprächsrunden wurden entfernt. Kurze und lange "
     "Textanfragen werden seitdem wieder beantwortet. Die Bildmodelle blieben während der "
     "Änderung geladen. Ob parallele Kameraauswertung die Antwortzeit beeinflusst, ist noch "
     "nicht getestet. Messwerte sollen am Mittwoch gesammelt werden.",
     "Zu großer Textkontext; Eingabelimit/alte Gesprächsrunden entfernen; Parallelbetrieb noch prüfen."),
    ("7 · schwer", "Python-Fehler finden",
     "Diese Python-Funktion soll den Mittelwert aller Werte außer None berechnen. Bei keinem "
     "gültigen Wert soll sie None zurückgeben:\n"
     "def mean(values):\n    valid = [v for v in values if v]\n"
     "    return sum(valid) / len(values)\n"
     "Nenne beide Fehler und gib eine korrigierte Funktion an. Nenne außerdem das Ergebnis "
     "für [0, 10, None, 20]. Antworte kurz.",
     "0 wird fälschlich entfernt; falscher Nenner. valid mit 'is not None', leere Liste prüfen; Ergebnis 10."),
    ("8 · schwer", "Abhängigkeiten planen",
     "Vier Aufgaben haben folgende Dauer und Abhängigkeiten: A dauert 3 Minuten und hat keine "
     "Voraussetzung. B dauert 4 Minuten und beginnt erst nach A. C dauert 5 Minuten und beginnt "
     "erst nach A. D dauert 2 Minuten und beginnt erst nach B und C. Es gibt genügend Arbeitskräfte; "
     "B und C können parallel laufen. Alle Aufgaben sollen so früh wie möglich starten. "
     "Nenne Start und Ende jeder Aufgabe, Gesamtdauer und kritischen Pfad. Höchstens sechs Zeilen.",
     "A 0–3, B 3–7, C 3–8, D 8–10; 10 Minuten; kritischer Pfad A–C–D."),
    ("9 · sehr schwer", "Logische Schlussfolgerung",
     "Drei geschlossene Kisten enthalten nur Äpfel, nur Orangen oder eine Mischung aus beiden. "
     "Die Etiketten 'Äpfel', 'Orangen' und 'Mischung' sind alle falsch. Du darfst aus genau einer "
     "Kiste genau eine Frucht ziehen. Aus welcher beschrifteten Kiste musst du ziehen? Erkläre "
     "für den Fall, dass du einen Apfel ziehst, den tatsächlichen Inhalt aller drei Kisten. "
     "Begründe jeden Schritt und halte dich an höchstens fünf Sätze.",
     "Aus 'Mischung' ziehen; bei Apfel ist diese nur Äpfel, 'Orangen' ist Mischung, 'Äpfel' nur Orangen."),
    ("10 · sehr schwer", "Optimierung mit Bedingungen",
     "Ein Akku enthält 3,0 kWh nutzbare Energie. Drei Verbraucher stehen zur Auswahl: "
     "A: 1,2 kWh, Nutzen 5; B: 1,8 kWh, Nutzen 8; C: 1,0 kWh, Nutzen 6. "
     "Jeder Verbraucher kann genau einmal vollständig betrieben werden; Teilbetrieb ist nicht "
     "erlaubt. C darf nur betrieben werden, wenn auch A betrieben wird. Wähle die zulässige "
     "Kombination mit dem höchsten Gesamtnutzen, ohne das Energiebudget zu überschreiten. "
     "Liste die zulässigen Kombinationen mit Energie und Nutzen auf und begründe das Optimum. "
     "Höchstens acht Zeilen.",
     "Leer 0/0, A 1,2/5, B 1,8/8, A+B 3,0/13, A+C 2,2/11. Optimum A+B; C und B+C unzulässig; A+B+C zu groß."),
]


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


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


class RequestError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


class Client:
    def __init__(self, url, key, timeout, context):
        self.url, self.key, self.timeout, self.context = url, key, timeout, context

    def json(self, path, payload=None):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise RequestError(exc.code, detail) from exc


def measure(client, model, prompt, args):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": args.max_tokens, "temperature": args.temperature,
               "seed": args.seed, "language": "de", "stream": False}
    if args.max_input_tokens is not None:
        payload["max_input_tokens"] = args.max_input_tokens
    result = {"model": model, "requested_at": timestamp(), "request": payload}
    started = time.perf_counter()
    try:
        data = client.json("/v1/chat/completions", payload)
        result["response"] = data
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Die Antwort enthält keinen Text")
        if data.get("model", model) != model:
            raise ValueError("Der Server hat ein anderes als das angeforderte Modell verwendet")
        result.update(ok=True, answer=text, finish_reason=data["choices"][0].get("finish_reason"),
                      metrics=data.get("metrics") or {}, usage=data.get("usage") or {})
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        timeout = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}", answer="", metrics={}, usage={},
                      timed_out=timeout or getattr(exc, "status", None) == 504)
    finally:
        result.update(responded_at=timestamp(), client_total_ms=(time.perf_counter() - started) * 1000)
    return result


def ms(value):
    return f"{value:.1f} ms ({value / 1000:.3f} s)" if number(value) else "nicht verfügbar"


def metric_rows(result):
    metrics, usage = result.get("metrics", {}), result.get("usage", {})

    def count(kind, fallback):
        value = metrics.get(kind + "_tokens", usage.get(fallback))
        source = metrics.get(kind + "_tokens_source")
        return str(value) + (f" ({source})" if source else "") if number(value) else "nicht verfügbar"

    return [
        ("Status", "OK" if result["ok"] else "FEHLER"),
        ("Anfrage (Client, UTC)", result["requested_at"]),
        ("Antwort (Client, UTC)", result["responded_at"]),
        ("Gesamtdauer (Client)", ms(result["client_total_ms"])),
        ("Bearbeitung (Server)", ms(metrics.get("processing_ms"))),
        ("Modelllaufzeit", ms(metrics.get("inference_ms"))),
        ("TTFT (Modell)", ms(metrics.get("ttft_ms"))),
        ("TTFT-Quelle", metrics.get("ttft_source", "nicht verfügbar")),
        ("Input-Token", count("input", "prompt_tokens")),
        ("Output-Token", count("output", "completion_tokens")),
        ("Prefill token/s", f'{metrics["prefill_tokens_per_second"]:.2f}'
         if number(metrics.get("prefill_tokens_per_second")) else "nicht verfügbar"),
        ("Decode token/s", f'{metrics["decode_tokens_per_second"]:.2f}'
         if number(metrics.get("decode_tokens_per_second")) else "nicht verfügbar"),
        ("Endegrund", result.get("finish_reason") or "nicht verfügbar"),
        ("Anfrage (Server, UTC)", metrics.get("requested_at", "nicht verfügbar")),
        ("Antwort (Server, UTC)", metrics.get("responded_at", "nicht verfügbar")),
        ("Request-ID", metrics.get("request_id", "nicht verfügbar")),
    ]


def display_width(text):
    return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
               for char in text)


def paired_text(left, right, width):
    def lines(value):
        return [wrapped for line in value.expandtabs(4).split("\n")
                for wrapped in (textwrap.wrap(line, width=width) or [""])]

    a, b = lines(left), lines(right)
    for index in range(max(len(a), len(b))):
        first = a[index] if index < len(a) else ""
        second = b[index] if index < len(b) else ""
        print(first + " " * max(0, width - display_width(first)) + " │ " + second)


def print_pair(case):
    width = max(32, min(90, (shutil.get_terminal_size((150, 40)).columns - 3) // 2))
    print(f'\nFrage {case["number"]}/10 · {case["difficulty"]} · {case["title"]}')
    print(case["prompt"] + "\n")
    left, right = case["results"]["llm"], case["results"]["vlm"]
    paired_text("LLM · " + left["model"], "VLM · " + right["model"], width)
    paired_text(left["answer"] if left["ok"] else left["error"],
                right["answer"] if right["ok"] else right["error"], width)
    print()
    paired_text("\n".join(f"{k}: {v}" for k, v in metric_rows(left)),
                "\n".join(f"{k}: {v}" for k, v in metric_rows(right)), width)
    print(flush=True)


def summary(cases, role):
    values = [case["results"][role] for case in cases if role in case["results"]]
    successful = [value for value in values if value["ok"]]
    totals = [value["client_total_ms"] for value in successful]
    return {"completed": len(values), "successful": len(successful), "failed": len(values) - len(successful),
            "median_client_ms": median(totals) if totals else None,
            "total_client_ms": sum(value["client_total_ms"] for value in values)}


def render_html(report):
    escape = lambda value: html.escape(str(value), quote=True)  # noqa: E731
    cards = []
    for case in report["cases"]:
        answers, rows = [], []
        results = case["results"]
        for role, label in (("llm", "LLM"), ("vlm", "VLM")):
            value = results.get(role)
            if value is None:
                answers.append(f"<article><h3>{label}</h3><p>Noch nicht ausgeführt.</p></article>")
            else:
                text = value["answer"] if value["ok"] else value["error"]
                answers.append(f'<article><h3>{label} · {escape(value["model"])}</h3><pre>{escape(text)}</pre></article>')
        columns = {role: dict(metric_rows(value)) for role, value in results.items()}
        if columns:
            labels = next(iter(columns.values()))
            for label in labels:
                a = columns.get("llm", {}).get(label, "nicht ausgeführt")
                b = columns.get("vlm", {}).get(label, "nicht ausgeführt")
                rows.append(f"<tr><th>{escape(label)}</th><td>{escape(a)}</td><td>{escape(b)}</td></tr>")
        cards.append(f'<section><h2>{case["number"]}. {escape(case["title"])} · {escape(case["difficulty"])}</h2>'
                     f'<p class="prompt">{escape(case["prompt"])}</p><div class="compare">{"".join(answers)}</div>'
                     '<div class="scroll"><table><thead><tr><th>Messwert</th><th>LLM</th><th>VLM</th></tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table></div><details><summary>Referenz zur manuellen Prüfung</summary>'
                     f'<p>{escape(case["reference"])}</p></details></section>')
    stats = []
    for role, label in (("llm", "LLM"), ("vlm", "VLM")):
        value = summary(report["cases"], role)
        stats.append(f'<p>{label}: {value["successful"]} erfolgreich, {value["failed"]} Fehler; '
                     f'Median der Gesamtdauer: {escape(ms(value["median_client_ms"]))}</p>')
    params = json.dumps(report["parameters"], ensure_ascii=False)
    return f'''<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hailo Text-Benchmark</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;background:#f3f5f8;color:#172033;margin:0}}
main{{max-width:1400px;margin:auto;padding:24px}}section,header{{background:white;border:1px solid #d5dce6;border-radius:12px;padding:22px;margin:18px 0}}
h1{{margin-top:0}}h2{{font-size:1.2rem}}h3{{font-size:1rem;color:#145c4a}}.prompt,pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}article{{min-width:0}}pre{{font:inherit;background:#f5f7fa;padding:16px;border-radius:8px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #d5dce6;padding:7px 10px;overflow-wrap:anywhere}}
th{{font-weight:600}}td{{width:35%}}summary{{cursor:pointer;margin-top:14px}}.meta{{font-size:.85rem;color:#526078}}
@media(max-width:650px){{main{{padding:10px}}section,header{{padding:12px}}.compare{{gap:10px}}pre{{padding:10px}}}}</style>
<main><header><h1>Hailo-10H: Text-Benchmark LLM ↔ VLM</h1>
<p>{escape(report["url"])} · Start: {escape(report["started_at"])} · Status: {escape(report["status"])}</p>
<p class="meta">{escape(params)}</p>{"".join(stats)}
<p class="meta">Jede Frage: frischer Kontext, identische Parameter, Modelle nacheinander, Reihenfolge abwechselnd.
Clientdauer umfasst Netzwerk und JSON-Verarbeitung. TTFT stammt vom Modell bzw. ersten Textblock;
fehlende Werte werden nicht geschätzt. Tokenquellen: native = Modellzähler; tokenizer = nachtokenisierter Text;
tokenizer_text = Textprompt ohne Bildtokens. Endegrund length bedeutet, dass das Ausgabelimit erreicht wurde.
Die Fragen haben unterschiedliche Antwortlängen; Median und Gesamtdauer beschreiben nur diesen Testmix.
Referenzen dienen der manuellen Qualitätsprüfung; es wird kein automatischer Qualitätsscore berechnet.</p></header>
{"".join(cards)}</main></html>'''


def save_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report["summary"] = {role: summary(report["cases"], role) for role in ("llm", "vlm")}
    for name, content in (("results.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                          ("comparison.html", render_html(report))):
        temporary = directory / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("HAILO_URL", "http://127.0.0.1:8090"),
                        help="Service-Adresse; /v1 am Ende wird akzeptiert (auch HAILO_URL)")
    parser.add_argument("--api-key", help="Bearer-API-Key; alternativ HAILO_API_KEY oder HAILO_KEY")
    parser.add_argument("--llm-model", help="Standard: llm_model aus /ui/config")
    parser.add_argument("--vlm-model", help="Standard: vlm_model aus /ui/config")
    parser.add_argument("--max-tokens", type=int, default=256, help="Identisches Ausgabelimit für beide Modelle (1..1024)")
    parser.add_argument("--max-input-tokens", type=int, help="Optional niedrigeres Eingabelimit; Standard: Service-Limits")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=240, help="Timeout pro HTTP-Anfrage in Sekunden")
    parser.add_argument("--warmup", action="store_true", help="Eine zusätzliche, nicht gewertete Anfrage je Modell")
    parser.add_argument("--output-dir", type=Path, help="Berichtsordner; Standard: benchmark-results/Zeitstempel")
    parser.add_argument("--ca-file", help="CA-/Zertifikatsdatei für HTTPS")
    parser.add_argument("--insecure", action="store_true", help="Zertifikatsprüfung bei eigenem HTTPS-Zertifikat deaktivieren")
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 1024:
        parser.error("--max-tokens muss zwischen 1 und 1024 liegen")
    if args.max_input_tokens is not None and not 1 <= args.max_input_tokens <= 131072:
        parser.error("--max-input-tokens muss zwischen 1 und 131072 liegen")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 1:
        parser.error("--temperature muss zwischen 0 und 1 liegen")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout muss positiv sein")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed muss zwischen 0 und 4294967295 liegen")
    if args.ca_file and args.insecure:
        parser.error("--ca-file und --insecure schließen sich aus")
    try:
        args.url = normalize_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context(cafile=args.ca_file)
    except OSError as exc:
        print(f"HTTPS-Konfiguration fehlgeschlagen: {exc}", file=sys.stderr)
        return 2
    key = args.api_key if args.api_key is not None else os.environ.get("HAILO_API_KEY") or os.environ.get("HAILO_KEY", "")
    client = Client(args.url, key, args.timeout, context)
    directory = args.output_dir or Path("benchmark-results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        config = client.json("/ui/config")
        available = {entry["id"] for entry in client.json("/v1/models")["data"]}
        models = {"llm": args.llm_model or config["llm_model"], "vlm": args.vlm_model or config["vlm_model"]}
        if models["llm"] == models["vlm"]:
            raise ValueError("LLM und VLM müssen unterschiedliche Modelle sein")
        for role, model in models.items():
            if model not in available:
                raise ValueError(f"{role.upper()} {model} ist nicht verfügbar; verfügbare Modelle: {', '.join(sorted(available))}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Service-Prüfung fehlgeschlagen: {exc}\n"
              "Service-Adresse mit --url und API-Key mit --api-key angeben.", file=sys.stderr)
        return 2
    report = {"url": args.url, "started_at": timestamp(), "status": "läuft", "models": models,
              "parameters": {"max_tokens": args.max_tokens, "max_input_tokens": args.max_input_tokens,
                             "temperature": args.temperature, "seed": args.seed, "warmup": args.warmup,
                             "stream": False, "question_count": len(QUESTIONS), "timeout_seconds": args.timeout},
              "cases": [], "warmup": {}}
    print(f'Service: {args.url}\nLLM: {models["llm"]}\nVLM: {models["vlm"]}\n'
          f'20 Messanfragen, frischer Kontext, max_tokens={args.max_tokens}.\nBerichte: {directory.resolve()}', flush=True)
    try:
        save_report(report, directory)
        if args.warmup:
            for role, model in models.items():
                print(f"Aufwärmen {role.upper()} …", flush=True)
                value = measure(client, model, "Antworte nur mit dem Wort bereit.", args)
                report["warmup"][role] = value
                save_report(report, directory)
                if not value["ok"]:
                    print(f'Aufwärmen fehlgeschlagen: {value["error"]}', file=sys.stderr)
                    report["status"] = "Aufwärmen fehlgeschlagen"
                    save_report(report, directory)
                    return 1
        for index, (difficulty, title, prompt, reference) in enumerate(QUESTIONS, 1):
            order = ("llm", "vlm") if index % 2 else ("vlm", "llm")
            case = {"number": index, "difficulty": difficulty, "title": title, "prompt": prompt,
                    "reference": reference, "order": list(order), "results": {}}
            report["cases"].append(case)
            for role in order:
                print(f"[{index}/10] {title}: {role.upper()} …", flush=True)
                case["results"][role] = measure(client, models[role], prompt, args)
                save_report(report, directory)
                if case["results"][role].get("timed_out"):
                    report["status"] = "abgebrochen nach Timeout"
                    print("Timeout: keine weiteren Messungen, da native Modellarbeit noch laufen kann.", file=sys.stderr)
                    break
            if len(case["results"]) == 2:
                print_pair(case)
            if report["status"] == "abgebrochen nach Timeout":
                break
        if report["status"] == "läuft":
            report["status"] = "abgeschlossen"
    except KeyboardInterrupt:
        report["status"] = "abgebrochen"
        print("\nAbgebrochen; abgeschlossene Messungen bleiben im Bericht.", file=sys.stderr)
    except OSError as exc:
        print(f"Berichte konnten nicht gespeichert werden: {exc}", file=sys.stderr)
        return 2
    report["finished_at"] = timestamp()
    try:
        save_report(report, directory)
    except OSError as exc:
        print(f"Berichte konnten nicht gespeichert werden: {exc}", file=sys.stderr)
        return 2
    for role in ("llm", "vlm"):
        value = report["summary"][role]
        print(f'{role.upper()}: {value["successful"]}/{value["completed"]} erfolgreich; '
              f'Median Clientdauer: {ms(value["median_client_ms"])}')
    print(f'\nVergleich im Browser: {(directory / "comparison.html").resolve()}\n'
          f'Rohdaten: {(directory / "results.json").resolve()}')
    if report["status"] == "abgebrochen":
        return 130
    return 1 if any(value["failed"] for value in report["summary"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
