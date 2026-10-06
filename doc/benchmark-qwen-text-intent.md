# Qwen2-VL vs Qwen3-VL: Text- und Intent-Benchmark

Dieser Benchmark prüft, ob der ohnehin resident auf Hailo-10H laufende VLM als schneller Text-Fast-Path sinnvoll eingesetzt werden kann.

Er misst bewusst **kein OpenAI Tool Calling**. Die Requests enthalten weder Tools noch den Home-Assistant-Produktions-Envelope mit `Static Context:`. Damit wird tatsächlich der ausgewählte VLM gemessen und nicht der deterministische HA-Router.

## Was wird getestet?

Der Benchmark enthält zwei getrennte Gruppen:

- `text`: kurze, eindeutig bewertbare Textaufgaben wie `2 + 2`, Hauptstadt Frankreich, einfache Multiplikation und Übersetzung.
- `intent`: kompakte Home-Assistant-Intent-/Slot-Extraktion im Format `INTENT|TARGET_TYPE|TARGET|VALUE`.

Beispiele:

```text
Schalte das Licht im Wohnzimmer auf 70 Prozent.
```

Erwartet:

```text
LIGHT_BRIGHTNESS|area|Wohnzimmer|70
```

und:

```text
Starte Deebot mini.
```

Erwartet:

```text
VACUUM_START|name|Deebot mini|-
```

Die Auswertung ist absichtlich streng. Sichere Groß-/Kleinschreibungsnormalisierung wird akzeptiert, aber umformulierte oder erfundene Ziele, zusätzliche Antworten und falsche Werte nicht.

## Warum zwei Läufe?

Der Service hält genau einen VLM resident auf Hailo. Qwen2-VL und Qwen3-VL werden deshalb nicht gleichzeitig geladen. Für einen fairen Vergleich wird derselbe Benchmark einmal mit Qwen3-VL und einmal mit Qwen2-VL ausgeführt. Die Ergebnisse werden modellbezogen gespeichert und anschließend verglichen.

## Lauf mit dem aktuell residenten Modell

Nach `git pull` kann das Skript das aktuell geladene Qwen-Modell automatisch über `/v1/models` erkennen:

```bash
python3 scripts/benchmark-qwen-text-intent.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY
```

Alternativ das erwartete Modell explizit angeben:

```bash
python3 scripts/benchmark-qwen-text-intent.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --model Qwen3-VL-2B-Instruct
```

Standardmäßig werden alle 16 Szenarien einmal ausgeführt und maximal 32 Output-Tokens erlaubt. Das kurze Output-Limit verhindert, dass ein Modell bei einer eigentlich kompakten Klassifikationsaufgabe lange weitergeneriert.

Nur normale Textaufgaben:

```bash
python3 scripts/benchmark-qwen-text-intent.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --groups text
```

Nur Intent-Extraktion:

```bash
python3 scripts/benchmark-qwen-text-intent.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --groups intent
```

Für belastbarere Messwerte können mehrere Läufe verwendet werden:

```bash
python3 scripts/benchmark-qwen-text-intent.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --repeats 3
```

## Modell wechseln

Beispielkonfiguration für Qwen3-VL:

```yaml
models:
  vlm:
    enabled: true
    model: Qwen3-VL-2B-Instruct
```

Beispiel für Qwen2-VL:

```yaml
models:
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
```

Nach dem Wechsel:

```bash
sudo systemctl restart hailo-10h-services
```

Danach denselben Benchmark erneut ausführen.

## Ergebnisdateien

Die letzten Ergebnisse jedes Modells werden getrennt abgelegt:

```text
qwen-text-intent-benchmark/Qwen2-VL-2B-Instruct.json
qwen-text-intent-benchmark/Qwen2-VL-2B-Instruct.csv
qwen-text-intent-benchmark/Qwen3-VL-2B-Instruct.json
qwen-text-intent-benchmark/Qwen3-VL-2B-Instruct.csv
```

Enthalten sind unter anderem:

- korrekte/falsche Antwort pro Szenario,
- vollständige Modellantwort,
- Input-Tokens,
- Input-Budget-Tokens,
- Output-Tokens,
- TTFT,
- Modell-Inference-Zeit,
- Server-/Processing-Zeit, soweit vom Service geliefert,
- gesamte HTTP-Laufzeit.

## Qwen2 und Qwen3 vergleichen

Sobald beide JSON-Dateien existieren:

```bash
python3 scripts/benchmark-qwen-text-intent.py --compare \
  qwen-text-intent-benchmark/Qwen2-VL-2B-Instruct.json \
  qwen-text-intent-benchmark/Qwen3-VL-2B-Instruct.json
```

Die Vergleichsansicht zeigt Accuracy und mittlere Laufzeiten getrennt für `text`, `intent` und insgesamt sowie alle Fehlversuche mit der tatsächlichen Antwort.

## Entscheidungskriterium

Der wichtigste Wert für den späteren HA-Fast-Path ist `intent`-Accuracy. Ein schneller VLM ist nur dann sinnvoll, wenn die Extraktion sehr zuverlässig ist. Der Benchmark wandelt absichtlich keine semantisch falsche Ausgabe automatisch in einen echten HA-Toolcall um.
