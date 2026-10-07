# Modellvergleich für Raspberry Pi 5 + Hailo-10H / Model comparison for Raspberry Pi 5 + Hailo-10H

> Stand: 6. Oktober 2026; Dokumentation ergänzt am 7. Oktober 2026. Die Ergebnisse stammen aus Messungen mit Hailo-10H-Services auf demselben Raspberry-Pi-5/CM5-System mit Hailo-10H. Sie sind eine praxisnahe Momentaufnahme dieses Setups und **kein allgemeiner Hersteller-Benchmark**. Es wurden für diese Dokumentationsänderung keine neuen Messungen durchgeführt.
>
> Snapshot: October 6, 2026; documentation updated October 7, 2026. Results were measured with Hailo-10H-Services on the same Raspberry Pi 5/CM5 system with Hailo-10H. They are a practical snapshot of this setup, **not a general vendor benchmark**. No new measurements were performed for this documentation update.

[Deutsch](#deutsch) · [English](#english)

GitHub-Markdown unterstützt keine portable clientseitige Tabellensortierung.
Deshalb enthält diese Seite bereits vorsortierte Tabellen und kompakte
Empfehlungen nach Einsatzprofil.

GitHub Markdown does not support portable client-side table sorting. This page
therefore provides presorted tables and compact recommendations by workload.

---

# Deutsch

## Ziel dieser Auswertung

Diese Seite bewertet die getesteten Modelle für die **allgemeine Nutzung von Hailo-10H-Services auf einem Raspberry Pi 5 mit Hailo-10H**. Im Mittelpunkt stehen:

- Antwortqualität und Korrektheit,
- Reaktionszeit und Time-to-First-Token (TTFT),
- nutzbare Kontext- und Input-Grenzen,
- Sprachqualität in Deutsch und Englisch,
- Reasoning und elementare Logik,
- Halluzinationen und Umgang mit fehlenden Informationen,
- strukturierte Ausgabe wie JSON,
- Verhalten bei kurzen und längeren Antworten,
- Eignung für lokalen Chat, Wissensfragen, RAG, Dokumentkontext, Automatisierung und multimodale Workloads.

Anwendungsspezifische Home-Assistant-Intent-Tests werden hier bewusst **nicht** bewertet und fließen weder in Rankings noch Empfehlungen ein. Für den quantitativen Vergleich wird nur der allgemeine Textteil des Benchmarks verwendet.

## Nicht getestetes Function-Calling-Modell

**`Qwen2-1.5B-Instruct-Function-Calling-v1` wurde für diesen Vergleich nicht
herangezogen.** Es gibt dafür hier keine gemessene Accuracy, Latenz oder
Sprachbewertung; es gehört nicht in die Ranglisten der getesteten Modelle.

Da die sprachlichen Ergebnisse von **Qwen2.5-1.5B-Instruct** bereits nicht
überzeugten, wurde auf den Einsatz und Test des Function-Calling-Modells als
universelles Sprachmodell für den Service verzichtet. Das war eine Entscheidung
über den Testumfang, **kein nachgewiesenes Qualitätsurteil über das ungetestete
Modell**; die Ergebnisse von Qwen2.5 lassen sich nicht als dessen Messwerte ausgeben.

Als spezialisierter Kandidat könnte es vor allem für **Home-Assistant-Aufgaben
mit Function Calling** interessant sein. Diese Eignung ist eine Vermutung und
müsste mit echten Tool-Schemata und Ziel-/Wertevalidierung separat getestet
werden. Voraussetzung ist, dass die Anfrage einschließlich Tool-Beschreibungen,
Katalog, Historie und reservierter Ausgabe in den **2048-Token-Gesamtkontext**
passt. Function Calling hebt diese Grenze nicht auf.

## Kurzfazit

Für einen allgemeinen lokalen KI-Dienst auf Raspberry Pi 5 + Hailo-10H ergibt sich aus den vorliegenden Messungen:

1. **Gemma 4 E2B** liefert die beste allgemeine Textqualität, die höchste Korrektheit im kontrollierten Kurztext-Test und das stärkste Reasoning. Der Preis dafür ist deutlich höhere CPU-Latenz und langsames Decoding.
2. **Qwen3-1.7B-Instruct** bietet im getesteten Setup das beste Verhältnis aus Qualität und Geschwindigkeit unter den nativen Hailo-LLMs. Es ist die derzeit sinnvollste Standardwahl für schnelle allgemeine Textanfragen auf dem Hailo-10H.
3. **Qwen2.5-1.5B-Instruct** ist noch etwas schneller, verliert aber sichtbar bei Sprachverständnis, Instruction Following und komplexeren Aufgaben. Es eignet sich eher für kurze, einfache Anfragen als für einen universellen Assistenten.
4. **Llama3.2-1B-Instruct** ist schnell, qualitativ aber deutlich schwächer. Das sichtbare `<|eot_id|>` wird hier nicht als Modellfehler gewertet, weil es technisch abgefangen werden kann; auch nach dessen Entfernung bleiben mehrere Antworten falsch.
5. **DeepSeek-R1-Distill-Qwen-1.5B** ist für einen interaktiven Low-Latency-Dienst in dieser Konfiguration wenig attraktiv: es denkt selbst bei trivialen Fragen sehr lange, produziert unnötig viel Text und zeigt besonders auf Deutsch deutliche Schwächen.
6. **Qwen2-VL** ist bei Text extrem schnell, zeigt aber bei längeren Antworten deutliche Sprach-/Decoder-Artefakte. Als allgemeines Textmodell ist es deshalb nicht zu empfehlen.
7. **Qwen3-VL** ist sprachlich stabiler als Qwen2-VL, aber langsamer und kann in Wiederholungsschleifen geraten. Seine eigentliche Stärke sollte in Bild-/Multimodal-Aufgaben bewertet werden; die vorliegenden Texttests reichen nicht aus, um seine Bildqualität zu beurteilen.

### Praktische Standardempfehlung

Für einen Raspberry Pi 5 mit Hailo-10H ist ein **mehrstufiger Dienst** sinnvoller als der Versuch, ein einziges kleines Modell für alles zu verwenden:

- **Qwen3-1.7B-Instruct auf Hailo** für schnelle allgemeine Textanfragen,
- **Gemma 4 E2B auf CPU** für schwierigere Sprach- und Reasoning-Aufgaben oder wenn mehr Kontext benötigt wird,
- **Qwen-VL auf Hailo** für Bilder und Video-Frames,
- **deterministische Funktionen/Tools** für Mathematik, Datum, aktuelle Daten und andere Aufgaben, die exakt berechnet oder abgefragt werden können,
- **RAG/Recherche** für Fakten, die nicht zuverlässig aus dem Modellwissen stammen sollten.

Damit werden die Stärken des Raspberry Pi 5 und des Hailo-10H besser genutzt: Hailo übernimmt die latenzkritische Inferenz, während CPU und Tools nur dort eingesetzt werden, wo sie einen echten Qualitäts- oder Kontextgewinn bringen.

## Kontext- und Input-Grenzen: 2k auf Hailo vs. 4k bei Gemma

Die Kontextgröße ist für die praktische Nutzung mindestens so wichtig wie die reine Inferenzgeschwindigkeit.

### Native Hailo-Modelle: 2048 Token Kontext

Die derzeit getesteten nativen Hailo-LLMs und Qwen-VLMs arbeiten in diesem Service mit einem **2048-Token-Kontext**. Bei den Hailo-Modellen teilen sich **Input und Output denselben kompilierten Kontext**.

Das bedeutet praktisch:

```text
2048 Gesamt-Kontext
- reservierte Output-Tokens
- Template-/Spezialtoken-Overhead
= tatsächlich nutzbarer Input
```

Bei einem Benchmark mit `max_tokens=32` lag das effektive Input-Limit beispielsweise bei ungefähr **2015 Tokens**. Wird mehr Output reserviert, sinkt der maximal mögliche Input entsprechend. Bei `max_tokens=256` bleibt grob nur noch ein Bereich um **1,8k Input-Tokens** übrig.

Diese 2k-Grenze ist für folgende Aufgaben ausreichend:

- kurze Chatfragen,
- kompakte Systemprompts,
- kleine RAG-Ausschnitte,
- kurze Zusammenfassungen,
- Klassifikation,
- einfache lokale Assistenz,
- einzelne Bildfragen bei VLM-Nutzung, sofern der restliche Prompt klein bleibt.

Sie wird aber schnell zum Engpass bei:

- längerer Chat-Historie,
- großen Systemprompts,
- mehreren Dokumentpassagen,
- umfangreichem RAG-Kontext,
- langen Codeausschnitten,
- großen JSON-Schemata,
- Agenten mit vielen Tool-Beschreibungen,
- langen Antworten, weil diese denselben Kontext verbrauchen.

Ein Hailo-Modell kann daher zwar in deutlich unter zwei Sekunden reagieren, aber nicht automatisch einen langen Gesprächs- oder Dokumentkontext verarbeiten.

### Gemma 4 E2B: 4096 Token Input-Grenze

Gemma läuft in diesem Setup über LiteRT-LM auf der Raspberry-Pi-CPU mit einer bewusst gesetzten **maximalen Input-Grenze von 4096 Tokens**.

Damit steht ungefähr der doppelte Prompt-Spielraum zur Verfügung wie bei den 2k-Hailo-Modellen. Praktisch ist das vor allem hilfreich für:

- längere Gesprächshistorien,
- größere RAG-Chunks oder mehrere Retrieval-Treffer,
- komplexere Instruktionen,
- längere Dokumentausschnitte,
- Aufgaben, bei denen mehr Kontext für Reasoning benötigt wird.

Die größere Grenze ist jedoch nicht kostenlos. Mehr Input erhöht Prefill-Zeit und Speicherbedarf. Frühere Versuche mit **8192 Input-Tokens** führten auf diesem Raspberry-Pi-Setup zu stark wachsendem RAM-Verbrauch und letztlich zu Instabilität/Absturz. Deshalb ist **4096** hier eine bewusste praktische Stabilitätsgrenze und nicht nur ein beliebiger Benchmarkwert.

### Konsequenz für allgemeine Nutzung

| Workload | 2k Hailo-Kontext | 4k Gemma-Kontext |
|---|---|---|
| kurze Frage/Antwort | **sehr gut** | gut, aber langsamer |
| kurze Chat-Historie | **gut** | **sehr gut** |
| längere Chat-Historie | eingeschränkt | **besser** |
| kleines RAG | **gut** | **sehr gut** |
| mehrere RAG-Passagen | schnell am Limit | **deutlich flexibler** |
| lange Dokumente | Chunking zwingend | Chunking weiterhin nötig, aber weniger aggressiv |
| große Tool-/Schema-Prompts | problematisch | besser, aber 4k bleibt klein gegenüber Cloud-LLMs |
| lange Antwort + langer Input | stark eingeschränkt | flexibler |

Für einen produktiven lokalen Dienst sollte deshalb **Retrieval und Prompt-Compaction modellabhängig** sein. Ein Request, der für Gemma mit 3.500 Tokens problemlos passt, kann nicht unverändert an Qwen3-1.7B auf Hailo geschickt werden.

### Routing nach Kontextlänge

Eine sinnvolle allgemeine Strategie ist:

```text
kurzer Prompt / kurze Historie
        │
        ▼
Qwen3-1.7B auf Hailo
        │
        ├── schnell, ca. 2k Kontext
        │
        └── ideal für interaktive Standardanfragen

längerer Prompt / mehr RAG-Kontext
        │
        ▼
Gemma 4 E2B auf CPU
        │
        ├── langsamer
        └── bis 4k Input im stabilen Setup
```

Für noch längere Inhalte müssen **beide** Pfade mit Chunking, Retrieval, Zusammenfassung oder hierarchischer Verarbeitung arbeiten. Auch 4k sind für vollständige Dokumente, große Codebasen oder lange Gesprächsarchive klein.

## Testbasis und Vergleichbarkeit

Es wurden zwei Arten von Daten verwendet:

- **Kontrollierter Textbenchmark**: sechs kurze, eindeutig bewertbare Aufgaben aus dem [historischen Benchmark-Skript](https://github.com/leonsio/hailo-10h-services/blob/4e6a4b4445890471e323cf8667e65d355e683548/scripts/benchmark-qwen-text-intent.py) (inzwischen entfernt), mit `temperature=0.1` und `max_tokens=32`. Dieser Teil ist die quantitative Basis für Accuracy, mittlere Laufzeit und TTFT.
- **Explorative WebGUI-Läufe**: Rechnen, Hauptstädte, Wetter, kreative Texte, Zählen, JSON sowie Datums-/Kalenderfragen in Deutsch und Englisch. Diese Läufe zeigen reales Verhalten, sind aber nicht immer vollständig isoliert. Teilweise wächst die Chat-Historie und damit die Zahl der Input-Tokens; spätere Laufzeiten dürfen deshalb nicht als reiner Modell-Speed-Benchmark interpretiert werden.

### Sonderbehandlung Llama3.2

Llama3.2 gibt im aktuellen Service teilweise `<|eot_id|>` sichtbar zurück. Das ist ein Stop-Token-/Cleanup-Thema und kann deterministisch entfernt werden. Für die **inhaltliche** Bewertung wird dieses Token ignoriert.

Der ursprüngliche Skript-Score war dadurch formal `0/6`. Inhaltlich sind jedoch `Paris` und `23` korrekt. `2+2 → 2`, `17×6 → 34`, `nach Montag → Montag` und `house → Entferntes Haus` bleiben echte Fehler. Für diese Seite wird Llama daher mit **2/6 = 33,3 %** bewertet.

## Vollständige Fragenübersicht und Testsprache

### Sechs kontrollierte Textaufgaben

Diese sechs deutschen Benutzerprompts bilden die quantitative Vergleichsbasis.
Der gemeinsame Systemtext lautete:

> Isolierter Sprachmodell-Benchmark. Folge der Benutzeranweisung exakt. Antworte nur mit der verlangten kurzen Antwort, ohne Erklärung, Markdown oder Präambel.

| ID | Originalfrage | Erwartete Antwort |
|---|---|---|
| C01 | Was ist 2 + 2? Antworte ausschließlich mit der Zahl. | `4` |
| C02 | Was ist die Hauptstadt von Frankreich? Antworte nur mit dem Stadtnamen. | `Paris` |
| C03 | Was ist 17 mal 6? Antworte ausschließlich mit der Zahl. | `102` |
| C04 | Welcher Wochentag kommt nach Montag? Antworte nur mit dem Wochentag. | `Dienstag` |
| C05 | Welche Zahl ist größer: 17 oder 23? Antworte nur mit der Zahl. | `23` |
| C06 | Übersetze das englische Wort house ins Deutsche. Antworte nur mit einem Wort. | `Haus` |

### Explorative WebGUI-Fragen

Die folgenden Originalfragen ergänzen die qualitative Bewertung. Sie sind
keine zusätzlichen Aufgaben im 6-Fragen-Accuracy-Score. Nicht jedes Modell
erhielt jede Sprachvariante; wiederholte Fragen sind keine unabhängigen Messungen.

| ID | Originalfrage | Prüfziel / Referenz |
|---|---|---|
| E01 | Berechne 2+2 gib nur die Antwort aus | 4 |
| E02 | Berechne 2+3 gib nur die Antwort aus | 5 |
| E03 | Was ist die Hauptstadt von Frankreich? | Paris |
| E04 | Was ist die Hauptstadt von Bolivien | Sucre; La Paz ist Regierungssitz |
| E05 | Schreibe genau drei Sätze über das Wetter heute. | Keine aktuellen Wetterdaten bereitgestellt |
| E06 | Erzähle eine sehr kurze, lustige Geschichte über eine Katze namens Miau (maximal 50 Wörter). | Kohärenz und Wortlimit |
| E07 | Zähle von 1 bis 20 auf. | Vollständige Folge 1–20 |
| E08 | Gib mir ein JSON-Objekt mit den Schlüsseln 'name' (Wert: 'Max') und 'alter' (Wert: 30). Keine Erklärung drumherum. | {"name":"Max","alter":30} |
| E09 | Heute ist 1. März 2028. Welches Wochentag und Datum waren vorgestern. | Montag, 28. Februar 2028 |

Die Grammatik in E09 ist absichtlich wie im Originalprotokoll erhalten.
E01/E02 wurden teilweise mit einem zusätzlichen Schlusspunkt gestellt; E04
auch mit Fragezeichen. E09 wurde auch ohne Schlusspunkt gestellt. Diese Satzzeichenvarianten zählen nicht als neue Aufgaben.

| ID | Zusätzlicher Originalprompt | Einsatz im Protokoll |
|---|---|---|
| E10 | Today is March 1, 2028. What day of the week and date was the day before yesterday? | Englische Kalenderfrage; teils nach Wechsel der UI-Sprache wiederholt. |
| E11 | Calculate 2+2 and just give the answer. | Englische Rechenfrage bei DeepSeek. |
| E12 | What is the capital of France? | Englische Faktenfrage bei DeepSeek, wiederholt. |
| E13 | What is the capital of Bolivia? | Englische Faktenfrage bei DeepSeek. |
| E14 | Count from 1 to 20. | Englische Zählaufgabe bei DeepSeek. |
| E15 | Rechte 2+2 | Tippfehler im Original bei Llama und DeepSeek; nicht stillschweigend korrigieren. |
| E16 | Addiere 2+2 .gib nur das Ergebnis aus | Zusätzliche Umformulierung bei Llama. |
| E17 | Gib mir ein JSON-Objekt mit den Schlüsseln 'name' (Wert: 'Max') und 'alter' (Wert: 30). Keine Erklärung drumherum. Gib nur eine Antwort aus | Zusätzliche JSON-Nachfrage bei Qwen3-VL. |

E10 ist eine tatsächlich getestete englische Variante. Die Übersetzungen im
englischen Dokumentteil dienen dagegen der Lesbarkeit und belegen **keinen**
zusätzlichen englischen Testlauf.

### HA-Intent-Fragen: dokumentiert, aber aus der Bewertung ausgeschlossen

Das historische Skript enthielt außerdem diese zehn Prompts mit dem Textformat
`INTENT\|TARGET_TYPE\|TARGET\|VALUE`. Dabei wurden keine OpenAI-Tools,
kein HA-Static-Context-Envelope und keine Geräteausführung verwendet.
Diese Aufgaben sind hier vollständig aufgeführt, fließen aber **nicht** in
die allgemeinen Qualitäts-/Geschwindigkeitsrankings ein und messen nicht die
Eignung des ungetesteten Function-Calling-Modells.

| ID | Originalfrage |
|---|---|
| H01 | Schalte das Licht im Wohnzimmer auf 70 Prozent. |
| H02 | Schalte das Licht in der Küche an. |
| H03 | Schalte die Nachttischlampe aus. |
| H04 | Stelle das Licht im Wohnzimmer auf rot. |
| H05 | Stelle die Temperatur im Schlafzimmer auf 19,5 Grad. |
| H06 | Fahre den Rollladen im Dachgeschoss auf 40 Prozent. |
| H07 | Schließe den Rollladen. |
| H08 | Starte Deebot mini. |
| H09 | Schalte die Kaffeemaschine an. |
| H10 | Wie warm ist es im Wohnzimmer? |

### Einfluss der deutschen Fragestellung

Die überwiegend deutschen Prompts dürften für Modelle mit stärkerer englischer
Sprachleistung höhere Anforderungen an **Input-Verständnis und
Instruktionsbefolgung** gestellt haben. Ohne einen isolierten, gleich aufgebauten
DE/EN-Vergleich lässt sich dieser Einfluss nicht beziffern.

Die Sprache ändert weder die zugrunde liegende Rechen-/Logikaufgabe noch die
grundsätzliche Reasoning-Fähigkeit des Modells. Sie kann aber beeinflussen,
wie zuverlässig das Modell die Aufgabe versteht und löst. Deshalb ist die
**gemessene Erfolgsquote keine sprachunabhängige Messung reiner
Reasoning-Fähigkeit**. Gemmas unterschiedliche Antworten auf die deutsche und
englische Kalenderfrage veranschaulichen diese Grenze; bei den explorativen
Läufen können zusätzlich Historie und UI-/Promptkontext unterschiedlich sein.

## Kontrollierter allgemeiner Textbenchmark

| Modell | Backend | Text korrekt | Ø Gesamtzeit | Ø TTFT | Kontext / Input | Charakter |
|---|---|---:|---:|---:|---|---|
| **Qwen2-VL-2B-Instruct** | Hailo VLM | 3/6 = 50,0 % | **637 ms** | **328 ms** | 2048 gesamt | extrem schnell, schwache Textrobustheit |
| **Qwen2.5-1.5B-Instruct** | Hailo LLM | 4/6 = 66,7 % | 934 ms | 381 ms | 2048 gesamt | schnell, begrenzte Sprach-/Logikqualität |
| **Llama3.2-1B-Instruct** | Hailo LLM | 2/6 = 33,3 %* | 959 ms | 686 ms | 2048 gesamt | schnell, aber viele Inhaltsfehler |
| **Qwen3-1.7B-Instruct** | Hailo LLM | **5/6 = 83,3 %** | 1.178 ms | 639 ms | 2048 gesamt | bester Hailo-Textkompromiss |
| **Qwen3-VL-2B-Instruct** | Hailo VLM | 4/6 = 66,7 % | 1.354 ms | 642 ms | 2048 gesamt | bessere Sprache als Qwen2-VL, langsamer |
| **Gemma 4 E2B** | CPU / LiteRT-LM | **6/6 = 100 %** | 2.369 ms | 2.005 ms | **4096 Input** | beste Qualität, höherer Kontext, höhere CPU-Latenz |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Hailo LLM | n/a** | 5.751 ms** | n/a | 2048 gesamt | starke Thinking-Overhead |

\* Inhaltliche Bewertung nach Entfernung des technisch abfangbaren `<|eot_id|>`.  
\** Im bereitgestellten DeepSeek-Journal waren die sechs isolierten Textrequests und ihre Laufzeiten vorhanden, aber nicht alle Benchmark-Response-Bodies in einer Form, aus der sich ein fairer Accuracy-Score rekonstruieren ließ.

## Rangliste nach allgemeiner Textqualität

| Rang | Modell | Accuracy im kontrollierten Texttest | Einordnung |
|---:|---|---:|---|
| 1 | **Gemma 4 E2B** | **100 %** | höchste Zuverlässigkeit im Test |
| 2 | **Qwen3-1.7B-Instruct** | **83,3 %** | beste native Hailo-Wahl |
| 3 | Qwen2.5-1.5B-Instruct | 66,7 % | schnell, aber deutlich schwächer |
| 3 | Qwen3-VL-2B-Instruct | 66,7 % | Text nur Nebenfunktion eines VLM |
| 5 | Qwen2-VL-2B-Instruct | 50,0 % | für generellen Text zu instabil |
| 6 | Llama3.2-1B-Instruct | 33,3 %* | niedrigste belastbare Textqualität |
| — | DeepSeek-R1-Distill-Qwen-1.5B | n/a | qualitative Bewertung siehe unten |

## Rangliste nach Reaktionsgeschwindigkeit

| Rang | Modell | Ø Gesamtzeit | Ø TTFT | Bewertung |
|---:|---|---:|---:|---|
| 1 | **Qwen2-VL-2B-Instruct** | **637 ms** | **328 ms** | schnellste Ausgabe, aber Qualitätsproblem |
| 2 | **Qwen2.5-1.5B-Instruct** | 934 ms | 381 ms | sehr guter Low-Latency-Pfad |
| 3 | Llama3.2-1B-Instruct | 959 ms | 686 ms | schnell, aber geringe Korrektheit |
| 4 | **Qwen3-1.7B-Instruct** | 1.178 ms | 639 ms | sehr guter Qualitäts-/Speed-Kompromiss |
| 5 | Qwen3-VL-2B-Instruct | 1.354 ms | 642 ms | akzeptabel für multimodale Nutzung |
| 6 | Gemma 4 E2B | 2.369 ms | 2.005 ms | merklich langsamer auf CPU |
| 7 | DeepSeek-R1-Distill-Qwen-1.5B | 5.751 ms | n/a | Thinking-Overhead dominiert |

## Qualitative Scorecard

Die folgenden 1–5-Werte sind **keine standardisierten Hersteller-Benchmarks**, sondern eine Zusammenfassung der vorliegenden Antworten. `5` bedeutet „innerhalb dieses Testsets am stärksten“.

| Modell | Korrektheit | Instruktionsfolge | Deutsch | Englisch | Reasoning | Halluzinationskontrolle | Strukturierte Ausgabe | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemma 4 E2B** | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 2 |
| **Qwen3-1.7B-Instruct** | 4 | 4 | 4 | 3 | 2 | 2 | 4 | 4 |
| **Qwen2.5-1.5B-Instruct** | 3 | 2 | 3 | 2 | 1 | 3 | 3 | 5 |
| **Qwen3-VL-2B-Instruct** | 3 | 2 | 3 | 2 | 1 | 1 | 1 | 3 |
| **Qwen2-VL-2B-Instruct** | 2 | 2 | 1 | 1 | 1 | 1 | 3 | 5 |
| **Llama3.2-1B-Instruct** | 2 | 1 | 2 | 2 | 1 | 1 | 3 | 4 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 2 | 1 | 1 | 3 | 1 | 1 | 2 | 2 |

## Antworten und Verhalten im Detail

### Mathematik und kurze Fakten

- **Gemma** löst alle sechs kontrollierten Kurzaufgaben korrekt: `2+2`, Paris, `17×6`, nächster Wochentag, Zahlvergleich und Übersetzung.
- **Qwen3-1.7B** löst fünf von sechs Aufgaben korrekt. Der Ausreißer ist elementar: auf „Welcher Wochentag kommt nach Montag?“ antwortet es mit **Mittwoch**.
- **Qwen2.5** löst `2+2`, Paris, `17×6` und `house → Haus`, scheitert aber an „nach Montag“ (`Dienstags.`) und am Zahlvergleich (`17 23 17`).
- **Qwen3-VL** beantwortet mehrere einfache Aufgaben richtig, macht aber ebenfalls elementare Fehler wie `17×6 → 126` und `größere Zahl → 24`.
- **Qwen2-VL** ist extrem schnell, aber nicht robust: `17×6 → 176`, nächster Wochentag → `2`, `house → Erlaßt`.
- **Llama3.2** bleibt auch nach Entfernung des sichtbaren Stop-Tokens bei mehreren sehr einfachen Aufgaben falsch.
- **DeepSeek-R1-Distill** kann triviale Mathematik grundsätzlich lösen, produziert dafür aber unverhältnismäßig lange Denksequenzen.

Für einen allgemeinen Dienst bedeutet das: **einfache Mathematik oder Kalenderlogik sollte nicht unnötig an ein LLM delegiert werden**, wenn ein deterministischer Rechner in Mikro- oder Millisekunden exakter arbeitet.

### Faktenwissen: Beispiel Bolivien

Die Frage nach der „Hauptstadt von Bolivien“ eignet sich nur bedingt für einen binären Benchmark: **Sucre** ist die verfassungsmäßige Hauptstadt, **La Paz** ist Regierungssitz. Beide Antworten zeigen relevantes Weltwissen. **Buenos Aires**, wie in einem DeepSeek-Lauf ausgegeben, ist dagegen eindeutig falsch.

Für Wissenssysteme gilt deshalb: mehrdeutige oder wichtige Fakten sollten präzisiert oder durch Retrieval/RAG abgesichert werden.

### Umgang mit fehlenden aktuellen Daten

Die Aufforderung „Schreibe genau drei Sätze über das Wetter heute“ wurde ohne Wetterdaten oder Wetter-Tool gestellt und ist deshalb ein guter Halluzinationstest.

- **Gemma** reagiert am sichersten und erklärt, dass keine aktuellen Wetterdaten verfügbar sind.
- **Qwen2.5** erkennt ebenfalls grundsätzlich das Informationsdefizit, formuliert aber sprachlich schwächer.
- **Qwen3-1.7B** erfindet Sonne, `25–28 °C` und Wind.
- **Qwen3-VL** erfindet bewölktes Wetter, Schneeflocken und `5–10 °C`.
- **Llama3.2** erfindet Wetter in **Wien** und übernimmt zusätzlich ein altes Datum aus dem Promptkontext als wäre es aktuell.
- **Qwen2-VL** zerfällt bei dieser längeren deutschen Ausgabe in deutsche, englische und chinesische Fragmente.

**Empfehlung:** aktuelle Daten immer über eine echte Datenquelle, API, RAG oder Tool-Funktion einspeisen. Kein getestetes kleines Modell sollte als Quelle für „heute“, Preise, Wetter oder andere dynamische Fakten betrachtet werden.

### Kreative Texte und längere Antworten

- **Gemma** erzeugt die kohärenteste und sprachlich stabilste Katzengeschichte.
- **Qwen3-1.7B** bleibt verständlich, die Geschichte ist aber sprachlich einfach und grammatikalisch nicht immer sauber.
- **Qwen2.5** verweigert die harmlose Geschichte mit einer unsinnigen Begründung, dass eine Katze keine Person sei.
- **Qwen3-VL** kann in Wiederholungsschleifen geraten; ein Lauf wiederholte „auf dem Kätzchen ...“ bis zum 256-Token-Limit und dauerte rund **55 Sekunden**.
- **Qwen2-VL** erzeugte beschädigte/multilinguale Tokenfragmente statt einer kohärenten Geschichte.
- **Llama3.2** erzeugt erkennbaren Text, aber mit unnatürlicher und semantisch brüchiger Sprache.
- **DeepSeek-R1-Distill** neigt selbst bei trivialen Aufgaben zu langem Thinking und ist deshalb für kurze interaktive Antworten ineffizient.

Niedrige TTFT allein reicht deshalb nicht: Für Chat, Content-Generierung und parallele Requests ist auch die **gesamte Generationsdauer** entscheidend.

### Strukturierte Ausgabe und JSON

- **Gemma** erzeugt das verlangte JSON korrekt, umgibt es allerdings teilweise mit Markdown-Codefences.
- **Qwen3-1.7B** erzeugt in den freien Tests sauberes kompaktes JSON.
- **Qwen2.5** erzeugt das korrekte Objekt, ergänzt aber teilweise unerwünschte Erklärung.
- **Qwen3-VL** ist deutlich unzuverlässiger: in einem Lauf erzeugt es zwei konkurrierende JSON-Objekte; in einem anderen verändert es den Schlüssel `alter` zu `altern`.
- **Qwen2-VL** erzeugt in diesem einzelnen Test korrektes JSON, obwohl seine allgemeine Sprachqualität schwach ist.
- **Llama3.2** kann ebenfalls korrektes JSON erzeugen, aber die geringe allgemeine Korrektheit bleibt ein Risiko.

**Empfehlung:** strukturierte Ausgaben immer gegen ein Schema validieren.

### Datums- und Kalender-Reasoning

Die Frage „Heute ist 1. März 2028. Welcher Wochentag und welches Datum waren vorgestern?“ trennt oberflächliche Sprachfähigkeit von echtem Reasoning.

- **Gemma** antwortet auf Deutsch falsch mit `30. Februar 2028`, löst die praktisch identische englische Frage dagegen korrekt als **Montag, 28. Februar 2028**.
- **Qwen3-VL** erfindet den `31. Februar 2028`.
- **Qwen3-1.7B**, **Qwen2.5**, **Qwen2-VL**, **Llama3.2** und **DeepSeek** zeigen ebenfalls deutliche Fehler oder inkonsistente Zwischenschritte.

Das ist ein wichtiges Ergebnis für allgemeine Nutzung: **kleine lokale LLMs sind keine zuverlässigen Rechen-, Kalender- oder Regel-Engines**. Eine einfache Systemfunktion ist schneller, exakter und spart Accelerator-Zeit.

## Sprachverständnis und Fremdsprachen

Die Tests enthalten vor allem Deutsch und Englisch. Aussagen zu weiteren Sprachen wären ohne eigene isolierte Tests nicht belastbar.

### Deutsch

- **Gemma 4 E2B**: insgesamt stärkste deutschsprachige Qualität, kohärent und meist natürlich; auch Gemma kann bei Reasoning gravierend falsch liegen.
- **Qwen3-1.7B**: brauchbares Deutsch und der beste native Hailo-Kompromiss. Grammatik und semantische Präzision liegen aber klar unter Gemma.
- **Qwen2.5-1.5B**: verständlich, aber häufiger unnatürliche Formulierungen und Fehlinterpretationen.
- **Llama3.2-1B**: versteht einfache deutsche Fragen teilweise, produziert aber ungewöhnliche Grammatik und falsche Antworten.
- **DeepSeek-R1-Distill**: deutlich schwächeres Deutsch; Antworten driften teilweise in gemischte Sprache und unnötige Gedankenketten.
- **Qwen2-VL**: bei längeren deutschen Antworten treten schwere Token-/Sprachzerfallsartefakte auf.
- **Qwen3-VL**: deutlich stabileres Deutsch als Qwen2-VL, jedoch weiterhin Halluzinationen und Wiederholungen.

### Englisch

- **Gemma** ist im vorliegenden Material auf Englisch besonders stark. Auffällig ist, dass die englische Kalenderfrage korrekt gelöst wird, während die deutsche Variante scheitert.
- **DeepSeek-R1-Distill** wirkt auf Englisch deutlich kohärenter als auf Deutsch, halluziniert aber trotzdem Fakten (`Buenos Aires` als Hauptstadt Boliviens) und kann falsche Zwischenschritte aufbauen.
- **Qwen3-1.7B**, **Qwen2.5** und **Llama3.2** wurden im freien Teil weniger umfassend englisch getestet; daraus lässt sich noch keine belastbare englische Rangfolge ableiten.

### Andere Sprachen

Bei **Qwen2-VL** erscheinen chinesische Zeichen und englische Fragmente innerhalb einer deutschen Antwort. Das ist **kein Beleg für gutes Chinesisch**, sondern hier eher ein Hinweis auf Decoder-/Tokenizer-/Generationsinstabilität. Für echte Mehrsprachigkeitsbewertungen sind separate Benchmarks je Sprache erforderlich.

## Technische Betrachtung auf Raspberry Pi 5 + Hailo-10H

### Qwen3-1.7B als nativer Standard-LLM

Qwen3-1.7B ist der überzeugendste Allrounder der getesteten nativen Hailo-Textmodelle:

- ca. **0,64 s TTFT** im kontrollierten Kurztext-Test,
- ca. **1,18 s** mittlere Gesamtdauer,
- 5/6 korrekte Kurzantworten,
- deutlich bessere allgemeine Qualität als Qwen2.5, Llama3.2 und die VLMs als Textmodelle,
- aber nur **2048 Token Gesamtkontext**.

Das macht es besonders attraktiv für lokale interaktive UIs, Voice-Frontends, kompakte Q&A-Dienste und kurze API-Anfragen.

### Qwen2.5-1.5B als Fast Path

Qwen2.5 ist mit ca. **0,38 s TTFT** schneller als Qwen3-1.7B. Für sehr einfache, kurze und tolerante Workloads kann das attraktiv sein. Der Qualitätsverlust ist aber bereits bei elementaren Aufgaben sichtbar. Auch hier gilt die 2k-Kontextgrenze.

### Gemma als Quality- und Long-Context-Pfad

Gemma zeigt, dass ein CPU-Modell auf dem Raspberry Pi 5 sinnvoll sein kann, wenn Qualität und Kontext wichtiger sind als Latenz:

- kontrollierte Kurztexte: **6/6 korrekt**,
- TTFT ungefähr **2,0 s**,
- **4096 Input-Tokens** im stabilen Setup,
- längere Antworten werden durch ungefähr **5–6 Token/s Decode** schnell teuer,
- eine 68-Token-Geschichte dauerte rund **13,5 s**,
- 71 Output-Tokens beim Zählen von 1 bis 20 benötigten rund **13,6 s**.

Für einen lokalen interaktiven Dienst ist Gemma daher eher ein **Quality-/Context-Backend** als ein universeller Low-Latency-Backend.

### VLMs als Textmodelle

Qwen2-VL kann `2+2` in unter einer halben Sekunde beantworten. Technisch ist das beeindruckend, aber die längeren Antworten zeigen, dass hohe Geschwindigkeit keine gute allgemeine Sprachqualität garantiert.

**Qwen-VL sollte primär nach Bildqualität ausgewählt werden.** Die vorliegenden Daten bewerten nur die Textseite. Ein separater Bildbenchmark mit identischen Frames ist notwendig, bevor zwischen Qwen2-VL und Qwen3-VL für Vision-Anwendungen entschieden wird.

## Empfehlungen nach Einsatzprofil

| Nutzung | Empfehlung | Warum |
|---|---|---|
| **Allgemeiner lokaler Chat** | **Qwen3-1.7B** | bestes Qualitäts-/Latenz-Verhältnis auf Hailo |
| **Maximale lokale Textqualität** | **Gemma 4 E2B** | stärkste Korrektheit, Sprache und 4k Input |
| **Ultra-Low-Latency für einfache Texte** | **Qwen2.5-1.5B** | sehr niedrige TTFT, Qualitätsverlust akzeptieren |
| **Längere Prompts / mehr Chat-Historie** | **Gemma 4 E2B** | 4k statt 2k Input-Spielraum |
| **Kleines RAG** | **Qwen3-1.7B** | schnell, wenn Retrieval stark komprimiert ist |
| **Größeres lokales RAG** | **Gemma 4 E2B** | doppelte Input-Grenze; trotzdem Chunking nötig |
| **Längere Erklärungen / kreative Texte** | **Gemma 4 E2B** | deutlich kohärenter als die kleinen Hailo-Modelle |
| **Einfaches lokales Q&A mit hoher Request-Rate** | **Qwen3-1.7B** | gute Balance aus Durchsatz und Qualität |
| **JSON / maschinenlesbare Ausgabe** | **Gemma oder Qwen3-1.7B + Schema-Validator** | beste beobachtete Strukturtreue |
| **Deutschsprachiger Assistent** | **Gemma**, alternativ **Qwen3-1.7B** | beste beobachtete Sprachqualität |
| **Englischsprachiger Assistent** | **Gemma** | stärkste freie englische Antworten im Test |
| **Reasoning / Kalender / Mathematik** | **Tool/Funktion zuerst**, Gemma für Erklärung | kleine LLMs machen selbst bei einfachen Regeln Fehler |
| **Aktuelle Fakten / Wetter / Preise** | **externe Datenquelle/RAG/Tool** | mehrere Modelle halluzinierten nicht vorhandene Daten |
| **Bild-/Multimodal-Anfragen** | **Qwen-VL**, Auswahl nach separatem Bildbenchmark | Texttest sagt wenig über Vision-Qualität aus |
| **Sehr lange Dokumente** | **Chunking/Retrieval vor jedem Modell** | weder 2k noch 4k reichen für lange Dokumente |
| **Deep-Reasoning mit DeepSeek 1.5B** | derzeit nicht empfohlen | Thinking-Overhead hoch, Qualität inkonsistent |

## Empfohlene Service-Architektur für allgemeine Nutzung

```text
                         Client / Anwendung
                                │
                                ▼
                         Hailo-10H-Services
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
      Qwen3-1.7B Hailo      Gemma CPU          Tools / RAG
       schneller Pfad       Quality/4k         exakte Daten
       ~2k Kontext          Kontextpfad              │
              │                 │                    │
              └─────────────────┴────────────────────┘
                                │
                                ▼
                         fertige Antwort

        Bild / Frame ─────────► Qwen-VL auf Hailo
```

Routing sollte dabei nicht nur nach Aufgabentyp, sondern auch nach **Promptgröße** erfolgen. Ein 3.000-Token-RAG-Prompt kann zu Gemma passen, ist für einen 2k-Hailo-LLM aber bereits zu groß.

## Gesamturteil

Der Hailo-10H macht aus dem Raspberry Pi 5 einen überraschend reaktionsschnellen lokalen Inferenzserver. Die wichtigsten Grenzen sind derzeit jedoch nicht nur die Rechenleistung, sondern auch **Modellqualität und Kontextgröße**.

Die nativen Hailo-Modelle liefern sehr niedrige Latenz, sind mit rund 2k Kontext aber auf kompakte Aufgaben angewiesen. Gemma ist deutlich langsamer, liefert dafür bessere Sprache, bessere Korrektheit und mit 4k ungefähr den doppelten Input-Spielraum. Für lange Dokumente oder große Chat-Historien benötigen trotzdem beide Ansätze Retrieval, Zusammenfassung oder Chunking.

Für das getestete System ist derzeit:

- **Qwen3-1.7B-Instruct** der beste allgemeine Hailo-LLM,
- **Gemma 4 E2B** das beste lokale Quality-/4k-Context-Modell,
- **Qwen2.5-1.5B** eine interessante Fast-Path-Option,
- **Qwen-VL** für Vision statt als primäres Textmodell sinnvoll,
- **Llama3.2-1B und DeepSeek-R1-Distill-Qwen-1.5B** in der getesteten Konfiguration nicht erste Wahl.

Der größte praktische Gewinn entsteht durch **Routing nach Qualität, Latenz und Kontextbedarf** statt durch ein einziges universelles Modell.

---

# English

## Purpose of this evaluation

This page evaluates the tested models for **general Hailo-10H-Services use on a Raspberry Pi 5 with Hailo-10H**. It focuses on:

- answer quality and correctness,
- response latency and Time-to-First-Token (TTFT),
- usable context and input limits,
- German and English language quality,
- reasoning and elementary logic,
- hallucinations and handling of missing information,
- structured output such as JSON,
- short and longer responses,
- suitability for local chat, factual questions, RAG, document context, automation and multimodal workloads.

Application-specific Home Assistant intent tests are intentionally **not** evaluated here and do not contribute to rankings or recommendations. Only the general text subset is used for quantitative comparison.

## Untested Function-Calling model

**`Qwen2-1.5B-Instruct-Function-Calling-v1` was not included in this comparison.** There are no measured accuracy, latency or language-quality results for it here, and it does not belong in the rankings of tested models.

Because the language results of **Qwen2.5-1.5B-Instruct** were already unconvincing, the Function-Calling model was not deployed or tested as a universal language model for the service. This was a decision about test scope, **not a demonstrated quality judgment about the untested model**; Qwen2.5's results cannot be presented as measurements of this model.

As a specialized candidate, it could be interesting primarily for **Home Assistant tasks involving function calling**. This suitability is a hypothesis requiring a separate test with real tool schemas and target/value validation. The request, including tool descriptions, catalogue, history and reserved output, must fit the **2048-token total context**. Function calling does not remove this limit.

## Executive summary

The supplied measurements suggest the following for a general local AI service on Raspberry Pi 5 + Hailo-10H:

1. **Gemma 4 E2B** provides the best overall text quality, the highest correctness in the controlled short-text test and the strongest reasoning. The cost is significantly higher CPU latency and slow decoding.
2. **Qwen3-1.7B-Instruct** provides the best quality/speed balance among native Hailo LLMs in this setup. It is currently the most useful default for fast general text requests on Hailo-10H.
3. **Qwen2.5-1.5B-Instruct** is slightly faster, but visibly weaker in language understanding, instruction following and complex tasks. It suits short, simple requests better than a universal assistant.
4. **Llama3.2-1B-Instruct** is fast but clearly weaker in quality. Visible `<|eot_id|>` is not counted as a model error here because it can be removed technically; several answers remain wrong after removal.
5. **DeepSeek-R1-Distill-Qwen-1.5B** is unattractive for interactive low-latency use in this configuration: it thinks for a long time even on trivial questions, produces unnecessary text and has marked weaknesses in German.
6. **Qwen2-VL** is extremely fast on text but has clear language/decoder artifacts on longer responses. It is therefore not recommended as a general text model.
7. **Qwen3-VL** is linguistically more stable than Qwen2-VL, but slower and vulnerable to repetition loops. Its main strength should be evaluated on image/multimodal tasks; these text tests cannot establish its image quality.

### Practical default recommendation

For Raspberry Pi 5 with Hailo-10H, a **service with several specialized paths** is more useful than trying to use one small model for everything:

- **Qwen3-1.7B-Instruct on Hailo** for fast general text requests,
- **Gemma 4 E2B on CPU** for harder language/reasoning tasks or larger context,
- **Qwen-VL on Hailo** for images and video frames,
- **deterministic functions/tools** for mathematics, dates, current data and tasks that can be calculated or queried exactly,
- **RAG/research** for facts that should not rely on model memory.

This uses the strengths of Raspberry Pi 5 and Hailo-10H more effectively: Hailo handles latency-sensitive inference, while CPU and tools are used where they provide a meaningful quality or context advantage.

## Context and input limits: 2k Hailo vs. 4k Gemma

Context size is at least as important for practical use as raw inference speed.

### Native Hailo models: 2048-token context

The currently tested native Hailo LLMs and Qwen VLMs operate in this service with a **2048-token context**. Hailo models share the **same compiled context between input and output**.

In practice:

```text
2048 total context
- reserved output tokens
- template/special-token overhead
= actually usable input
```

With `max_tokens=32`, the effective input limit in one benchmark was approximately **2015 tokens**. Reserving more output reduces the maximum input accordingly. With `max_tokens=256`, roughly **1.8k input tokens** remain.

This 2k limit is sufficient for:

- short chat questions,
- compact system prompts,
- small RAG snippets,
- short summaries,
- classification,
- simple local assistance,
- individual image questions with a VLM, provided the remaining prompt stays small.

It quickly becomes a bottleneck for:

- longer chat history,
- large system prompts,
- multiple document passages,
- extensive RAG context,
- long code snippets,
- large JSON schemas,
- agents with many tool descriptions,
- long answers, which consume the same context.

A Hailo model can therefore respond in well under two seconds without necessarily being able to process a long conversation or document context.

### Gemma 4 E2B: 4096-token input ceiling

Gemma runs through LiteRT-LM on the Raspberry Pi CPU with a deliberately configured **maximum input of 4096 tokens**.

This provides roughly twice the prompt space of the 2k Hailo models. It is particularly useful for:

- longer conversation histories,
- larger RAG chunks or multiple retrieval results,
- more complex instructions,
- longer document excerpts,
- tasks requiring more reasoning context.

The larger limit is not free. More input increases prefill time and memory consumption. Earlier attempts with **8192 input tokens** on this Raspberry Pi setup caused rapidly increasing RAM use and eventual instability/crashes. **4096** is therefore a deliberate practical stability limit, not an arbitrary benchmark setting.

### Consequences for general use

| Workload | 2k Hailo context | 4k Gemma context |
|---|---|---|
| short Q&A | **excellent** | good, but slower |
| short chat history | **good** | **very good** |
| longer chat history | limited | **better** |
| small RAG | **good** | **very good** |
| multiple RAG passages | quickly reaches the limit | **much more flexible** |
| long documents | chunking essential | chunking still needed, but less aggressive |
| large tool/schema prompts | problematic | better, but 4k remains small compared with cloud LLMs |
| long answer + long input | severely restricted | more flexible |

A production local service should make **retrieval and prompt compaction model-dependent**. A request that fits Gemma comfortably at 3,500 tokens cannot be forwarded unchanged to Qwen3-1.7B on Hailo.

### Routing by context length

A useful general strategy is:

```text
short prompt / short history
        |
        v
Qwen3-1.7B on Hailo
        |
        +-- fast, approximately 2k context
        |
        +-- ideal for standard interactive requests

longer prompt / more RAG context
        |
        v
Gemma 4 E2B on CPU
        |
        +-- slower
        +-- up to 4k input in the stable setup
```

For longer content, **both** paths require chunking, retrieval, summarization or hierarchical processing. Even 4k is small for complete documents, large codebases or long conversation archives.

## Test basis and comparability

Two types of evidence were used:

- **Controlled text benchmark**: six short tasks with clear expected answers from the [historical benchmark script](https://github.com/leonsio/hailo-10h-services/blob/4e6a4b4445890471e323cf8667e65d355e683548/scripts/benchmark-qwen-text-intent.py) (since removed), with `temperature=0.1` and `max_tokens=32`. This subset provides the quantitative basis for accuracy, average total time and TTFT.
- **Exploratory WebGUI runs**: arithmetic, capitals, weather, creative text, counting, JSON and date/calendar questions in German and English. These show practical behavior but are not always fully isolated. Chat history and input-token counts sometimes grow; later timings must not be interpreted as pure model-speed comparisons.

### Special treatment of Llama3.2

Llama3.2 sometimes returns visible `<|eot_id|>` in the current service. This is a stop-token/cleanup issue that can be removed deterministically. The token is ignored for **content** evaluation.

The original script score was formally `0/6` because of it. However, `Paris` and `23` are correct in content. `2+2 → 2`, `17×6 → 34`, `after Monday → Monday` and `house → Entferntes Haus` remain real errors. This page therefore scores Llama as **2/6 = 33.3%**.

## Complete question inventory and test language

### Six controlled text tasks

These six German user prompts form the quantitative comparison basis.
The common system instruction was:

> Isolierter Sprachmodell-Benchmark. Folge der Benutzeranweisung exakt. Antworte nur mit der verlangten kurzen Antwort, ohne Erklärung, Markdown oder Präambel.

English translation: “Isolated language-model benchmark. Follow the user's
instruction exactly. Answer only with the requested short answer, without
explanation, Markdown or preamble.”

| ID | Original German question | English translation | Expected answer |
|---|---|---|---|
| C01 | Was ist 2 + 2? Antworte ausschließlich mit der Zahl. | What is 2 + 2? Answer with the number only. | `4` |
| C02 | Was ist die Hauptstadt von Frankreich? Antworte nur mit dem Stadtnamen. | What is the capital of France? Answer with the city name only. | `Paris` |
| C03 | Was ist 17 mal 6? Antworte ausschließlich mit der Zahl. | What is 17 times 6? Answer with the number only. | `102` |
| C04 | Welcher Wochentag kommt nach Montag? Antworte nur mit dem Wochentag. | Which day of the week comes after Monday? Answer with the weekday only. | `Dienstag` |
| C05 | Welche Zahl ist größer: 17 oder 23? Antworte nur mit der Zahl. | Which number is larger: 17 or 23? Answer with the number only. | `23` |
| C06 | Übersetze das englische Wort house ins Deutsche. Antworte nur mit einem Wort. | Translate the English word house into German. Answer with one word only. | `Haus` |

Expected German output remains German even in this translation: C04 requires
`Dienstag`, and C06 requires `Haus`. The translations are for readers, not
substitute test inputs.

### Exploratory WebGUI questions

These original questions supplement the qualitative assessment; they do not
add tasks to the six-question accuracy score. Not every model received every
language variant; repeated questions are not independent measurements.

| ID | Original German question | English translation | Check / reference |
|---|---|---|---|
| E01 | Berechne 2+2 gib nur die Antwort aus | Calculate 2+2; give only the answer. | 4 |
| E02 | Berechne 2+3 gib nur die Antwort aus | Calculate 2+3; give only the answer. | 5 |
| E03 | Was ist die Hauptstadt von Frankreich? | What is the capital of France? | Paris |
| E04 | Was ist die Hauptstadt von Bolivien | What is the capital of Bolivia? | Sucre; La Paz is the seat of government |
| E05 | Schreibe genau drei Sätze über das Wetter heute. | Write exactly three sentences about today's weather. | No live weather data supplied |
| E06 | Erzähle eine sehr kurze, lustige Geschichte über eine Katze namens Miau (maximal 50 Wörter). | Tell a very short, funny story about a cat named Miau (maximum 50 words). | Coherence and word limit |
| E07 | Zähle von 1 bis 20 auf. | Count from 1 to 20. | Complete sequence 1–20 |
| E08 | Gib mir ein JSON-Objekt mit den Schlüsseln 'name' (Wert: 'Max') und 'alter' (Wert: 30). Keine Erklärung drumherum. | Give me a JSON object with the keys 'name' (value: 'Max') and 'alter' (value: 30). No surrounding explanation. | `{"name":"Max","alter":30}` |
| E09 | Heute ist 1. März 2028. Welches Wochentag und Datum waren vorgestern. | Today is March 1, 2028. What day of the week and date was the day before yesterday? | Monday, February 28, 2028 |

E09 deliberately preserves the original German grammar. E01/E02 were sometimes
submitted with an additional final period; E04 also with a question mark.
E09 was also submitted without the final period. These punctuation variants
are not counted as new tasks.

| ID | Additional original prompt | English translation / language note | Use in the supplied log |
|---|---|---|---|
| E10 | Today is March 1, 2028. What day of the week and date was the day before yesterday? | Already English; German counterpart: E09. | English calendar question; sometimes repeated after switching the UI language. |
| E11 | Calculate 2+2 and just give the answer. | Already English; German counterpart: E01. | English arithmetic prompt used with DeepSeek. |
| E12 | What is the capital of France? | Already English; German counterpart: E03. | English factual question used with DeepSeek, repeated. |
| E13 | What is the capital of Bolivia? | Already English; German counterpart: E04. | English factual question used with DeepSeek. |
| E14 | Count from 1 to 20. | Already English; German counterpart: E07. | English counting prompt used with DeepSeek. |
| E15 | Rechte 2+2 | Malformed German input; probably intended as an arithmetic request for 2+2. | Typo in the original Llama and DeepSeek input; do not silently correct it. |
| E16 | Addiere 2+2 .gib nur das Ergebnis aus | Add 2+2. Give only the result. | Additional rewording used with Llama. |
| E17 | Gib mir ein JSON-Objekt mit den Schlüsseln 'name' (Wert: 'Max') und 'alter' (Wert: 30). Keine Erklärung drumherum. Gib nur eine Antwort aus | Give me a JSON object with the keys 'name' (value: 'Max') and 'alter' (value: 30). No surrounding explanation. Give only one answer. | Additional JSON follow-up used with Qwen3-VL. |

E10 is an English variant that was actually tested. English translations
elsewhere in this section are provided for readability and **do not establish**
additional English test runs.

### HA intent questions: documented but excluded from evaluation

The historical script also contained these ten prompts using the text format
`INTENT\|TARGET_TYPE\|TARGET\|VALUE`. No OpenAI tools, HA static-context
envelope or device execution were used. They are listed for completeness but
**do not** contribute to the general quality/speed rankings and do not measure
the suitability of the untested Function-Calling model.

| ID | Original German question | English translation |
|---|---|---|
| H01 | Schalte das Licht im Wohnzimmer auf 70 Prozent. | Set the light in the living room to 70 percent. |
| H02 | Schalte das Licht in der Küche an. | Turn on the light in the kitchen. |
| H03 | Schalte die Nachttischlampe aus. | Turn off the bedside lamp. |
| H04 | Stelle das Licht im Wohnzimmer auf rot. | Set the light in the living room to red. |
| H05 | Stelle die Temperatur im Schlafzimmer auf 19,5 Grad. | Set the temperature in the bedroom to 19.5 degrees. |
| H06 | Fahre den Rollladen im Dachgeschoss auf 40 Prozent. | Set the roller shutter in the attic to 40 percent. |
| H07 | Schließe den Rollladen. | Close the roller shutter. |
| H08 | Starte Deebot mini. | Start Deebot mini. |
| H09 | Schalte die Kaffeemaschine an. | Turn on the coffee machine. |
| H10 | Wie warm ist es im Wohnzimmer? | What is the temperature in the living room? |

### Effect of asking in German

The predominantly German prompts probably placed greater demands on **input
understanding and instruction following** for models stronger in English.
Without an isolated, equivalently designed DE/EN comparison, the size of this
effect cannot be quantified.

Language changes neither the underlying mathematical/logical task nor the
model's fundamental reasoning capability. It can affect how reliably the
model understands and solves the task. The **observed success rate is
therefore not a language-independent measure of pure reasoning ability**.
Gemma's different answers to the German and English calendar questions
illustrate this limitation; exploratory runs can also differ in conversation
history and UI/prompt context.


## Controlled general text benchmark

| Model | Backend | Correct text answers | Avg total | Avg TTFT | Context/input | Character |
|---|---|---:|---:|---:|---|---|
| **Qwen2-VL-2B-Instruct** | Hailo VLM | 3/6 = 50.0% | **637 ms** | **328 ms** | 2048 total | extremely fast, weak text robustness |
| **Qwen2.5-1.5B-Instruct** | Hailo LLM | 4/6 = 66.7% | 934 ms | 381 ms | 2048 total | fast, limited language/logic quality |
| **Llama3.2-1B-Instruct** | Hailo LLM | 2/6 = 33.3%* | 959 ms | 686 ms | 2048 total | fast, many content errors |
| **Qwen3-1.7B-Instruct** | Hailo LLM | **5/6 = 83.3%** | 1,178 ms | 639 ms | 2048 total | best Hailo text compromise |
| **Qwen3-VL-2B-Instruct** | Hailo VLM | 4/6 = 66.7% | 1,354 ms | 642 ms | 2048 total | better language than Qwen2-VL, slower |
| **Gemma 4 E2B** | CPU / LiteRT-LM | **6/6 = 100%** | 2,369 ms | 2,005 ms | **4096 input** | best quality, larger context, higher CPU latency |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Hailo LLM | n/a** | 5,751 ms** | n/a | 2048 total | heavy thinking overhead |

\* Content evaluation after removal of the technically removable `<|eot_id|>`.  
\** The supplied DeepSeek journal included the six isolated text requests and their timings, but not all benchmark response bodies in a form permitting a fair reconstructed accuracy score.

## Ranking by general text quality

| Rank | Model | Controlled text accuracy | Assessment |
|---:|---|---:|---|
| 1 | **Gemma 4 E2B** | **100%** | highest reliability in the test |
| 2 | **Qwen3-1.7B-Instruct** | **83.3%** | best native Hailo choice |
| 3 | Qwen2.5-1.5B-Instruct | 66.7% | fast, but clearly weaker |
| 3 | Qwen3-VL-2B-Instruct | 66.7% | text is a secondary VLM function |
| 5 | Qwen2-VL-2B-Instruct | 50.0% | too unstable for general text |
| 6 | Llama3.2-1B-Instruct | 33.3%* | lowest reliably evaluated text quality |
| — | DeepSeek-R1-Distill-Qwen-1.5B | n/a | see qualitative evaluation below |

## Ranking by response speed

| Rank | Model | Avg total | Avg TTFT | Assessment |
|---:|---|---:|---:|---|
| 1 | **Qwen2-VL-2B-Instruct** | **637 ms** | **328 ms** | fastest output, but quality problems |
| 2 | **Qwen2.5-1.5B-Instruct** | 934 ms | 381 ms | very good low-latency path |
| 3 | Llama3.2-1B-Instruct | 959 ms | 686 ms | fast, but low correctness |
| 4 | **Qwen3-1.7B-Instruct** | 1,178 ms | 639 ms | very good quality/speed compromise |
| 5 | Qwen3-VL-2B-Instruct | 1,354 ms | 642 ms | acceptable for multimodal use |
| 6 | Gemma 4 E2B | 2,369 ms | 2,005 ms | noticeably slower on CPU |
| 7 | DeepSeek-R1-Distill-Qwen-1.5B | 5,751 ms | n/a | thinking overhead dominates |

## Qualitative scorecard

The following 1–5 scores are **not standardized vendor benchmarks**; they summarize the supplied answers. `5` means strongest within this test set.

| Model | Correctness | Instruction following | German | English | Reasoning | Hallucination control | Structured output | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemma 4 E2B** | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 2 |
| **Qwen3-1.7B-Instruct** | 4 | 4 | 4 | 3 | 2 | 2 | 4 | 4 |
| **Qwen2.5-1.5B-Instruct** | 3 | 2 | 3 | 2 | 1 | 3 | 3 | 5 |
| **Qwen3-VL-2B-Instruct** | 3 | 2 | 3 | 2 | 1 | 1 | 1 | 3 |
| **Qwen2-VL-2B-Instruct** | 2 | 2 | 1 | 1 | 1 | 1 | 3 | 5 |
| **Llama3.2-1B-Instruct** | 2 | 1 | 2 | 2 | 1 | 1 | 3 | 4 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 2 | 1 | 1 | 3 | 1 | 1 | 2 | 2 |

## Answers and behavior in detail

### Mathematics and short facts

- **Gemma** solves all six controlled short tasks correctly: `2+2`, Paris, `17×6`, next weekday, number comparison and translation.
- **Qwen3-1.7B** solves five of six correctly. The exception is elementary: for the German question about the weekday after Monday, it answers **Mittwoch** (Wednesday).
- **Qwen2.5** solves `2+2`, Paris, `17×6` and `house → Haus`, but fails after Monday (`Dienstags.`) and the number comparison (`17 23 17`).
- **Qwen3-VL** answers several simple tasks correctly but also makes elementary errors such as `17×6 → 126` and `larger number → 24`.
- **Qwen2-VL** is extremely fast but unreliable: `17×6 → 176`, next weekday → `2`, `house → Erlaßt`.
- **Llama3.2** remains wrong on several very simple tasks after removal of the visible stop token.
- **DeepSeek-R1-Distill** can solve trivial arithmetic but produces disproportionately long thinking sequences.

For a general service, **simple mathematics or calendar logic should not unnecessarily be delegated to an LLM** when a deterministic calculator can be more exact in microseconds or milliseconds.

### Factual knowledge: Bolivia example

The question about Bolivia's capital is only partly suitable for a binary benchmark: **Sucre** is the constitutional capital; **La Paz** is the seat of government. Both answers show relevant world knowledge. **Buenos Aires**, returned in a DeepSeek run, is clearly wrong.

Knowledge systems should clarify ambiguous or important facts or support them with retrieval/RAG.

### Handling missing current data

The request for exactly three sentences about today's weather was made without weather data or a weather tool, making it a useful hallucination test.

- **Gemma** behaves most safely, explaining that current weather data is unavailable.
- **Qwen2.5** also recognizes the missing information, but uses weaker language.
- **Qwen3-1.7B** invents sunshine, `25–28 °C` and wind.
- **Qwen3-VL** invents cloudy weather, snowflakes and `5–10 °C`.
- **Llama3.2** invents weather in **Vienna** and treats an old date from the prompt context as current.
- **Qwen2-VL** degrades into German, English and Chinese fragments on this longer German output.

**Recommendation:** supply current data through a real data source, API, RAG or tool. None of the tested small models should be treated as a source for today, prices, weather or other dynamic facts.

### Creative text and longer answers

- **Gemma** produces the most coherent and linguistically stable cat story.
- **Qwen3-1.7B** remains understandable, but the story uses simple language and imperfect grammar.
- **Qwen2.5** refuses the harmless story with the nonsensical explanation that a cat is not a person.
- **Qwen3-VL** can enter repetition loops; one run repeated `auf dem Kätzchen ...` until the 256-token limit and took approximately **55 seconds**.
- **Qwen2-VL** produces corrupted/multilingual token fragments instead of a coherent story.
- **Llama3.2** produces recognizable text but unnatural and semantically fragile language.
- **DeepSeek-R1-Distill** tends to think at length even for trivial tasks, making it inefficient for short interactive answers.

Low TTFT alone is insufficient: **total generation time** also matters for chat, content generation and parallel requests.

### Structured output and JSON

- **Gemma** produces the requested JSON correctly, but sometimes surrounds it with Markdown fences.
- **Qwen3-1.7B** produces clean compact JSON in the exploratory tests.
- **Qwen2.5** produces the correct object but sometimes adds unwanted explanation.
- **Qwen3-VL** is much less reliable: one run produces two competing JSON objects; another changes the key `alter` to `altern`.
- **Qwen2-VL** produces correct JSON in this single test despite weak general language quality.
- **Llama3.2** can also produce correct JSON, but low general correctness remains a risk.

**Recommendation:** always validate structured output against a schema.

### Date and calendar reasoning

The question about the weekday and date two days before March 1, 2028 separates surface language ability from reasoning. The German original and its English translation are listed above.

- **Gemma** incorrectly answers `30. Februar 2028` in German but solves the practically identical English question correctly as **Monday, February 28, 2028**.
- **Qwen3-VL** invents `31. Februar 2028`.
- **Qwen3-1.7B**, **Qwen2.5**, **Qwen2-VL**, **Llama3.2** and **DeepSeek** also show clear errors or inconsistent intermediate steps.

For general use, **small local LLMs are not reliable calculation, calendar or rule engines**. A simple system function is faster, more exact and saves accelerator time.

## Language understanding and foreign languages

The tests mainly include German and English. Claims about other languages would require separate isolated tests.

### German

- **Gemma 4 E2B**: strongest overall German quality, coherent and usually natural; it can still make serious reasoning errors.
- **Qwen3-1.7B**: usable German and the best native Hailo compromise, but grammar and semantic precision are clearly below Gemma.
- **Qwen2.5-1.5B**: understandable, but more unnatural phrasing and misinterpretation.
- **Llama3.2-1B**: partly understands simple German questions but produces unusual grammar and incorrect answers.
- **DeepSeek-R1-Distill**: noticeably weaker German; responses sometimes drift into mixed languages and unnecessary chains of thought.
- **Qwen2-VL**: severe token/language degradation on longer German answers.
- **Qwen3-VL**: more stable German than Qwen2-VL, but still hallucinations and repetition.

### English

- **Gemma** is particularly strong in the supplied English material. Its English calendar answer is correct while the German variant fails.
- **DeepSeek-R1-Distill** seems significantly more coherent in English than German, but still hallucinates facts (`Buenos Aires` as Bolivia's capital) and can build incorrect intermediate steps.
- **Qwen3-1.7B**, **Qwen2.5** and **Llama3.2** received less comprehensive English exploratory testing; this does not support a reliable English ranking.

### Other languages

**Qwen2-VL** emits Chinese characters and English fragments inside German answers. This is **not evidence of good Chinese**; here it instead suggests decoder/tokenizer/generation instability. Proper multilingual evaluation requires separate benchmarks for each language.

## Technical assessment on Raspberry Pi 5 + Hailo-10H

### Qwen3-1.7B as the native default LLM

Qwen3-1.7B is the strongest all-rounder among the tested native Hailo text models:

- approximately **0.64 s TTFT** in the controlled short-text test,
- approximately **1.18 s** average total time,
- 5/6 correct short answers,
- clearly better general quality than Qwen2.5, Llama3.2 and the VLMs as text models,
- but only **2048 tokens of total context**.

This makes it attractive for local interactive UIs, voice frontends, compact Q&A services and short API requests.

### Qwen2.5-1.5B as a fast path

At approximately **0.38 s TTFT**, Qwen2.5 is faster than Qwen3-1.7B. This can be attractive for very simple, short and tolerant workloads. However, quality loss is visible even on elementary tasks. The 2k context limit also applies.

### Gemma as the quality and longer-context path

Gemma demonstrates that a CPU model on Raspberry Pi 5 can be useful when quality and context matter more than latency:

- controlled short text: **6/6 correct**,
- approximately **2.0 s TTFT**,
- **4096 input tokens** in the stable setup,
- longer answers quickly become expensive at approximately **5–6 decoded tokens/s**,
- a 68-token story took approximately **13.5 s**,
- 71 output tokens when counting from 1 to 20 took approximately **13.6 s**.

For an interactive local service, Gemma is a **quality/context backend** rather than a universal low-latency backend.

### VLMs as text models

Qwen2-VL can answer `2+2` in under half a second. This is technically impressive, but longer responses show that high speed does not guarantee good general language quality.

**Qwen-VL should primarily be selected by image quality.** These results evaluate only text. A separate image benchmark using identical frames is needed before choosing between Qwen2-VL and Qwen3-VL for vision applications.

## Recommendations by workload

| Workload | Recommendation | Reason |
|---|---|---|
| **general local chat** | **Qwen3-1.7B** | best Hailo quality/latency balance |
| **maximum local text quality** | **Gemma 4 E2B** | strongest correctness, language and 4k input |
| **ultra-low-latency simple text** | **Qwen2.5-1.5B** | very low TTFT; accept the quality loss |
| **longer prompts / more chat history** | **Gemma 4 E2B** | 4k instead of 2k input space |
| **small RAG** | **Qwen3-1.7B** | fast with strongly compressed retrieval |
| **larger local RAG** | **Gemma 4 E2B** | twice the input ceiling; chunking still needed |
| **longer explanations / creative text** | **Gemma 4 E2B** | much more coherent than small Hailo models |
| **simple local Q&A at high request rates** | **Qwen3-1.7B** | good throughput/quality balance |
| **JSON / machine-readable output** | **Gemma or Qwen3-1.7B + schema validator** | strongest observed structural fidelity |
| **German-language assistant** | **Gemma**, alternatively **Qwen3-1.7B** | best observed language quality |
| **English-language assistant** | **Gemma** | strongest exploratory English answers in the test |
| **reasoning / calendar / mathematics** | **tool/function first**, Gemma for explanation | small LLMs fail even on simple rules |
| **current facts / weather / prices** | **external source/RAG/tool** | several models hallucinated missing data |
| **image / multimodal requests** | **Qwen-VL**, select using a separate image benchmark | text tests say little about vision quality |
| **very long documents** | **chunking/retrieval before every model** | neither 2k nor 4k is enough for long documents |
| **deep reasoning with DeepSeek 1.5B** | currently not recommended | high thinking overhead, inconsistent quality |

## Recommended service architecture for general use

```text
                         Client / application
                                |
                                v
                         Hailo-10H-Services
                                |
              +-----------------+------------------+
              |                 |                  |
              v                 v                  v
      Qwen3-1.7B Hailo      Gemma CPU          Tools / RAG
          fast path        quality/4k           exact data
        ~2k context       context path              |
              |                 |                    |
              +-----------------+--------------------+
                                |
                                v
                           final answer

        image / frame -------> Qwen-VL on Hailo
```

Routing should consider **prompt size** as well as task type. A 3,000-token RAG prompt can fit Gemma but already exceeds a 2k Hailo LLM's budget.

## Overall conclusion

Hailo-10H turns Raspberry Pi 5 into a surprisingly responsive local inference server. The main current limits are not only compute performance, but also **model quality and context size**.

Native Hailo models provide very low latency but need compact tasks with roughly 2k context. Gemma is much slower, but provides better language, better correctness and approximately twice the input space at 4k. Both paths still need retrieval, summarization or chunking for long documents and extensive chat history.

For the tested system:

- **Qwen3-1.7B-Instruct** is the best general Hailo LLM,
- **Gemma 4 E2B** is the best local quality/4k-context model,
- **Qwen2.5-1.5B** is an interesting fast-path option,
- **Qwen-VL** is useful for vision rather than primary text inference,
- **Llama3.2-1B and DeepSeek-R1-Distill-Qwen-1.5B** are not first choices in the tested configuration.

The largest practical gain comes from **routing by quality, latency and context requirement** rather than forcing one universal model to handle everything.
