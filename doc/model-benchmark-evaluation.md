# Modellvergleich für Raspberry Pi 5 + Hailo-10H / Model comparison for Raspberry Pi 5 + Hailo-10H

> Stand / Snapshot: 6. Oktober 2026. Die Ergebnisse stammen aus Messungen mit Hailo-10H-Services auf demselben Raspberry-Pi-5/CM5-System mit Hailo-10H. Sie sind eine praxisnahe Momentaufnahme dieses Setups und **kein allgemeiner Hersteller-Benchmark**.

[Deutsch](#deutsch) · [English](#english)

GitHub-Markdown unterstützt keine portable clientseitige Tabellensortierung. Deshalb enthält diese Seite bereits vorsortierte Tabellen und kompakte Empfehlungen nach Einsatzprofil.

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

- **Kontrollierter Textbenchmark**: sechs kurze, eindeutig bewertbare Aufgaben aus `scripts/benchmark-qwen-text-intent.py`, mit `temperature=0.1` und `max_tokens=32`. Dieser Teil ist die quantitative Basis für Accuracy, mittlere Laufzeit und TTFT.
- **Explorative WebGUI-Läufe**: Rechnen, Hauptstädte, Wetter, kreative Texte, Zählen, JSON sowie Datums-/Kalenderfragen in Deutsch und Englisch. Diese Läufe zeigen reales Verhalten, sind aber nicht immer vollständig isoliert. Teilweise wächst die Chat-Historie und damit die Zahl der Input-Tokens; spätere Laufzeiten dürfen deshalb nicht als reiner Modell-Speed-Benchmark interpretiert werden.

### Sonderbehandlung Llama3.2

Llama3.2 gibt im aktuellen Service teilweise `<|eot_id|>` sichtbar zurück. Das ist ein Stop-Token-/Cleanup-Thema und kann deterministisch entfernt werden. Für die **inhaltliche** Bewertung wird dieses Token ignoriert.

Der ursprüngliche Skript-Score war dadurch formal `0/6`. Inhaltlich sind jedoch `Paris` und `23` korrekt. `2+2 → 2`, `17×6 → 34`, `nach Montag → Montag` und `house → Entferntes Haus` bleiben echte Fehler. Für diese Seite wird Llama daher mit **2/6 = 33,3 %** bewertet.

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

## Purpose

This page evaluates the tested models for **general Hailo-10H-Services use on a Raspberry Pi 5 with a Hailo-10H accelerator**. It focuses on correctness, latency, context limits, German/English language quality, reasoning, hallucinations, structured output, longer generations, RAG and multimodal use.

Application-specific Home Assistant intent tests are intentionally excluded from all rankings and recommendations. Only the general text subset is used quantitatively.

## Executive summary

1. **Gemma 4 E2B** provides the strongest overall text quality and reasoning, but is substantially slower on the Raspberry Pi CPU.
2. **Qwen3-1.7B-Instruct** provides the best quality/speed balance among the native Hailo LLMs and is the strongest default for fast general text inference.
3. **Qwen2.5-1.5B-Instruct** is faster but clearly weaker on language understanding and more complex prompts.
4. **Llama3.2-1B-Instruct** is fast but weak in this setup. Visible `<|eot_id|>` tokens are treated as removable service artifacts; several answers remain genuinely wrong after cleanup.
5. **DeepSeek-R1-Distill-Qwen-1.5B** has too much thinking overhead and inconsistent quality for this low-latency use case.
6. **Qwen2-VL** is extremely fast for text but unstable on longer generations.
7. **Qwen3-VL** is more linguistically stable but slower and vulnerable to repetition loops; its real value should be judged in a separate vision benchmark.

## Context and input limits: 2k Hailo vs. 4k Gemma

### Hailo models

The tested native Hailo LLMs and Qwen VLMs use a **2048-token compiled context**. Input and output share this budget.

With `max_tokens=32`, the effective accepted input in the benchmark was roughly **2015 tokens** after output reservation and template overhead. Reserving more output directly reduces the available input budget.

This is well suited to short prompts, compact chat, small RAG snippets and interactive Q&A. It becomes restrictive for long chat history, large schemas, multiple retrieved passages, code and document-heavy prompts.

### Gemma

Gemma runs through LiteRT-LM on the Raspberry Pi CPU with a deliberately configured **4096-token input ceiling** in this setup.

That provides roughly twice the prompt space of the Hailo path and is useful for longer chat history, larger RAG contexts and more complex instructions. It is not free: larger input increases prefill latency and memory consumption.

Tests with **8192 input tokens** caused substantially higher RAM use and eventual instability/crashes on this Raspberry Pi setup, which is why **4096** is the practical stable limit used here.

### Practical impact

| Workload | 2k Hailo | 4k Gemma |
|---|---|---|
| short Q&A | **excellent** | good but slower |
| short chat history | **good** | **very good** |
| long chat history | limited | **better** |
| small RAG | **good** | **very good** |
| multiple RAG passages | quickly constrained | **more flexible** |
| long documents | chunking required | chunking still required |
| large schemas/tool prompts | difficult | better, still limited vs cloud models |
| long input + long output | strongly constrained | more flexible |

Routing should therefore consider **prompt size as well as task type**. A 3,000-token request can fit the Gemma path but cannot be sent unchanged to a 2k Hailo model.

## Controlled general text benchmark

| Model | Backend | Correct | Avg total | Avg TTFT | Context/input | Character |
|---|---|---:|---:|---:|---|---|
| **Qwen2-VL-2B-Instruct** | Hailo VLM | 3/6 = 50.0% | **637 ms** | **328 ms** | 2048 total | fastest, weak text robustness |
| **Qwen2.5-1.5B-Instruct** | Hailo LLM | 4/6 = 66.7% | 934 ms | 381 ms | 2048 total | low latency, lower quality |
| **Llama3.2-1B-Instruct** | Hailo LLM | 2/6 = 33.3%* | 959 ms | 686 ms | 2048 total | fast but inaccurate |
| **Qwen3-1.7B-Instruct** | Hailo LLM | **5/6 = 83.3%** | 1,178 ms | 639 ms | 2048 total | best Hailo text balance |
| **Qwen3-VL-2B-Instruct** | Hailo VLM | 4/6 = 66.7% | 1,354 ms | 642 ms | 2048 total | better language, multimodal role |
| **Gemma 4 E2B** | CPU / LiteRT-LM | **6/6 = 100%** | 2,369 ms | 2,005 ms | **4096 input** | best quality and larger context |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Hailo LLM | n/a** | 5,751 ms** | n/a | 2048 total | heavy thinking overhead |

\* Content score after ignoring removable `<|eot_id|>`.  
\** Timings are available, but the supplied DeepSeek logs did not expose all response bodies in a form suitable for a fair reconstructed accuracy score.

## Quality and speed ranking

| Quality rank | Model | Accuracy |
|---:|---|---:|
| 1 | **Gemma 4 E2B** | **100%** |
| 2 | **Qwen3-1.7B-Instruct** | **83.3%** |
| 3 | Qwen2.5-1.5B-Instruct | 66.7% |
| 3 | Qwen3-VL-2B-Instruct | 66.7% |
| 5 | Qwen2-VL-2B-Instruct | 50.0% |
| 6 | Llama3.2-1B-Instruct | 33.3%* |

| Speed rank | Model | Avg total | Avg TTFT |
|---:|---|---:|---:|
| 1 | **Qwen2-VL-2B-Instruct** | **637 ms** | **328 ms** |
| 2 | **Qwen2.5-1.5B-Instruct** | 934 ms | 381 ms |
| 3 | Llama3.2-1B-Instruct | 959 ms | 686 ms |
| 4 | **Qwen3-1.7B-Instruct** | 1,178 ms | 639 ms |
| 5 | Qwen3-VL-2B-Instruct | 1,354 ms | 642 ms |
| 6 | Gemma 4 E2B | 2,369 ms | 2,005 ms |
| 7 | DeepSeek-R1-Distill-Qwen-1.5B | 5,751 ms | n/a |

## Qualitative observations

### Correctness and reasoning

Gemma is the only model to complete all six controlled short tasks correctly. Qwen3-1.7B misses one simple weekday question but is clearly the strongest Hailo text model. Qwen2.5, the VLMs and Llama all show elementary errors.

Calendar reasoning is weak across the small models. Gemma itself fails the German March 1, 2028 question with an impossible February date, then correctly solves the equivalent English version. Fluent language therefore does not imply reliable symbolic reasoning.

### Hallucinations and live data

When asked about today's weather without any weather source, Gemma correctly states that it lacks live data. Several other models invent temperatures, conditions, wind or even a location.

A generally useful local service should use **tools, APIs or RAG for live and important facts** instead of relying on model memory.

### Longer outputs

Longer generations expose weaknesses that short latency benchmarks hide. Qwen3-VL entered a repetition loop and took roughly **55 seconds** to fill a 256-token output despite a first token after about 0.64 seconds. Qwen2-VL degraded into corrupted/multilingual fragments. Gemma was much more coherent but slower at roughly 5–6 decoded tokens per second.

### Structured output

Gemma and Qwen3-1.7B showed the strongest structured-output behavior in the supplied examples. Qwen3-VL sometimes emitted duplicate objects or mutated keys. Programmatic consumers should always validate JSON or other structured responses against a schema.

## Language behavior

### German

Gemma provides the best overall German quality. Qwen3-1.7B is the strongest native Hailo alternative. Qwen2.5 is understandable but less natural and less precise. Llama3.2 and DeepSeek are substantially weaker. Qwen2-VL has severe longer-output degradation, while Qwen3-VL is more stable but still hallucination-prone.

### English

Gemma is particularly strong in the supplied English examples. DeepSeek is noticeably more coherent in English than German, but still hallucinates facts such as `Buenos Aires` as Bolivia's capital. The other Hailo text models need a larger isolated English benchmark before a strong language-specific ranking can be made.

## Recommendations by workload

| Workload | Recommendation | Reason |
|---|---|---|
| **general local chat** | **Qwen3-1.7B** | best Hailo quality/latency balance |
| **maximum local text quality** | **Gemma 4 E2B** | strongest correctness and language quality |
| **ultra-low-latency simple text** | **Qwen2.5-1.5B** | very low TTFT |
| **larger prompts/chat history** | **Gemma 4 E2B** | 4k instead of 2k input space |
| **small RAG** | **Qwen3-1.7B** | fast if retrieval is compact |
| **larger local RAG** | **Gemma 4 E2B** | twice the input budget |
| **longer explanations/creative text** | **Gemma 4 E2B** | more coherent generations |
| **high-rate short Q&A** | **Qwen3-1.7B** | strong speed/quality compromise |
| **JSON/structured output** | **Gemma or Qwen3-1.7B + validator** | strongest observed structure handling |
| **German assistant** | **Gemma**, then **Qwen3-1.7B** | best observed German quality |
| **English assistant** | **Gemma** | strongest supplied English outputs |
| **math/calendar/rules** | **deterministic tool first** | faster and more reliable than small LLMs |
| **live facts/weather/prices** | **API/RAG/tool** | prevents unsupported hallucinations |
| **vision/multimodal** | **Qwen-VL** | select with a separate image benchmark |
| **very long documents** | **chunking/retrieval for every model** | neither 2k nor 4k is a long-context solution |
| **DeepSeek 1.5B reasoning** | currently not recommended | high thinking overhead, inconsistent quality |

## Recommended general architecture

```text
                         Client / application
                                  │
                                  ▼
                           Hailo-10H-Services
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               ▼                  ▼                  ▼
       Qwen3-1.7B Hailo       Gemma CPU          Tools / RAG
        fast ~2k path        quality / 4k        exact/live data
               │                  │                  │
               └──────────────────┴──────────────────┘
                                  │
                                  ▼
                              response

          image / frame ───────► Qwen-VL on Hailo
```

The largest practical gain comes from routing by **quality, latency and context requirement** rather than forcing one model to handle every request.

## Overall conclusion

Hailo-10H turns the Raspberry Pi 5 into a surprisingly responsive local inference server. The main limitations are not only raw compute, but also **model quality and context size**.

Native Hailo models provide excellent latency but currently operate with roughly 2k context and therefore need compact prompts. Gemma is significantly slower, but provides stronger language quality and a 4k input ceiling. Both still require retrieval, summarization or chunking for long documents and long-lived conversations.

For the tested system, **Qwen3-1.7B-Instruct is the best general Hailo LLM**, while **Gemma 4 E2B is the strongest local quality/context backend**.