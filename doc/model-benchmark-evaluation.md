# Modellvergleich: Text, Reasoning und Home-Assistant-Intent / Model comparison: text, reasoning and Home Assistant intent

> Stand / Snapshot: 6. Oktober 2026. Die Ergebnisse stammen aus Messungen auf demselben Hailo-10H-Services-System. Sie sind eine praxisnahe Momentaufnahme, **kein allgemeiner Hersteller-Benchmark**.

[Deutsch](#deutsch) · [English](#english)

GitHub-Markdown unterstützt keine portable clientseitige Tabellensortierung (JavaScript wird auf GitHub nicht als beliebiges Seitenskript ausgeführt). Deshalb enthält diese Seite mehrere **vorsortierte Ranglisten** und aufklappbare `<details>`-Abschnitte.

---

# Deutsch

## Kurzfazit

Für den hier getesteten Einsatz ergibt sich ein klares Bild:

1. **Gemma 4 E2B** liefert die höchste Text-Korrektheit und das beste Reasoning der getesteten Modelle, ist auf CPU/LiteRT-LM aber deutlich langsamer.
2. **Qwen3-1.7B-Instruct** ist der aktuell beste native Hailo-LLM-Kandidat für schnelle allgemeine Textanfragen: 5/6 im kontrollierten Kurztext-Benchmark bei etwa 1,18 s mittlerer Gesamtdauer.
3. **Qwen2.5-1.5B-Instruct** ist sehr schnell, aber sprachlich und semantisch deutlich schwächer; besonders komplexere Antworten, kreative Aufgaben und Reasoning sind unzuverlässig.
4. **Qwen2-VL** ist extrem schnell, zeigt bei längeren Texten jedoch deutliche Decoder-/Sprachzerfallsartefakte. Als Textmodell sollte es derzeit nicht bevorzugt werden.
5. **Qwen3-VL** formuliert besser als Qwen2-VL, ist aber langsamer und zeigt Wiederholungsschleifen, Halluzinationen und unzuverlässige strukturierte Ausgabe.
6. **Llama3.2-1B-Instruct** ist schnell, aber in diesem Setup qualitativ schwach. Das sichtbare `<|eot_id|>` ist ein abfangbares Ausgabe-Artefakt und wird in der inhaltlichen Bewertung **nicht** als Modellfehler gewertet.
7. **DeepSeek-R1-Distill-Qwen-1.5B** ist für diesen Low-Latency-Assistenten ungünstig: das Template erzwingt `<think>`, die Antworten sind oft unnötig lang, Deutsch ist deutlich schwächer als Englisch und mehrere Fakten-/Reasoning-Aufgaben scheitern.

Für Home Assistant ist die wichtigste Schlussfolgerung unabhängig vom Modell: **eindeutige Gerätebefehle und Statusfragen sollten deterministisch verarbeitet werden.** Ein LLM sollte erst bei Ambiguität, freier Sprache oder echtem Reasoning übernehmen. Rohes LLM-Tool-Calling bzw. rohe Slot-JSON-Ausgabe ist bei keinem der kleinen Modelle zuverlässig genug, um ohne Validierung Aktionen auszuführen.

## Testbasis und Fairness

Es gibt zwei unterschiedliche Datensätze:

- **Kontrollierter Benchmark**: `scripts/benchmark-qwen-text-intent.py`, 16 Szenarien, davon 6 kurze Textaufgaben und 10 HA-Intent-/Slot-Aufgaben, `temperature=0.1`, `max_tokens=32`. Dieser Datensatz ist für quantitative Vergleiche am aussagekräftigsten.
- **Explorative WebGUI-Läufe**: freie Fragen zu Rechnen, Hauptstädten, Wetter, Geschichte, Zählen, JSON und Datums-Reasoning. Diese Läufe zeigen reales Verhalten, sind aber nicht vollständig isoliert: bei mehreren Modellen wächst die Chat-Historie und damit die Zahl der Input-Tokens. Spätere WebGUI-Laufzeiten dürfen daher nicht als reiner Modell-Speed-Benchmark interpretiert werden.

Die strikte HA-Intent-Auswertung verlangt exakt:

```text
INTENT|TARGET_TYPE|TARGET|VALUE
```

Das ist absichtlich streng. Deshalb wird unten zusätzlich zwischen **Schema-/Format-Compliance** und **semantischem Verständnis** unterschieden.

### Sonderbehandlung Llama

Llama3.2 gibt im aktuellen Service teilweise das Special Token `<|eot_id|>` sichtbar zurück. Das ist ein Service-/Stop-Token-Cleanup-Thema und kann deterministisch entfernt werden. Für die qualitative Bewertung wird dieses Token ignoriert. Dadurch steigt die inhaltliche Kurztext-Bewertung des kontrollierten Benchmarks von formal `0/6` auf **2/6 (33,3 %)**: `Paris` und `23` sind inhaltlich korrekt; `2`, `34`, `Montag` und `Entferntes Haus` bleiben inhaltlich bzw. instruktionsbezogen falsch.

## Kontrollierter Benchmark

| Modell | Backend | Text korrekt | Text Ø gesamt | Text Ø TTFT | Intent strikt | Intent Ø gesamt | Intent Ø TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen2-VL-2B-Instruct** | Hailo VLM | 3/6 = 50,0 % | **637 ms** | **328 ms** | 0/10 | 3.009 ms | 949 ms |
| **Qwen2.5-1.5B-Instruct** | Hailo LLM | 4/6 = 66,7 % | 934 ms | 381 ms | 0/10 | 3.523 ms | 1.109 ms |
| **Llama3.2-1B-Instruct** | Hailo LLM | 2/6 = 33,3 %* | 959 ms | 686 ms | 0/10 | 3.987 ms | 1.021 ms |
| **Qwen3-1.7B-Instruct** | Hailo LLM | **5/6 = 83,3 %** | 1.178 ms | 639 ms | 0/10 | 5.267 ms | 1.875 ms |
| **Qwen3-VL-2B-Instruct** | Hailo VLM | 4/6 = 66,7 % | 1.354 ms | 642 ms | 0/10 | 4.596 ms | 1.890 ms |
| **Gemma 4 E2B** | CPU / LiteRT-LM | **6/6 = 100 %** | 2.369 ms | 2.005 ms | **1/10 = 10 %** | 5.691 ms | 3.673 ms |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Hailo LLM | n/a** | 5.751 ms** | n/a | n/a** | 7.234 ms** | n/a |

\* Nach Entfernung von `<|eot_id|>` nur für die Inhaltsbewertung. Der ursprüngliche Skript-Score war 0/6, weil das Special Token Teil des Antwortstrings war.  
\** Im bereitgestellten DeepSeek-Journal sind Request und Laufzeit vorhanden, aber nicht der Benchmark-Response-Body/TTFT in einer Form, aus der sich der Accuracy-Score zuverlässig rekonstruieren lässt. Die Mittelwerte basieren auf den sechs Text- bzw. zehn Intent-HTTP-Requests.

### Rangliste: Korrektheit bei kurzen Textaufgaben

| Rang | Modell | Accuracy |
|---:|---|---:|
| 1 | Gemma 4 E2B | **100 %** |
| 2 | Qwen3-1.7B-Instruct | **83,3 %** |
| 3 | Qwen2.5-1.5B-Instruct | 66,7 % |
| 3 | Qwen3-VL-2B-Instruct | 66,7 % |
| 5 | Qwen2-VL-2B-Instruct | 50,0 % |
| 6 | Llama3.2-1B-Instruct | 33,3 %* |
| — | DeepSeek-R1-Distill-Qwen-1.5B | nicht aus Journal rekonstruierbar |

### Rangliste: Latenz bei kurzen Textaufgaben

| Rang | Modell | Ø gesamt |
|---:|---|---:|
| 1 | Qwen2-VL-2B-Instruct | **637 ms** |
| 2 | Qwen2.5-1.5B-Instruct | 934 ms |
| 3 | Llama3.2-1B-Instruct | 959 ms |
| 4 | Qwen3-1.7B-Instruct | 1.178 ms |
| 5 | Qwen3-VL-2B-Instruct | 1.354 ms |
| 6 | Gemma 4 E2B | 2.369 ms |
| 7 | DeepSeek-R1-Distill-Qwen-1.5B | 5.751 ms |

## Qualitative Scorecard

Die folgenden 1–5-Werte sind **keine standardisierten Benchmarks**, sondern eine zusammenfassende Bewertung genau der hier vorliegenden Antworten. `5` bedeutet „innerhalb dieses Testsets am stärksten“.

| Modell | Korrektheit | Instruktionsfolge | Deutsch | Englisch | Reasoning | Halluzinationskontrolle | Strukturierte Ausgabe | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemma 4 E2B** | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 2 |
| **Qwen3-1.7B-Instruct** | 4 | 4 | 4 | 3 | 2 | 2 | 4 | 4 |
| **Qwen2.5-1.5B-Instruct** | 3 | 2 | 3 | 2 | 1 | 3 | 3 | 5 |
| **Qwen3-VL-2B-Instruct** | 3 | 2 | 3 | 2 | 1 | 1 | 1 | 3 |
| **Qwen2-VL-2B-Instruct** | 2 | 2 | 1 | 1 | 1 | 1 | 3 | 5 |
| **Llama3.2-1B-Instruct** | 2 | 1 | 2 | 2 | 1 | 1 | 3 | 4 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 2 | 1 | 1 | 3 | 1 | 1 | 2 | 2 |

## Antworten im Detail

### Mathematik und einfache Fakten

- **Gemma** löst im kontrollierten Benchmark alle sechs Kurztextaufgaben korrekt, darunter `2+2`, `17×6`, den nächsten Wochentag, Zahlvergleich und Übersetzung.
- **Qwen3-1.7B** ist der stärkste Hailo-LLM: korrekt bei `2+2`, Paris, `17×6`, Zahlvergleich und `house → Haus`; es antwortet jedoch auf „Welcher Wochentag kommt nach Montag?“ mit **Mittwoch**.
- **Qwen2.5** löst `2+2`, Paris, `17×6` und `house → Haus`, scheitert aber an „nach Montag“ (`Dienstags.`) und dem einfachen Zahlvergleich (`17 23 17`).
- **Qwen2-VL** ist sehr schnell, aber selbst einfache Aufgaben sind nicht robust (`17×6 → 176`, Wochentag → `2`, `house → Erlaßt`).
- **Qwen3-VL** ist sprachlich stabiler, macht jedoch ebenfalls elementare Fehler (`17×6 → 126`, größerer Wert → `24`).
- **Llama3.2**: nach Entfernung des sichtbaren `<|eot_id|>` sind `Paris` und `23` korrekt. `2+2 → 2`, `17×6 → 34`, „nach Montag“ → `Montag` bleiben echte Modellfehler.
- **DeepSeek-R1-Distill** kann `2+2` korrekt lösen, generiert dafür aber unnötig lange Gedankenketten. In einer englischen Folge verwechselt es bei der Analyse sogar kurz die zweite Zahl mit `3`, obwohl die Aufgabe `2+2` lautet.

### Hauptstadt von Bolivien: kein guter binärer Test

Die Antworten `Sucre` und `La Paz` sollten nicht pauschal als Halluzination gewertet werden. **Sucre ist die verfassungsmäßige Hauptstadt**, während **La Paz Regierungssitz** ist. Für zukünftige automatische Benchmarks sollte die Frage präzisiert werden, zum Beispiel:

```text
Wie heißt die verfassungsmäßige Hauptstadt Boliviens?
```

oder:

```text
In welcher Stadt sitzt die bolivianische Regierung?
```

**Buenos Aires** (DeepSeek) ist dagegen eindeutig falsch.

### Wetter: Test auf Halluzinationskontrolle

Die Frage „Schreibe genau drei Sätze über das Wetter heute“ enthält im Test **keine Wetterdaten und kein Wetter-Tool**.

- **Gemma** reagiert am sichersten: es erklärt, dass keine aktuellen Wetterdaten verfügbar sind, statt Wetter zu erfinden.
- **Qwen2.5** erkennt ebenfalls grundsätzlich die fehlende Wetterfähigkeit, formuliert aber grammatikalisch schwach und folgt „genau drei Sätze“ nicht sauber.
- **Qwen3-1.7B** erfindet `25–28 °C`, Sonne und Wind. Das ist flüssig formuliert, aber faktisch nicht gestützt.
- **Qwen3-VL** erfindet bewölktes Wetter, Schneeflocken und `5–10 °C`.
- **Llama3.2** erfindet warmes Wetter in **Wien** und übernimmt zusätzlich ein Datum aus dem Modell-/Promptkontext (`26 Jul 2024`) als wäre es aktuell.
- **Qwen2-VL** zerfällt bei dieser längeren deutschen Ausgabe in gemischte deutsche, englische und chinesische Fragmente.
- **DeepSeek** wurde in den vorliegenden freien Antworten für dieses Wetter-Szenario nicht ausreichend isoliert dokumentiert.

Für produktive aktuelle Daten gilt: **Tool/RAG/Datenquelle oder explizites Abstention-Verhalten**, nie reines Modellwissen.

### Kreative Sprache

- **Gemma** erzeugt die kohärenteste Geschichte. Sie bleibt sprachlich stabil und inhaltlich nachvollziehbar.
- **Qwen3-1.7B** erzeugt eine kurze, einfache Geschichte. Grammatik (`Sie fressen`) und Humor sind schwach, aber die Ausgabe bleibt kohärent.
- **Qwen2.5** verweigert die harmlose Aufgabe mit einer sachlich unsinnigen Begründung („Katze ... keine Person“). Das zeigt schwaches Instruktions- und Bedeutungsverständnis.
- **Qwen3-VL** gerät in eine starke Wiederholungsschleife („auf dem Kätzchen ...“) bis zum Output-Limit; ein einzelner Lauf dauerte rund **55 s**.
- **Qwen2-VL** produziert beschädigte/multilinguale Tokenfragmente statt einer Geschichte.
- **Llama3.2** liefert zwar eine erkennbare Katzengeschichte, die Sprache ist jedoch unnatürlich und semantisch brüchig.
- **DeepSeek** zeigt bei einfachen Aufgaben bereits starke Tendenz zum Überdenken; für kurze kreative Antworten ist dieses Verhalten in der vorliegenden Konfiguration ungünstig.

### Zählen 1–20

Dieses Szenario zeigt besonders gut den Unterschied zwischen Sprachfluss und elementarer Verlässlichkeit:

- Gemma, Qwen2-VL, Qwen3-VL, Qwen2.5 und Qwen3-1.7B zählen korrekt.
- Llama3.2 erzeugt eine bizarre Potenzfolge (`0, 100, 1000, 10, ...`) statt 1–20.
- DeepSeek liefert in einem freien englischen Lauf nach `Count from 1 to 20.` nur die Thinking-/End-Markierung und **keine Nutzantwort**.

### JSON / strukturierte Ausgabe

- **Qwen3-1.7B** liefert im freien Test die sauberste Ausgabe: `{"name": "Max", "alter": 30}`.
- **Gemma** liefert inhaltlich korrektes JSON, aber in einem Markdown-Codeblock. Für eine API, die rohes JSON verlangt, ist das eine reparierbare Formatabweichung.
- **Qwen2-VL** liefert ebenfalls korrektes JSON im Codeblock.
- **Qwen2.5** liefert korrektes JSON, ergänzt aber entgegen der Anweisung noch einen erklärenden Hinweis.
- **Llama3.2** liefert nach Entfernen von `<|eot_id|>` ein inhaltlich korrektes JSON-Objekt im Codeblock.
- **Qwen3-VL** ist für exakte Schemas problematisch: einmal erzeugt es **zwei konkurrierende JSON-Objekte**, verändert Groß-/Kleinschreibung (`ALTER`) und in einem zweiten Versuch sogar den Schlüssel zu `altern`.
- Für **DeepSeek** liegt in den bereitgestellten freien Läufen kein direkt vergleichbarer JSON-Test vor.

**Produktionsfolgerung:** Modelloutput sollte nicht direkt als Aktionsobjekt ausgeführt werden. Erst strikt parsen, Schema validieren und nur erwartete Enums/Entities/Werte zulassen.

## Reasoning: 1. März 2028

Die korrekte Antwort auf:

```text
Today is March 1, 2028. What day of the week and date was the day before yesterday?
```

ist:

```text
Monday, February 28, 2028
```

Die Aufgabe ist besonders aussagekräftig, weil Schaltjahr, Monatsgrenze und Wochentag zusammenkommen.

- **Gemma Englisch** ist der einzige vorliegende freie Lauf mit vollständig korrekter Herleitung und Endantwort.
- **Gemma Deutsch** erfindet dagegen den ungültigen **30. Februar 2028**. Das zeigt eine deutliche sprachabhängige Reasoning-Inkonsistenz.
- **Qwen3-1.7B Deutsch** antwortet `Dienstag, 30. Februar 2028`. Nach frischem englischem Chat wird die Antwort ausführlicher, bleibt aber einen Tag daneben (`Sunday, February 29, 2028`) und setzt zudem den Wochentag von March 1 falsch an.
- **Qwen3-VL Deutsch** erfindet den **31. Februar 2028**.
- **Qwen2-VL Deutsch** zerfällt zu `VortrefflicheUCCESS`; Englisch liefert ein falsches Jahr/Datum.
- **Qwen2.5** scheitert in Deutsch und Englisch und erzeugt falsche Monatsübergänge.
- **Llama3.2** springt auf 2024 und erzeugt ungültige Datumsfolgen.
- **DeepSeek** setzt `yesterday` fälschlich auf 27. Februar und springt anschließend auf den 15. Februar; Endantwort und Herleitung sind falsch.

Damit ist Datums-/Kalenderlogik ein starkes Argument für einen **deterministischen Date/Time-Resolver** statt LLM-Reasoning.

## Sprachverständnis und Fremdsprachen

### Deutsch

**Gemma** besitzt das beste allgemeine deutsche Sprachverständnis im Test, ist aber beim Kalender-Reasoning auf Deutsch überraschend schwach.  
**Qwen3-1.7B** formuliert deutlich natürlicher als die kleineren Hailo-LLMs und VLMs.  
**Qwen2.5** versteht einfache deutsche Fakten und Befehle, zeigt aber grammatische Fehler, unnötige Ablehnungen und semantische Aussetzer.  
**Llama3.2** kann einfache deutsche Fakten beantworten, ignoriert aber häufig die gewünschte Form und halluziniert Kontext.  
**DeepSeek-R1-Distill** zeigt die stärkste Sprachasymmetrie: die deutsche Frankreich-Frage degeneriert in weitgehend sinnlosen Text, während dieselbe Frage auf Englisch korrekt beantwortet wird.  
**Qwen2-VL** zeigt bei längerer deutscher Generation deutliche multilinguale Decoder-Artefakte.

### Englisch

- **Gemma** profitiert bei der schwierigsten Reasoning-Aufgabe deutlich von Englisch und liefert als einziges Modell die korrekte Kalenderantwort.
- **DeepSeek** ist auf Englisch wesentlich verständlicher als auf Deutsch, bleibt aber faktisch unzuverlässig (`Bolivia → Buenos Aires`) und überdenkt einfache Aufgaben.
- **Qwen3-1.7B** formuliert nach einem frischen englischen Chat kohärent, löst die Kalenderaufgabe jedoch weiterhin falsch.
- **Llama3.2** bleibt auch auf Englisch schwach; ein Sprachwechsel behebt die Datumsfehler nicht.
- **Qwen2.5** produziert auf Englisch lange, aber falsche Datumslogik.
- **Qwen2-VL** bleibt auch auf Englisch bei komplexerem Reasoning instabil.

### Andere Sprachen / multilinguale Artefakte

Es gab keinen kontrollierten Spanisch-/Französisch-/Russisch-Benchmark. Deshalb sollte aus diesen Daten **keine allgemeine Multilingual-Rangliste** abgeleitet werden. Sichtbare chinesische oder anderssprachige Fragmente bei Qwen2-VL sind in diesem Test kein Beleg für gutes Fremdsprachenverständnis, sondern eher ein Decoder-/Generationsartefakt.

## Home-Assistant-Intent: striktes Format vs. echtes Verständnis

Der rohe Accuracy-Wert ist hart: Gemma 1/10, alle vollständig ausgewerteten Hailo-Modelle 0/10. Trotzdem sind die Fehlerarten sehr unterschiedlich.

### Gemma

Gemma versteht mehrere Befehle semantisch richtig, scheitert aber häufig am exakt verlangten Vier-Felder-Format:

- `LIGHT_BRIGHTNESS|Wohnzimmer|70`: Intent, Ziel und Wert stimmen, `TARGET_TYPE` fehlt.
- `CLIMATE_TEMPERATURE|Schlafzimmer|19,5`: semantisch stark, Schema unvollständig.
- `VACUUM_START|name|Deebot mini|-`: vollständig korrekt.
- `STATE_READ_TEMPERATURE|Wohnzimmer|- -`: Intent/Ziel plausibel, Schema fehlerhaft.
- Cover-/Switch-Beispiele zeigen aber auch echte Intent-Fehler.

Das macht Gemma als **semantischen Fallback mit nachgelagerter deterministischer Validierung** deutlich interessanter als der strikte 10-%-Score allein vermuten lässt.

### Qwen3-1.7B

Qwen3-1.7B erkennt in mehreren Antworten den richtigen Intent-Namen, verschiebt ihn aber in die falsche Spalte oder gibt Literal-Platzhalter aus:

```text
INTENT|LIGHT_BRIGHTNESS|Wohnzimmer|70%
INTENT|LIGHT_ON|Küche|-.
```

Das spricht für **teilweises Aufgabenverständnis**, aber schwaches Schema-Lernen bei diesem Prompt. Für direkte Tool-Ausführung reicht das nicht.

### Qwen2.5

Qwen2.5 kollabiert bei sehr unterschiedlichen Aufgaben wiederholt auf `STATE_READ_TEMPERATURE`, übersetzt Targets (`Wohnzimmer → living_room`, `Küche → kitchen`) und erfindet Werte. Das ist nicht nur Formatversagen, sondern ein klarer semantischer Routingfehler.

### Llama3.2

Llama ignoriert die Klassifikationsform häufig vollständig und antwortet in freier Prosa („Ich kann nicht direkt auf deine Geräte ...“). Das Special Token ist hier nicht das Problem; auch nach dessen Entfernung bleibt die Ausgabe für Intent-/Slot-Extraktion unbrauchbar.

### Qwen2-VL / Qwen3-VL

Beide VLMs erreichen 0/10. Qwen3-VL ist sprachlich sauberer, halluziniert aber Felder, dupliziert Strukturen und verändert Namen. Qwen2-VL produziert zusätzlich beschädigte Token-/HTML-artige Fragmente. Für direkte Geräteaktionen sind beide ungeeignet.

### DeepSeek

Der Journal-Log zeigt für die Intent-Gruppe sehr konstante Laufzeiten um 7,2 s bei `max_tokens=32`. Das Template startet jede Antwort mit `<think>`, wodurch ein großer Teil des knappen Output-Budgets für Reasoning statt für das geforderte kompakte Schema verbraucht wird. Da die eigentlichen Benchmark-Response-Bodies im bereitgestellten Journal nicht vollständig vorliegen, wird kein Accuracy-Wert erfunden.

## Technische Beobachtungen

<details>
<summary><strong>1. TTFT ist nicht gleich Antwortzeit</strong></summary>

Qwen2-VL, Qwen2.5 und Qwen3-1.7B beginnen sehr schnell zu generieren. Bei längeren Antworten dominiert aber der Decode-Anteil. Qwen3-VL startet eine Katzengeschichte nach rund 0,64 s, läuft wegen einer Wiederholungsschleife aber insgesamt etwa 55 s.

Für Assist-Sprachinteraktion sind daher mindestens drei Werte nötig: **TTFT, Decode-Durchsatz und tatsächliche End-to-End-Zeit**.

</details>

<details>
<summary><strong>2. WebGUI-Historie verfälscht spätere Latenzvergleiche</strong></summary>

In mehreren freien Läufen steigen die Input-Tokens mit jeder Frage stark an. Qwen3-1.7B wächst beispielsweise von sehr kleinem Anfangskontext auf mehrere hundert Input-Tokens. Nach „New chat“ fallen sie wieder deutlich.

Deshalb dienen die WebGUI-Läufe primär der **Qualitätsanalyse**. Für Speed-Rankings wird der isolierte Benchmark bevorzugt.

</details>

<details>
<summary><strong>3. Stop-/Special-Token müssen im Service bereinigt werden</strong></summary>

Llama3.2 gibt `<|eot_id|>` sichtbar zurück. DeepSeek zeigt `<｜end▁of▁sentence｜>` und Thinking-Marker. Solche bekannten Modell-Special-Tokens sollten backend-spezifisch entfernt bzw. als Stop-Sequenzen behandelt werden.

Das ändert jedoch nur die Darstellung: falsche Inhalte wie `2+2 → 2` oder `17×6 → 34` bleiben falsch.

</details>

<details>
<summary><strong>4. DeepSeek-Template erzwingt Reasoning</strong></summary>

Der gerenderte Prompt endet mit:

```text
<｜Assistant｜><think>
```

Damit wird selbst für `2+2` ein Reasoning-Modus angestoßen. In einem Assistenten, der oft nur eine kurze Antwort oder einen Intent braucht, ist das kontraproduktiv: hohe Latenz, höheres Output-Budget und schlechtere Instruktionsfolge.

</details>

<details>
<summary><strong>5. Qwen2-VL: möglicher Decoder-/Tokenizer-/Runtime-Effekt</strong></summary>

Die längeren Qwen2-VL-Ausgaben enthalten ungewöhnliche Sprachmischungen, beschädigte Unicode-Fragmente und fremde Tokenstücke. Das kann reine Modellschwäche sein, ist aber auffällig genug, um zusätzlich **HEF-Version, Tokenizer, Prompt-Template und HailoRT-Decoder-Kompatibilität** zu prüfen, bevor daraus eine generelle Aussage über das Basismodell abgeleitet wird.

</details>

<details>
<summary><strong>6. Kleine Modelle sollten keine berechenbaren Aufgaben „reasonen“</strong></summary>

Datum/Wochentag, einfache Mathematik, Vergleiche, Prozentwerte und bekannte HA-Zustände sind deterministisch lösbar. Die Tests zeigen, dass selbst gute Sprachmodelle dort überraschend halluzinieren können.

Ein deterministischer Resolver ist gleichzeitig **schneller, billiger und korrekter**.

</details>

## Empfehlung für Hailo-10H-Services

### Text-Routing

```text
Benutzer / Home Assistant
          |
          v
  deterministischer Router
     / MiniLM retrieval
      /           \
 eindeutig       unklar / frei
    |                |
    v                v
 direkt          Gemma 4 E2B
```

Optional kann **Qwen3-1.7B-Instruct** als schneller allgemeiner Hailo-Textmodus angeboten werden, wenn geringe Latenz wichtiger ist als maximale Zuverlässigkeit. Für sicherheits- oder aktionsrelevante HA-Aufgaben sollte es aber nicht ungeprüft das finale Tool-Objekt erzeugen.

### Empfohlene Rollen

| Rolle | Empfehlung |
|---|---|
| Deterministische HA-Aktionen/Status | Parser + Entity-/Tool-Resolver + Validierung |
| Allgemeine Antwort mit höchster Qualität | **Gemma 4 E2B** |
| Schneller nativer Hailo-Textmodus | **Qwen3-1.7B-Instruct** |
| Sehr einfache/experimentelle Low-Latency-Texte | Qwen2.5-1.5B, nur mit klaren Grenzen |
| Bildanalyse | VLM separat nach **Bildbenchmark** auswählen |
| Kalender/Mathematik | deterministischer Code, nicht LLM |
| Aktuelles Wetter/Live-Daten | Tool/Datenquelle; sonst abstain |
| Rohes Tool-/JSON-Objekt | niemals ungeprüft ausführen |

### Derzeit nicht als Standard-Textmodell empfohlen

- Qwen2-VL: Decoder-/Sprachstabilität
- Qwen3-VL: Wiederholung, Halluzination, strukturelle Instabilität
- Llama3.2-1B: geringe Korrektheit/Instruktionsfolge in diesem Setup
- DeepSeek-R1-Distill-Qwen-1.5B: erzwungenes Reasoning, hohe Latenz, starke Deutsch/Englisch-Asymmetrie

## Nächste sinnvolle Benchmarks

1. Jeden kontrollierten Test mindestens 5–10 Mal ausführen und Median/P95 erfassen.
2. Frische Konversation pro WebGUI-Qualitätstest erzwingen.
3. Deutsch und Englisch als identische Testpaare führen.
4. Ergänzen: Russisch/Französisch/Spanisch, falls diese Sprachen produktiv relevant sind.
5. Aktuelle Daten getrennt testen: „ohne Tool muss abstain“ vs. „mit Tool muss korrekt zitieren“.
6. JSON-Schema-Constrained-Decoding testen, falls HailoRT/Backend das unterstützt.
7. Separaten **VLM-Bildbenchmark** für den eigentlichen Bild-/Frigate-Anwendungsfall durchführen.
8. HA-Benchmark zusätzlich mit semantischem Score auswerten: `intent korrekt`, `target korrekt`, `value korrekt`, `schema korrekt` getrennt statt nur Gesamt-PASS/FAIL.

---

# English

## Executive summary

The results show a clear separation between raw speed and trustworthy behavior:

1. **Gemma 4 E2B** is the most accurate text model in this sample and shows the best overall reasoning, but CPU/LiteRT-LM inference is substantially slower.
2. **Qwen3-1.7B-Instruct** is the strongest native Hailo LLM for fast general text: 5/6 on the controlled short-text benchmark at about 1.18 s average end-to-end latency.
3. **Qwen2.5-1.5B-Instruct** is fast but materially weaker on language quality, instruction following and reasoning.
4. **Qwen2-VL** is extremely fast but becomes unstable on longer text, including mixed-language/token corruption.
5. **Qwen3-VL** is linguistically cleaner than Qwen2-VL but slower and prone to repetition loops, hallucination and unreliable structured output.
6. **Llama3.2-1B-Instruct** is fast but weak in this runtime configuration. The visible `<|eot_id|>` marker is treated as a removable service artifact, not as an answer-quality failure.
7. **DeepSeek-R1-Distill-Qwen-1.5B** is a poor fit for this low-latency assistant configuration: the template forces `<think>`, simple requests become verbose, German performance is much worse than English, and factual/reasoning failures remain common.

For Home Assistant the architecture conclusion is more important than the winner: **deterministic device actions and state queries should stay deterministic**. LLM inference should be reserved for ambiguity, natural-language explanation and real reasoning. No tested small model is reliable enough to execute raw generated tool/slot output without validation.

## Controlled benchmark

| Model | Backend | Text accuracy | Text avg total | Text avg TTFT | Strict intent | Intent avg total | Intent avg TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen2-VL-2B-Instruct** | Hailo VLM | 50.0% | **637 ms** | **328 ms** | 0/10 | 3,009 ms | 949 ms |
| **Qwen2.5-1.5B-Instruct** | Hailo LLM | 66.7% | 934 ms | 381 ms | 0/10 | 3,523 ms | 1,109 ms |
| **Llama3.2-1B-Instruct** | Hailo LLM | 33.3%* | 959 ms | 686 ms | 0/10 | 3,987 ms | 1,021 ms |
| **Qwen3-1.7B-Instruct** | Hailo LLM | **83.3%** | 1,178 ms | 639 ms | 0/10 | 5,267 ms | 1,875 ms |
| **Qwen3-VL-2B-Instruct** | Hailo VLM | 66.7% | 1,354 ms | 642 ms | 0/10 | 4,596 ms | 1,890 ms |
| **Gemma 4 E2B** | CPU / LiteRT-LM | **100%** | 2,369 ms | 2,005 ms | **1/10** | 5,691 ms | 3,673 ms |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Hailo LLM | n/a** | 5,751 ms** | n/a | n/a** | 7,234 ms** | n/a |

\* Content-adjusted after stripping `<|eot_id|>`. The raw script reported 0/6.  
\** The supplied server journal contains request timings but not enough response-body/TTFT data to reconstruct a trustworthy accuracy score.

## Key quality findings

### Correctness

Gemma is the only model with 6/6 on the controlled short-text set. Qwen3-1.7B is the strongest accelerator-native text model at 5/6. Qwen2.5 and Qwen3-VL reach 4/6, Qwen2-VL 3/6, and content-normalized Llama3.2 2/6.

### Hallucination control

The unsupported “weather today” prompt is a useful test. Gemma refuses to invent live weather; Qwen2.5 also recognizes the limitation, although awkwardly. Qwen3-1.7B, Qwen3-VL and Llama invent temperatures, conditions, dates or locations. This is a major distinction for production assistants.

### Structured output

Qwen3-1.7B produced the cleanest free-form JSON sample. Gemma, Qwen2-VL and Llama produced semantically correct JSON with repairable wrappers. Qwen2.5 added unwanted prose. Qwen3-VL duplicated JSON blocks and mutated keys (`alter` → `ALTER` / `altern`), which is unsafe for direct schema execution.

### Reasoning

The March 1, 2028 calendar task is the strongest discriminator. The correct answer is **Monday, February 28, 2028**. Only the English Gemma run reaches the correct result. German Gemma, both VLMs, Qwen2.5, Qwen3-1.7B, Llama and DeepSeek all fail in different ways, often inventing impossible dates.

### Language understanding

German and English performance is not symmetric:

- Gemma is strong in both languages, but the hard calendar task succeeds in English and fails in German.
- DeepSeek shows the largest gap: the German France query degenerates into nonsensical text while the English equivalent is answered correctly.
- Qwen3-1.7B produces relatively natural German, but better English phrasing does not fix its date reasoning.
- Qwen2.5 understands basic German but has grammar/refusal problems and weak reasoning.
- Qwen2-VL shows multilingual token corruption on longer German generations.
- Llama can answer simple German facts but remains highly context- and prompt-sensitive.

No controlled Spanish/French/Russian benchmark was supplied, so no broader multilingual ranking should be inferred from this dataset.

## Home Assistant interpretation

A strict 0/10 intent score does not always mean zero semantic understanding.

**Gemma** often identifies the correct action and slots but omits `TARGET_TYPE` or breaks the requested four-field schema. This makes it a plausible semantic fallback behind deterministic validation.

**Qwen3-1.7B** often contains the intended intent token but places it in the wrong column or prints literal schema labels. This is partial understanding, not safe schema compliance.

**Qwen2.5** repeatedly collapses unrelated actions into `STATE_READ_TEMPERATURE`; that is a semantic routing failure rather than a formatting issue.

**Llama3.2** commonly ignores the classification task and answers in prose. Stripping `<|eot_id|>` does not repair that.

The VLMs remain unsuitable for direct HA action generation. DeepSeek's forced thinking mode wastes a small output budget before it can emit the compact schema.

## Technical conclusions

- Measure **TTFT, decode speed and full completion latency** separately.
- Use isolated sessions for latency comparisons; WebGUI history growth materially changes later prompts.
- Strip backend-specific stop tokens (`<|eot_id|>`, DeepSeek end/thinking markers) in the service.
- Do not use a reasoning template that forces `<think>` for trivial classification requests.
- Investigate Qwen2-VL HEF/tokenizer/template/runtime compatibility because its long-form token corruption is unusual.
- Move arithmetic, calendar logic, comparisons, percentages and deterministic HA state/action resolution out of the LLM.
- Validate every generated action against an explicit schema, known entities and allowed values.

## Recommended production roles

| Role | Recommendation |
|---|---|
| Deterministic HA actions/state | Parser + entity/tool resolver + validator |
| Highest-quality general text | **Gemma 4 E2B** |
| Fast native Hailo text | **Qwen3-1.7B-Instruct** |
| Experimental ultra-low-latency text | Qwen2.5-1.5B with narrow scope |
| Image analysis | Select VLM using a separate image benchmark |
| Math/calendar | Deterministic code |
| Live weather/current data | Tool/data source or abstain |
| Raw tool/JSON execution | Never execute without validation |

## Recommended next tests

1. Repeat each controlled scenario 5–10 times and report median/P95.
2. Force a fresh conversation for every WebGUI quality case.
3. Mirror every reasoning task in German and English.
4. Add controlled Russian/French/Spanish tests if those languages matter.
5. Separate unsupported-current-data abstention from tool-assisted live-data tests.
6. Evaluate constrained JSON/schema decoding if supported by the backend.
7. Run a dedicated image/VLM benchmark for camera/Frigate workloads.
8. Split HA scoring into intent, target, value and schema correctness instead of a single PASS/FAIL.
