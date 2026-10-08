# Languages and editable resources

`src/hailo_services/locales/` contains one catalogue per language: German (`de`),
English (`en`), Russian (`ru`), French (`fr`), Spanish (`es`), Italian (`it`),
Dutch (`nl`) and Portuguese (`pt`). HA-Assist, Frigate-Assist and the browser UI use
the same catalogues and supported-code list. Chat `language` accepts these codes
with an optional region suffix such as `fr-FR`.

| JSON section | What to edit |
| --- | --- |
| `text` | Deterministic replies, clarifications and model-facing prompt fragments. Keep all formatting placeholders unchanged. |
| `ui` | Browser status, chat, recording, synthesis and metric labels. |
| `states`, `domains`, `weather`, `wait` | State/domain/weather labels and wait sentences. |
| `aliases`, `patterns` | Local vocabulary mapped to canonical matching concepts and localized state-location patterns. Never translate actual entity names or tool arguments. |
| `ha_intents` | Supplemental HA HassIL sentence forms, currently brightness commands. Official grammars still come from `home-assistant-intents`. |
| `frigate.sentences` | Frigate HassIL forms. Slots resolve only against the current request's camera catalogue and exposed tools. |
| `frigate.vocabulary` | Time units, feature names, object labels and synonyms. Language resources are combined for multilingual matching. Canonical values stay unchanged. |
| `frigate.hints` | Vocabulary used for language detection. |
| `frigate.tool_hints` | Compact model-facing tool descriptions, selected in the request language catalogue. |
| `language_name`, `language_names` | Human-readable language names. |

`de.json.routing` holds older shared canonical matching rules and bilingual/history
patterns. HA matching first normalizes localized aliases into that canonical
representation; this section is not a German reply catalogue. User replies exist
in every language's `text` section. Protocol markers, logs, schema identifiers and
developer errors are stable code contracts.

Explicit request language takes precedence over detection. A `ContextVar` scopes
HA localization to the current request and restores it afterwards. Frigate replies
can request a translation explicitly without changing that context. Image prompt
preparation preserves the detected human-question language even when Frigate adds
an English live-frame caption. Technical model instructions may stay in English
while requiring the selected response language.

To extend an existing phrase, edit its JSON resource and restart the service.
To add a language, create a complete catalogue following the existing schema,
translate the replies/UI/labels, and supply its vocabulary and sentence forms.
The service discovers codes from JSON filenames; schema validation, settings and
UI choices follow that list without a Python allowlist change. Existing grammar
forms and text can change without editing source code. New behavior still needs
an implementation. Run the locale/assistant regression tests after edits.

Speech capabilities remain model-specific. Wyoming advertises Whisper languages
and installed Piper voice languages independently of the UI catalogue. Adding a
UI language does not install a voice or change an ASR model's capabilities.
