# Languages and Wyoming

## Resources and precedence

`locales/de.json`, `en.json` and `ru.json` contain UI text, deterministic responses,
minimal prompt text, domain/state/weather labels, varied wait sentences and
input/device/group aliases. Matching grammar is stored externally with the
canonical German resource and shared across languages; aliases canonicalize
recognizable words for the existing parsers. This is matching, not translation
of an arbitrary user sentence. Actual entity names, areas, tool identifiers and
original model-facing user messages retain their original spelling.

Browser selection: saved manual choice → first supported `navigator.languages`
entry → `HAILO_SERVICE_LANGUAGE`. “Automatic” restores browser selection. Only
the interface preference is saved locally; API keys and chat/media are not saved.

HA response selection: explicit JSON `language` (`de`, `en`, `ru`, optionally a
region) → input-language detection → `HAILO_SERVICE_LANGUAGE`. Request-scoped
contexts prevent simultaneous users from changing one another's language.
Whisper's independent default is `HAILO_LANGUAGE`; an explicit STT language wins.
The API/UI exposes `stt_languages` separately from UI languages.

Vocabulary includes light/lamp, switch, cover, lock, climate, vacuum, weather,
temperature, humidity, percentages and common action/state words. It can match
“Licht”, “Light” or “свет”, including localized generic names within a device or
area name. Unknown proper names are not automatically translated; HA aliases
are the appropriate way to expose them in several languages. MiniLM's existing
model is retained; vocabulary matching helps explicit multilingual requests,
while free-form semantic coverage depends on that model and Gemma. Adding an
alias does not warrant a direct action without an unambiguous HA target.

## Why HA might offer only German

The service's Wyoming `describe` response already advertises `de`, `en`, `ru` and
other supported Whisper languages. `HAILO_LANGUAGE=de` does **not** reduce that
list. Tests verify the model language advertisement independently of defaults.

HA's [Wyoming STT provider](https://github.com/home-assistant/core/blob/dev/homeassistant/components/wyoming/stt.py)
collects the languages of installed advertised ASR models. Assist's
[language-list API](https://github.com/home-assistant/core/blob/dev/homeassistant/components/assist_pipeline/websocket_api.py)
intersects STT, conversation and TTS language support. A German-only agent or
TTS voice can therefore limit the complete pipeline despite multilingual STT.

To diagnose an installation:

1. Check `/ui/config` → `stt_languages` and the Wyoming `describe` model list.
2. Reload the Wyoming integration after service changes to refresh discovery.
3. Check the conversation agent's advertised supported languages.
4. Check the TTS provider/installed voice models for English and Russian.
5. Create separate Assist pipelines/voices for each required language.

Without the actual HA configuration, the limiting component cannot be identified
conclusively. No artificial German restriction was found in this service.

## Extension checks

Keep `text`/`ui` keys and format placeholders identical across locales; the tests
verify them. Never interpolate UI translations with `innerHTML`; translations
use `textContent` and explicitly localized safe attributes. Locale asset routes
are exact read-only public endpoints and do not expose service secrets.
