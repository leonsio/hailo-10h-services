# Hailo-10H Services

One resident gateway for **Qwen2-VL**, **Whisper Base**, **MiniLM** and **Gemma 4 E2B**.
Qwen2-VL handles images, Whisper speech, MiniLM HA context retrieval, and Gemma
text/tool reasoning. Hailo models use `VDevice group_id="SHARED"`; Gemma runs
through LiteRT-LM on the CPU with its own serialized queue. Models stay loaded.

## Install and update

Requires working HailoRT GenAI, matching HEFs and access to `/dev/h1x-0`.
Provision the Gemma `.litertlm` artifact and configure `HAILO_LITERT_MODEL_PATH`.
MiniLM's HEF/tokenizer resources are prepared at startup.

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
sudo bash scripts/install.sh
```

For an existing installation:

```bash
git pull --ff-only
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services
```

Configure `/etc/hailo-10h-services.env`. Gemma's default input ceiling remains
**4096 tokens**. `/health` reports actual readiness, loaded models and errors;
a configured but unavailable Gemma never silently falls back to another model.

## Usage

- Browser playground: `http://<host>:8090/` (chat, images, speech and status).
- OpenAI clients/HA: `http://<host>:8090/v1`, model `gemma-4-E2B-it` for text and
  device control, `Qwen2-VL-2B-Instruct` for images. Use the configured API key.
- Wyoming STT: port **10300**, multilingual Whisper Base.
- MCP `/mcp`, WebSocket `/ws`, MQTT and HTTPS are supported.

The UI uses the first supported browser language, with a persistent manual
language selector and the service default as fallback. German, English and
Russian resources live in `src/hailo_services/locales/`. `HAILO_SERVICE_LANGUAGE`
sets the general reply/UI fallback; `HAILO_LANGUAGE` sets Whisper's default.
HA requests detect their input language or accept an explicit `language` field.
Entity names and API identifiers retain the values supplied by HA.

Recognized HA requests use conservative deterministic paths first; ambiguous
requests use MiniLM and the minimal prompt compiler before Gemma. Other OpenAI
requests keep their system prompts, tools and conversation intact. Streaming HA
requests can receive a varying localized wait sentence at actual Gemma inference
start. Tool output remains buffered until validated. Spoken early playback also
requires streaming support in the HA agent and TTS provider.

## Documentation

- [Installation, Gemma provisioning and HTTPS](doc/installation.md)
- [APIs and Home Assistant integration](doc/api.md)
- [Routing and deterministic pipelines](doc/pipelines.md)
- [Languages, vocabulary and Wyoming/HA language selection](doc/languages.md)
- [Diagnostics, memory and hardware checks](doc/troubleshooting.md)

## Verification

```bash
pip install -e '.[test]'
ruff check .
pytest -q
node --test tests/test_web.cjs
bash -n scripts/install.sh scripts/enable-https.sh
```

Protocol/routing tests use simulated backends; accelerator initialization,
measured inference times and audible Assist streaming require target hardware.
