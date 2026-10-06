# Installation and deployment

## Install on Debian / Raspberry Pi / Proxmox LXC

Prerequisites: working Hailo-10H kernel driver, firmware, matching **HailoRT 5.x
GenAI Python wheel** for your architecture and Python version. `hailo_platform`
with `VLM`, `LLM` and `Speech2Text` must already import. The installer does not replace
your kernel driver, HailoRT or firmware. In LXC, `/dev/h1x-0` must first be passed
through by the Proxmox host; a service inside LXC cannot grant itself that device.
For sharing with host/other containers, they need compatible HailoRT libraries
and matching group IDs too; validate sharing across those boundaries on hardware.

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
sudo bash scripts/install.sh
# If the Hailo wheel lives in another Python environment:
# sudo HAILO_PYTHON=/path/to/hailo/venv/bin/python bash scripts/install.sh
journalctl -u hailo-10h-services -f
curl http://127.0.0.1:8090/health
```

The installer creates a dedicated `hailo-services` user, a virtual environment
with system packages visible, device/resource ACLs, a YAML configuration and systemd unit.
The service manages its own model catalogue and downloads without installing a
helper repository. Native Hailo wheel
imports are checked again as the service user. For a wheel installed in a custom
venv, the installer adds that environment’s vendor package directory to the
service Python path when necessary. The Python ABI must match. Protected home
paths are not visible to the systemd service; keep vendor environments under
`/opt` or install the matching vendor wheel directly into the service venv.

First installation generates an API key in `/etc/hailo-10h-services.yaml`.
Configuration is preserved on reinstall. The account's home and working directory
are `/var/lib/hailo-10h-services`, managed by systemd `StateDirectory`. Hailo can
write `$HOME/.hailo` and cwd log files there. `ProtectSystem=full` stays enabled
with a `ReadWritePaths` exception for the shared model directory. ACLs grant the
service user access to that exception. A startup preflight checks real temporary
file creation and rename in these locations inside the systemd sandbox. Reinstall
also migrates the original `/nonexistent` account home and clears failed-start
rate limiting. Read/edit this file as root, then:

```bash
sudo systemctl restart hailo-10h-services
sudo systemctl status hailo-10h-services
```

`/health` is the readiness check, not merely `systemctl is-active`: while HEFs are
being downloaded/loaded there is no listening HTTP socket. It includes loaded
model names, paths, mandatory SHARED group, pending work and MQTT connection state.

## Docker Compose

Requires Docker Engine with Compose v2 on a Linux host and a working Hailo-10H
kernel driver/firmware. The image installs the **userspace** runtime, GenAI
wheel, Python dependencies and, by default, optional `litert-lm`. No driver or
DKMS package is installed in the container. The default recipe targets Debian
13 / Python 3.13 / ARM64, matching the Raspberry Pi 5/CM5 setup.

Put the vendor packages in `deploy/vendor/` (exactly one runtime DEB and one
matching wheel). These proprietary binaries are ignored by Git. Use HailoRT
5.4.0 with matching host driver/firmware:

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
cp /root/hailort_5.4.0_arm64.deb deploy/vendor/
cp /root/hailort-5.4.0-cp313-cp313-linux_aarch64.whl deploy/vendor/
cp deploy/hailo-10h-services.yaml.example deploy/hailo-10h-services.yaml
# Edit this file BEFORE startup: set a strong settings.api_key and select models.
nano deploy/hailo-10h-services.yaml
docker compose up -d --build
docker compose logs -f hailo-services
curl http://127.0.0.1:8090/health
```

On x86-64 use matching amd64 DEB and CPython 3.13 x86-64 wheel instead;
LiteRT-LM availability must match that platform. The build verifies GenAI imports.
To omit CPU Gemma's dependency, build with `INSTALL_LITERT=0` and leave Gemma
disabled in YAML. To enable Gemma later, rebuild with `INSTALL_LITERT=1`.
The YAML bind mount is read-only and must exist; Compose refuses to silently
create a directory in its place. This deployment does not generate an API key:
set it in the YAML yourself. Keep this file private (`chmod 600`).

`/dev/h1x-0` is passed explicitly; no privileged container is required. Override
the device or published ports using environment variables, for example:

```bash
HAILO_DEVICE=/dev/h1x-1 HTTP_PORT=8091 WYOMING_PORT=10301 docker compose up -d --build
```

The named `models` volume holds downloaded HEFs, MiniLM assets and Gemma;
`state` holds Hailo logs and runtime home. Downloads are automatic for enabled
models. For existing files, copy them into the model volume before startup or
replace that volume mapping with a writable host directory mounted at
`/usr/local/hailo/resources/models/hailo10h`. Local paths in YAML must be paths
inside the container. Changing `model_store` requires a matching mount.

Compose runs one service process; `SHARED` remains enforced by the application.
Other applications sharing the accelerator need compatible runtimes and matching
group IDs; validate actual concurrent use on hardware. Docker uses its own
network namespace; MCP clients need the correct `mcp_hosts` entries. Configure
MQTT with a reachable external broker address (`localhost` is the container);
MQTT is optional, so no broker is required for HTTP/Wyoming deployment.

The health check tests `/health`, allowing 30 minutes for initial downloads.
An unhealthy status does not automatically restart a running container; inspect
logs if downloads take longer or a configured model fails. HTTP is on 8090 and
Wyoming on 10300; change YAML internal ports only together with the Compose
port mappings and health check.

Update with `git pull --ff-only && docker compose up -d --build`. Stop with
`docker compose down`; the model and state volumes remain. `down -v` deletes
those volumes, including models. HTTPS is optional: use an external TLS reverse
proxy with SSE/WebSocket support. `scripts/enable-https.sh` is intended for the
native systemd installation, not this Docker container.

## Proxmox LXC on ARM64

Run the installer on an **existing ARM64 Proxmox 9 / Pimox host** (Raspberry Pi
5/CM5), not inside a container. It creates a new unprivileged Debian 13 ARM64
LXC with `dev0` passthrough, DHCP on `vmbr0`, 4 cores, 4096 MiB RAM, 512 MiB
swap and 32 GiB disk. Use more RAM when enabling CPU Gemma. It installs no Docker
and no kernel driver in LXC. Host networking, storage and Hailo driver/firmware
must already work. The userspace DEB must be `hailort` or `h10-hailort` 5.4.0.

Put these packages **on the Proxmox host**:

- `/root/hailort_5.4.0_arm64.deb`
- `/root/hailort-5.4.0-cp313-cp313-linux_aarch64.whl`

Supply an existing Debian 13 ARM64 template volume. The template filename below
is an example; replace it with the actual installed template
reported by `pveam list local`. The installer does not guess or download a
possibly unavailable ARM64 template.

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
bash scripts/install-proxmox-lxc.sh \
  --ctid 110 \
  --template local:vztmpl/debian-13-standard_13.6-1_arm64.tar.zst \
  --storage local-lvm --device /dev/h1x-0 --check
# Repeat without --check to create/install the container.
bash scripts/install-proxmox-lxc.sh \
  --ctid 110 \
  --template local:vztmpl/debian-13-standard_13.6-1_arm64.tar.zst \
  --storage local-lvm --device /dev/h1x-0
```

`--help` lists overrides for package paths, memory, disk, cores, bridge, hostname
and repository ref. Without `--device`, one unique `/dev/h1x-*` or `/dev/hailoN`
character device must exist. Proxmox `dev0` handles device permissions for the
unprivileged container. The installer grants the service account a device ACL,
checks `hailortcli --version` and `hailortcli fw-control identify`, and verifies
GenAI imports. Confirm identify reports HAILO10H and firmware 5.4.0. No DKMS,
or GPU passthrough is needed. Sharing still requires
compatible runtimes and `SHARED` clients across all participating processes.

A dedicated `/opt/hailort-venv` holds the vendor wheel; the existing native
installer then provisions the service, its dependencies, API key, ACLs and
systemd unit. Gemma is initially disabled; after enabling it in YAML, rerun the
native installer to install LiteRT-LM. Configuration and model downloads behave
like the native installation. For reused host models, deliberately add a
writable bind mount and matching ACLs rather than copying the driver into LXC.

```bash
pct exec 110 -- journalctl -u hailo-10h-services -f
pct exec 110 -- curl http://127.0.0.1:8090/health
pct exec 110 -- cat /etc/hailo-10h-services.yaml
pct exec 110 -- hostname -I
# Update inside the container:
pct enter 110
cd /opt/hailo-10h-services-source
git pull --ff-only
HAILO_PYTHON=/opt/hailort-venv/bin/python bash scripts/install.sh
```

Use the container IP for HTTP/Wyoming clients. HTTPS can be enabled inside LXC
with the native HTTPS script. Existing CT/VM IDs are rejected; failed installs
retain the container for diagnosis and are not automatically deleted. Fix a
failure inside that container or choose a new ID; this creation script is not an
in-place updater. `--check` validates inputs without creating a container; it
cannot verify the eventual application startup or simultaneous Hailo inference.

## YAML model selection and download catalogue

Edit `/etc/hailo-10h-services.yaml`; see `deploy/hailo-10h-services.yaml.example`.
The `models` mapping selects Qwen2-VL or Qwen3-VL, Whisper Tiny/Base/Small,
MiniLM retrieval, native Hailo HEF LLMs, and optional Gemma E2B on CPU.
Set `models.hailo_llm.enabled: true` and select any `kind: llm` catalogue entry
(e.g. `Qwen2.5-1.5B-Instruct` or `Qwen3-1.7B-Instruct`). The service instantiates
`hailo_platform.genai.LLM` directly on its SHARED VDevice. Ollama is not needed.
Enable either VLM or Hailo LLM according to your hardware; the service intentionally
does not enforce mutual exclusion or swap models on requests. All Hailo inference
shares the serialized owner thread. Gemma runs independently on CPU with a separate
engine and queue and can stay enabled with either Hailo model type.
Legacy ENV equivalents are `HAILO_VLM_ENABLED=false`,
`HAILO_HAILO_LLM_ENABLED=true`, `HAILO_HAILO_LLM_MODEL=Qwen2.5-1.5B-Instruct`,
and `HAILO_HAILO_LLM_MAX_INPUT_TOKENS=2048`. ENV overrides YAML.
Only enabled models are downloaded at startup, before device allocation, and
remain resident. Requests never cause model swapping. Enabling Gemma requires
`litert-lm` in the service interpreter; rerun the installer after enabling it.
An existing `HAILO_LITERT_MODEL_PATH` remains supported as an ENV override.

All download URLs, including MiniLM host assets and Gemma, are defined in
`src/hailo_services/model_catalog.yaml` (included in the installed Python package).
To customize it, copy it to `/etc/hailo-10h-models.yaml` and set
`settings.model_catalog` to that path. No catalogue is fetched from a helper repository.
The catalogue includes documented 5.1.1/5.2/5.3/5.4 releases; `model_release: auto`
uses the loaded HailoRT binding's exact major/minor for Qwen3 and Whisper.
**Qwen2-VL is the exception: its preferred release is v5.1.1**, including with
HailoRT 5.4. Set `models.vlm.release` to override only the VLM release.
Qwen2 downloads live under `<model_store>/<release>/Qwen2-VL-2B-Instruct.hef`,
so an existing HEF at the store root remains untouched. To reuse your existing
v5.1.1 file, set `models.vlm.path` to its full path; an explicit path bypasses
release selection. The final HEF compatibility check occurs when HailoRT loads
it; validate SHARED inference on your hardware after changing versions.
Unknown runtimes and unavailable model/release combinations fail clearly instead
of silently trying an incompatible HEF. Explicit local HEF paths are supported.
The MiniLM HEF is a community build; HailoRT checks compatibility when loading it.

Existing valid local files are reused. The verified 5.4 file sizes also detect
stale/truncated cached HEFs and trigger replacement. Downloads use bounded size checks and
atomic rename; failed downloads do not replace existing files. For older release entries without exact sizes, remove/move an incompatible cache
before downloading that release.
All twelve current 5.4 HEF URLs were verified with HTTP HEAD (200 and file size)
and against the official release documentation, without downloading all weights. Hardware inference must be verified locally.

YAML is loaded safely and validated; unknown keys and invalid types are rejected.
Existing `/etc/hailo-10h-services.env` is optional and retains precedence through
`HAILO_<SETTING>` overrides. Move settings/secrets to YAML and remove the legacy
ENV file when ready. Keep the YAML readable by group `hailo-services` (0640).
All `Settings` fields can be configured under `settings`, including MQTT/secrets.
A non-default model store needs matching permissions and a systemd `ReadWritePaths`
exception. HTTPS setup reads numeric proxy settings from YAML, then legacy ENV.

The complete YAML example includes all legacy ENV parameters, with model paths
and enable flags under `models`; MQTT and MCP settings live under `settings`.
Existing YAML files are preserved by the installer: add the new example keys
manually as needed. Missing settings retain their defaults.

## HTTPS with a local self-signed CA

On an existing installation, enable the additional HTTPS listener with:

```bash
cd ~/hailo-10h-services
git switch main
git pull --ff-only
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services
sudo bash scripts/enable-https.sh
```

Open **`https://<host>:8443/`**. HTTP on port 8090 remains available for existing
clients. nginx terminates TLS and forwards requests to the same resident service;
it does not start another Hailo process. The setup installs nginx/OpenSSL if needed,
adds one nginx server block and preserves existing sites. SSE is unbuffered and
WebSocket upgrades are forwarded. API authentication and limits remain enforced.

Certificates are generated under `/etc/hailo-10h-services/tls`. A self-signed local
root CA signs a 365-day server certificate with Subject Alternative Names for
localhost, the machine hostname, its `.local` name and detected interface IPs.
Add other names or addresses, or choose port 443, explicitly:

```bash
sudo bash scripts/enable-https.sh --dns hailo.example.lan --ip 192.168.1.42 --port 443
```

Re-running the script reuses the CA and valid matching certificates. It renews
the server certificate when names/IPs change or less than 30 days remain; there
is no automatic renewal timer. Keep the CA private key on the server. Both private
keys are root-readable only. The public CA can be downloaded from
`https://<host>:8443/hailo-ca.crt`; its SHA-256 fingerprint is printed during setup.

**Trust is required for reliable microphone capture.** A browser warning is
expected until the local CA is installed and trusted. Simply clicking through a
certificate warning may still prevent microphone access.

For an iPhone/iPad:

1. In Safari open `https://<host>:8443/hailo-ca.mobileconfig` and download the
   certificate profile (initially acknowledge the certificate warning).
2. Install **Hailo-10H Local CA** via Settings → General → VPN & Device Management
   (or the **Profile Downloaded** entry).
3. Under Settings → General → About → Certificate Trust Settings, enable full
   trust for **Hailo-10H Local CA**. Compare the certificate fingerprint with the
   value printed on your server before trusting it.
4. Reload the HTTPS page in Safari and allow microphone access.

On a desktop, import `hailo-ca.crt` into the trusted root certificate store used
by the browser. Remove this trust/profile when the local CA is no longer needed.
See [Apple's manual certificate-trust instructions](https://support.apple.com/102390).

Verify from the server without bypassing certificate validation:

```bash
curl --cacert /etc/hailo-10h-services/tls/hailo-ca.crt https://localhost:8443/health
sudo nginx -t
systemctl status nginx
```

If a firewall is enabled, allow the chosen HTTPS port on the local network. After
an IP change, run setup again and use a name/address included in the certificate.

## Gemma 4 E2B through LiteRT-LM

The full service keeps a LiteRT-LM `Engine` resident and route text-only
OpenAI-compatible chat requests to it. LiteRT-LM runs explicitly on CPU in its
own single-worker queue; Qwen2-VL and Whisper continue to use the Hailo `SHARED`
device. This uses LiteRT-LM's Python API rather than starting a second HTTP
server. The model is loaded once at service startup and each API request gets a
fresh conversation populated from the supplied message history.

Configure `/etc/hailo-10h-services.env`:

```bash
HAILO_LITERT_MODEL_PATH=/var/lib/hailo-10h-services/models/gemma-4-E2B-it.litertlm
```

Then run the installer/update script so the service virtual environment has
`litert-lm` and the service account can read the model. Restart and check
`/health`; the `litert_lm.ready` field and `gemma-4-E2B-it` entry in `/v1/models`
should be present. If LiteRT-LM cannot load, Hailo service startup still
completes and the reason appears in `litert_lm.error` and the service log.
Provision Gemma for text and tool inference. A missing model leaves only the accelerator services available; `/health` reports this degraded configuration.

Chat requests select the backend using the model field:

```json
{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"Hallo!"}],"max_tokens":128}
```

Use the selected `Qwen2-VL-2B-Instruct` or `Qwen3-VL-2B-Instruct` for text and
image analysis. The service asks the loaded VLM for its frame shape; the catalogue
fallback is 336×336 for Qwen2 and 512×288 for Qwen3. Qwen3 accepts one image per
request; Qwen2 accepts up to four. Frames are writable contiguous RGB UINT8 arrays.
Gemma and Hailo LLMs are text-only. Omitted `model` selects ready Gemma, then
an enabled Hailo LLM, then VLM for text; images require an enabled VLM.
Explicit model IDs select that backend without fallback. The web model selector
lists the enabled models, disables images for text-only models, and retains
history when switching. Image parts are omitted only from requests to text models.

Configure the model-bound limits beside their models:

```yaml
models:
  vlm:
    enabled: true
    model: Qwen3-VL-2B-Instruct # Or Qwen2-VL-2B-Instruct
    release: auto
    max_input_tokens: 2048
  gemma:
    enabled: false
    max_input_tokens: 4096
  hailo_llm:
    enabled: false # Set true (and VLM false above) to load a native Hailo LLM.
    model: Qwen2.5-1.5B-Instruct
    max_input_tokens: 2048
```

Gemma's 4096 is the service's RAM safety default, not the original model's
manufacturer context maximum. Hailo's Qwen2/Qwen3 VLM and listed HEF LLM context
limits are 2048 (catalogue `context_length`). Output, templates, selected tools
and image tokens also consume context; the usable text input may be smaller.
The API `max_input_tokens` can only lower the configured budget. HTTP requests
may exceed it (up to `max_body`) because retrieval/compilation precedes the final
native-tokenizer check. Required current-turn contents that still do not fit
produce HTTP 400 `input_token_limit_exceeded`. Debug logs show the final prompt
(`final_vlm_request` or `final_llm_request`) and measured budget
(`vlm_input_budget` or `llm_input_budget`); VLM logs also show frame size/image count.
Native Hailo LLMs render their own HEF template with string text content and
`enable_thinking=false`, then pass that exact measured raw prompt to `LLM.generate`.
Both Hailo chat types require `temperature > 0` (default 0.1). Gemma also accepts 0.
Validate loading/inference on the target hardware; automated tests use simulated
Hailo bindings and cannot establish hardware residency compatibility.
Home Assistant or another OpenAI-compatible client can point to the same service
base URL (`https://<raspberry-pi>:8443/v1` with the default HTTPS setup or
`http://<raspberry-pi>:8090/v1` without HTTPS), use the service API key, and
select `gemma-4-E2B-it`, the enabled Hailo LLM ID, or `Qwen2-VL-2B-Instruct`. For a self-signed certificate,
the client must trust that certificate. LiteRT-LM and
Gemma share system RAM with the service; verify memory headroom on the Pi before
raising the request queue size.

For Home Assistant, use the **llama.cpp** conversation integration with the service
URL, API key and model `HA-Assist`. Configure its text/image targets as below. Enable Home Assistant control in the
conversation agent options and expose the devices you want Assist to control.


## Configure HA-Assist

The new virtual model is enabled by default. **Existing HA agents must change
model selection to `HA-Assist`**; physical Gemma/Qwen model IDs now use the ordinary
inference path without HA optimizations. The default text target is Gemma; fresh
installations with Gemma disabled must enable it or select an enabled Hailo LLM.
The service never substitutes the VLM for a disabled text target.

Merge into `/etc/hailo-10h-services.yaml`:

```yaml
settings:
  ha_assist_enabled: true
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct
  ha_assist_fuzzy_enabled: true
  ha_assist_fuzzy_threshold: 90.0
  ha_assist_fuzzy_margin: 8.0
models:
  gemma:
    enabled: true
    max_input_tokens: 4096
```

Keep your existing model settings. If using Qwen3-VL, update the vision target to
its exact enabled ID. If using a native Hailo text LLM, set the text target to
`models.hailo_llm.model`. Enable those backends before routing to them; setting a
target does not download/enable it. All keys have `HAILO_<SETTING>` ENV overrides.

Update/reinstall the service dependencies to install HassIL, the pinned official
HA grammar package and RapidFuzz, then restart. The installer preserves existing
configuration files, so add these keys yourself. Check `/health.ha_assist` for
`text_ready`/`vision_ready`. `/v1/models` includes `HA-Assist`; its availability
is per target. See [request pipelines](pipelines.md) for deterministic handling,
diagnostics and the guarantee of one generative target per request.

## Piper CPU text-to-speech

Native/LXC installation and Docker images include the optional `piper` Python
extra. Piper is disabled by default and no voice is downloaded during service
startup. Existing installations can install it in their service environment:

```bash
sudo /opt/hailo-10h-services/venv/bin/pip install 'piper-tts>=1.3,<2'
sudo install -d -o hailo-services -g hailo-services /var/lib/hailo-10h-services/piper
sudo -u hailo-services /opt/hailo-10h-services/venv/bin/python -m piper.download_voices \
  --download-dir /var/lib/hailo-10h-services/piper de_DE-thorsten-medium
```

Set the following in `/etc/hailo-10h-services.yaml`, then restart the service:

```yaml
settings:
  piper_enabled: true
  piper_voice: de_DE-thorsten-medium
  piper_voice_dir: /var/lib/hailo-10h-services/piper
  piper_language: de_DE
  piper_max_input_chars: 4096
```

Merge these keys into your existing `settings` block. Matching ENV overrides are
`HAILO_PIPER_ENABLED`, `HAILO_PIPER_VOICE`, `HAILO_PIPER_VOICE_DIR`,
`HAILO_PIPER_LANGUAGE` and `HAILO_PIPER_MAX_INPUT_CHARS`. Restart with
`sudo systemctl restart hailo-10h-services` and inspect `/health` → `piper`.
The update script installs Piper dependencies when Piper is enabled.

Each installed voice requires both `<voice>.onnx` and `<voice>.onnx.json`.
For English, for example, install `en_US-lessac-medium` and configure
`piper_voice: en_US-lessac-medium` plus `piper_language: en_US`. The language
must match the default voice metadata. Multiple installed voices can be selected
by `voice` per API request; a single model remains resident and is replaced when
the voice changes. A trusted absolute ONNX path is also allowed for the configured
default, but clients can only select IDs from `piper_voice_dir`.

For Docker, the default voice directory lives in the persistent `state` volume.
After building the image, provision the voice there before enabling Piper:

```bash
docker compose run --rm --no-deps --entrypoint /opt/venv/bin/python hailo-services \
  -m piper.download_voices --download-dir /var/lib/hailo-10h-services/piper \
  de_DE-thorsten-medium
docker compose up -d --build
```

Use the same YAML keys in `deploy/hailo-10h-services.yaml`. Keep voice files
readable by the service account. Piper runs on CPU with CUDA disabled and uses
no Hailo device/model resources. Voice licences vary; inspect the chosen voice's
model card. See the [speech API](api.md#post-v1audiospeech) for request formats.

### Piper in Home Assistant via Wyoming

Once `/health` reports `piper.ready: true`, Piper is available over both HTTP and
the existing Wyoming listener (default TCP port 10300). Update/restart the service,
then reload its **Wyoming Protocol** integration in Home Assistant. Select
**hailo-piper** as the text-to-speech provider in the Assist voice assistant and
choose the installed voice. There is no separate TTS port or API key for Wyoming.
See [Wyoming STT and TTS](api.md#home-assistant-wyoming-stt-and-tts).

Piper emits INFO startup logs with the installed `piper-tts` version, CPU device,
default voice, configured/actual language, ONNX path, sample rate and load time.
Successful startup logs queue/input/audio limits; disabled Piper logs
`Piper TTS disabled`. Switching voices logs the newly loaded voice and language.
