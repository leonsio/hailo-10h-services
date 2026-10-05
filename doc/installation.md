# Installation and deployment

## Install on Debian / Raspberry Pi / Proxmox LXC

Prerequisites: working Hailo-10H kernel driver, firmware, matching **HailoRT 5.x
GenAI Python wheel** for your architecture and Python version. `hailo_platform`
with `VLM` and `Speech2Text` must already import. The installer does not replace
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

## YAML model selection and download catalogue

Edit `/etc/hailo-10h-services.yaml`; see `deploy/hailo-10h-services.yaml.example`.
The `models` mapping selects Qwen2-VL or Qwen3-VL, Whisper Tiny/Base/Small,
MiniLM retrieval, and optional Gemma E2B on CPU. Hailo HEF LLM execution is
currently disabled pending hardware tests. `models.hailo_llm.enabled: true`
is rejected before any download/device allocation, even if VLM is disabled.
This also applies to ENV overrides and programmatic configuration. The catalogue
retains HEF LLM links for future support. VLM plus Gemma on CPU is supported.
Only enabled models are downloaded at startup, before device allocation, and
remain resident. Requests never cause model swapping. Enabling Gemma requires
`litert-lm` in the service interpreter; rerun the installer after enabling it.
An existing `HAILO_LITERT_MODEL_PATH` remains supported as an ENV override.

All download URLs, including MiniLM host assets and Gemma, are defined in
`src/hailo_services/model_catalog.yaml` (included in the installed Python package).
To customize it, copy it to `/etc/hailo-10h-models.yaml` and set
`settings.model_catalog` to that path. No catalogue is fetched from a helper repository.
The catalogue includes documented 5.1.1/5.2/5.3/5.4 releases; `model_release: auto`
uses the loaded HailoRT binding's exact major/minor. HailoRT 5.4 selects **v5.4.0**.
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

Use `Qwen2-VL-2B-Instruct` for text and image analysis. Gemma is text-only.
Home Assistant or another OpenAI-compatible client can point to the same service
base URL (`https://<raspberry-pi>:8443/v1` with the default HTTPS setup or
`http://<raspberry-pi>:8090/v1` without HTTPS), use the service API key, and
select `gemma-4-E2B-it` or `Qwen2-VL-2B-Instruct`. For a self-signed certificate,
the client must trust that certificate. LiteRT-LM and
Gemma share system RAM with the service; verify memory headroom on the Pi before
raising the request queue size.

For Home Assistant, use the **llama.cpp** conversation integration with the service
URL, API key and model `gemma-4-E2B-it`. Enable Home Assistant control in the
conversation agent options and expose the devices you want Assist to control.
