# Browser playground

Status, text/image chat and speech tests, request metrics and language controls. `app.js` loads UI catalogues from the API. `recorder-worklet.js` captures mono audio; microphone access requires browser-supported secure context.

The monitoring overview is available at `/monitor` and refreshes every five seconds. Its authenticated `/v1/monitor` endpoint includes `/health` status, one-hour per-service request counts and timings, application/host resources per gateway and worker PID, and up to 50 recent application log lines separately for each service process. Metrics are kept in memory for the current gateway process lifetime; old counts expire after one hour. CPU usage needs two refreshes. Detailed inference/request logs require `settings.debug_log`; the regular journal remains the complete log source.

The Playground, monitoring and settings pages share an API key in `sessionStorage` for the current browser tab. `settings.api_no_auth_networks` can allow selected HTTP/WebSocket socket peers to access the API without a key, independently from the MCP exemption.

The `/settings` page edits existing scalar YAML fields, validates the resulting configuration, and writes it atomically with a version check. API key and MQTT password values are never sent to the browser; leaving them blank retains their current values. Saving reformats YAML comments away. Every setting requires a service restart because the gateway and workers load the configuration at startup. The systemd deployment exposes a restart button; `scripts/install.sh` and `scripts/update.sh` migrate `/etc/hailo-10h-services.yaml` to a writable state file and retain the `/etc` path as a symlink. Environment overrides still win on restart. The log view switches among Gateway/LLM/Hailo/Piper process tabs and offers optional service/operation filters for tagged lines.

The Hailo-10H pane beneath MQTT combines the resident worker and loaded model status with PCIe device, driver, link and optional hwmon temperature. It reads sysfs without opening the accelerator; fields unavailable inside containers or unsupported by the driver remain blank.
