# Browser playground

Status, text/image chat and speech tests, request metrics and language controls. `app.js` loads UI catalogues from the API. `recorder-worklet.js` captures mono audio; microphone access requires browser-supported secure context.

The monitoring overview is available at `/monitor` and refreshes every five seconds. Its authenticated `/v1/monitor` endpoint includes `/health` status, one-hour per-service request counts and timings, application/host resources and a bounded recent application log. Metrics are kept in memory for the current gateway process lifetime; old counts expire after one hour. CPU usage needs two refreshes. Logs contain entries emitted in the gateway process; native worker process logs are not collected here.
