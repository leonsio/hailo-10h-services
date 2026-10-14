"use strict";
const $ = id => document.getElementById(id);
const keyStorageName = "hailo-api-key";
try { $("key").value = sessionStorage.getItem(keyStorageName) || ""; } catch { /* Storage may be disabled. */ }
$("key").addEventListener("input", () => {
  try {
    const key = $("key").value.trim();
    if (key) sessionStorage.setItem(keyStorageName, key);
    else sessionStorage.removeItem(keyStorageName);
  } catch { /* The page can still send a key without storage. */ }
});
const tr = (key, values) => window.HailoUI.t(key, values);
let lastData = null;
const locale = () => window.HailoUI.language;
const fmt = value => typeof value === "number" ? new Intl.NumberFormat(locale(), {maximumFractionDigits: 1}).format(value) : "—";
const bytes = value => typeof value === "number" ? `${fmt(value / 1048576)} MiB` : "—";
const when = value => value ? new Date(value).toLocaleString(locale()) : tr("monitor.no_request");
const item = (label, value) => {
  const row = document.createElement("div"), name = document.createElement("span"), content = document.createElement("strong");
  name.textContent = label; content.textContent = String(value ?? "—"); row.append(name, content); return row;
};
let selectedProcess = "gateway", selectedService = "Alle", renderedLogKey = null;
const logPositions = new Map();
let appliedTop = 0, appliedLeft = 0;
function rememberLogPosition(lines) {
  if (!renderedLogKey) return;
  const position = logPositions.get(renderedLogKey) || {top: 0, left: 0, follow: false};
  // Ignore programmatic/clamped scroll events after a refresh. Keep the user's
  // desired horizontal offset even if a shorter batch temporarily cannot fit it.
  if (lines.scrollLeft !== appliedLeft) { position.left = lines.scrollLeft; appliedLeft = lines.scrollLeft; }
  if (lines.scrollTop !== appliedTop) {
    position.top = lines.scrollTop; position.follow = lines.scrollHeight - lines.clientHeight - lines.scrollTop <= 12;
    appliedTop = lines.scrollTop;
  }
  logPositions.set(renderedLogKey, position);
  if (logPositions.size > 48) logPositions.delete(logPositions.keys().next().value);
}
function replaceTabs(id, buttons) {
  const target = $(id), left = target.scrollLeft;
  target.replaceChildren(...buttons); target.scrollLeft = left;
}
const processNames = {gateway: "Gateway", llm: "LLM", hailo: "Hailo", piper: "Piper"};
function tab(label, selected, onClick) {
  const button = document.createElement("button");
  button.type = "button"; button.role = "tab"; button.textContent = label;
  button.setAttribute("aria-selected", String(selected)); button.addEventListener("click", onClick);
  return button;
}
function serviceOf(line) {
  const match = line.match(/\bservice=(.+?)(?:\s+\w+=|$)/) || line.match(/\boperation=([^ ]+)/);
  return match ? match[1] : null;
}
const logFilters = {event: "", operation: "", protocol: "", model: ""};
function logTag(line, name) {
  const names = name === "event" ? "events?" : name;
  const match = line.match(new RegExp(`\\b(?:${names})=(.+?)(?=\\s+[\\w.-]+=|$)`));
  return match ? match[1].trim() : null;
}
function renderLogs(groups) {
  const ordered = ["gateway", "llm", "hailo", "piper"].map(name => groups.find(group => group.name === name)).filter(Boolean);
  if (!ordered.some(group => group.name === selectedProcess)) selectedProcess = ordered[0]?.name || "gateway";
  replaceTabs("log-tabs", ordered.map(group => tab(`${processNames[group.name] || group.name} · ${group.pid ?? "—"}`, group.name === selectedProcess, () => {selectedProcess = group.name; selectedService = "Alle"; renderLogs(groups);} )));
  const group = ordered.find(entry => entry.name === selectedProcess);
  const services = ["Alle", ...new Set((group?.lines || []).map(serviceOf).filter(Boolean))];
  if (!services.includes(selectedService)) selectedService = "Alle";
  replaceTabs("service-tabs", services.map(service => tab(service === "Alle" ? tr("monitor.all") : service, selectedService === service, () => {selectedService = service; renderLogs(groups);} )));
  for (const name of Object.keys(logFilters)) {
    const select = $(`log-${name}`);
    if (!select) continue;
    const values = [...new Set((group?.lines || []).map(line => logTag(line, name)).filter(Boolean))].sort();
    if (logFilters[name] && !values.includes(logFilters[name])) values.push(logFilters[name]);
    select.replaceChildren(...["", ...values].map(value => { const option = document.createElement("option"); option.value = value; option.textContent = value || tr("monitor.all"); return option; }));
    select.value = logFilters[name];
  }
  const selected = (group?.lines || []).filter(line => (selectedService === "Alle" || serviceOf(line) === selectedService)
    && Object.entries(logFilters).every(([name, value]) => !value || logTag(line, name) === value));
  const key = `${group?.name}:${group?.pid}:${selectedService}:${JSON.stringify(logFilters)}`;
  const panel = $("logs");
  let article = panel.firstElementChild;
  if (!article) {
    article = document.createElement("article");
    const log = document.createElement("pre"); log.addEventListener("scroll", () => rememberLogPosition(log));
    article.append(document.createElement("h3"), log);
    panel.replaceChildren(article);
  }
  const [heading, lines] = article.children;
  rememberLogPosition(lines);
  heading.textContent = tr("monitor.log_heading", {name: processNames[group?.name] || group?.name || tr("monitor.process"), pid: group?.pid ?? "—", shown: selected.length, total: group?.lines.length || 0});
  const content = selected.join("\n") || tr("monitor.no_logs");
  if (lines.textContent !== content || key !== renderedLogKey) {
    const position = logPositions.get(key) || {top: 0, left: 0, follow: false};
    lines.textContent = content;
    lines.scrollTop = position.follow ? lines.scrollHeight : position.top;
    lines.scrollLeft = position.left;
    appliedTop = lines.scrollTop; appliedLeft = lines.scrollLeft;
    logPositions.set(key, position);
  }
  renderedLogKey = key;
}
function render(data) {
  lastData = data;
  $("restart").hidden = !data.restart_available;
  const startup = data.startup || {};
  $("startup").textContent = startup.state === "loading" ? tr("monitor.loading", {phase: startup.phase || ""}) : Object.entries(startup.errors || {}).map(([name, error]) => `${name}: ${error}`).join(" · ");
  $("startup").className = startup.state === "error" ? "bad" : "hint";
  const pageX = window.scrollX, pageY = window.scrollY;
  const h = data.health || {}, r = data.resources || {};
  const ready = h.ready && (!h.vision?.enabled || h.vision?.ready);
  $("state").textContent = ready ? tr("ui_1") : tr("monitor.limited");
  $("state").className = `pill ${ready ? "good" : "bad"}`;
  $("updated").textContent = tr("monitor.updated", {updated: when(data.generated_at), started: when(data.started_at)});
  const cards = Object.entries(data.services || {}).filter(([, service]) => service.enabled).map(([name, s]) => {
    const card = document.createElement("article"), title = document.createElement("h3");
    title.textContent = name;
    if (s.models?.length && !["HA-Assist", "Frigate-Assist"].includes(name)) { const model = document.createElement("small"); model.textContent = ` (${s.models.join(" / ")})`; title.append(model); }
    card.append(title);
    if (s.available === false) card.append(item(tr("monitor.readiness"), tr("metrics.unavailable")));
    card.append(item(tr("monitor.requests"), fmt(s.requests_1h)), item(tr("monitor.errors"), fmt(s.errors_1h)),
      item(tr("monitor.active"), fmt(s.active)), item(tr("monitor.average"), s.average_ms == null ? "—" : `${fmt(s.average_ms)} ms`),
      item(tr("monitor.maximum"), s.maximum_ms == null ? "—" : `${fmt(s.maximum_ms)} ms`));
    if (["LLM", "VLM", "Hailo LLM", "HA-Assist", "Frigate-Assist"].includes(name))
      card.append(item(tr("monitor.ttft"), s.average_ttft_ms == null ? "—" : tr("monitor.ttft_value", {time: fmt(s.average_ttft_ms), count: fmt(s.ttft_samples_1h)})));
    card.append(item(tr("monitor.last_request"), when(s.last_request_at)));
    return card;
  });
  $("services").replaceChildren(...cards);
  if (!cards.length) { const empty = document.createElement("p"); empty.textContent = tr("monitor.no_services"); $("services").append(empty); }
  const memory = r.host_memory || {};
  $("resources").replaceChildren(item(tr("monitor.host_temperature"), r.host_temperature_c == null ? tr("metrics.unavailable") : `${fmt(r.host_temperature_c)} °C`),item(tr("monitor.cpu_cores"), fmt(r.cpu_cores)), item(tr("monitor.load"), fmt(r.load_average)),
    item(tr("monitor.cpu_cgroup"), r.cgroup_cpu_percent == null ? tr("metrics.pending") : `${fmt(r.cgroup_cpu_percent)} %`),
    item(tr("monitor.cpu_gateway"), r.process_cpu_percent == null ? tr("metrics.pending") : `${fmt(r.process_cpu_percent)} %`),
    item(tr("monitor.ram_gateway"), bytes(r.process_rss_bytes)), item(tr("monitor.ram_cgroup"), bytes(r.cgroup_memory_bytes)),
    item(tr("monitor.cgroup_limit"), bytes(r.cgroup_memory_limit_bytes)), item(tr("monitor.ram_available"), bytes(memory.available_bytes)),
    item(tr("monitor.ram_total"), bytes(memory.total_bytes)));
  $("processes").replaceChildren(...(r.processes || []).map(process => {
    const row = document.createElement("tr");
    for (const value of [process.name, process.pid ?? "—", process.cpu_percent == null ? "—" : `${fmt(process.cpu_percent)} %`,
      process.rss_bytes == null ? process.status : bytes(process.rss_bytes)]) {
      const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell);
    }
    return row;
  }));
  const system = data.system_services || {}, host = system.host || {};
  const statuses = {running: tr("common.running"), listening: tr("ui_1"), connected: tr("ui_3"), disconnected: tr("ui_4"), disabled: tr("common.off"), stopped: tr("common.stopped")};
  const address = (name, port) => name ? `${name.includes(":") ? `[${name}]` : name}:${port ?? "—"}` : "—";
  const transport = (name, service, endpoint) => item(name, `${endpoint} · ${statuses[service?.status] || tr("common.unknown")}`);
  const http = system.http || {}, local = http.local_address;
  $("health").replaceChildren(item("Host", `${host.name || "—"} · ${statuses[host.status] || tr("common.unknown")}`),
    item(tr("monitor.local_address"), local ? address(String(local[0]), local[1]) : address(http.host, http.port)),
    transport("HTTP / MCP", http, address(http.host, http.port)),
    transport("Wyoming", system.wyoming, address(system.wyoming?.host, system.wyoming?.port)),
    transport("MQTT", system.mqtt, address(system.mqtt?.host, system.mqtt?.port)),
    transport("ZMQ", system.zmq, system.zmq?.endpoint || "—"));
  const device = data.hailo_device || {};
  const percent = value => value == null ? tr("metrics.unavailable") : `${fmt(value)} %`;
  const gib = value => new Intl.NumberFormat(locale(), {minimumFractionDigits: 1, maximumFractionDigits: 1}).format(value / 1073741824);
  const ram = device.ram_used_bytes == null || device.ram_total_bytes == null ? tr("metrics.unavailable")
    : `${gib(device.ram_used_bytes)} / ${gib(device.ram_total_bytes)} GiB`;
  $("hailo-device").replaceChildren(item("NNC", percent(device.nnc_percent)), item("CPU", percent(device.cpu_percent)),
    item("RAM", ram),
    item(tr("monitor.temperature"), device.temperature_c == null ? tr("metrics.unavailable") : `${fmt(device.temperature_c)} °C`),
    item("Throttling", device.throttling == null ? tr("metrics.unavailable") : device.throttling ? tr("common.yes") : tr("common.no")));
  $("hailo-errors").hidden = !Object.keys(device.errors || {}).length;
  $("hailo-errors-text").textContent = Object.entries(device.errors || {}).map(([name, error]) => `${name}: ${error}`).join("\n");
  const age = device.sampled_at ? Date.now() - new Date(device.sampled_at).getTime() : null;
  $("hailo-note").textContent = device.sampled_at ? tr("monitor.hailo_updated", {time: when(device.sampled_at)}) + (age > 15000 ? ` · ${tr("monitor.stale")}` : device.state === "partial" ? ` · ${tr("monitor.partial")}` : device.state === "unavailable" ? ` · ${tr("monitor.no_measurements")}` : "")
    : device.state === "pending" ? tr("metrics.pending") : tr("monitor.no_measurements");
  $("resident-models").replaceChildren(...(device.resident_models || []).map(model => item(model.name, model.state === "loaded" ? tr("common.loaded") : model.state)));
  if (!device.resident_models?.length) $("resident-models").append(item(tr("ui_52"), tr("monitor.no_models")));
  renderLogs(data.logs || []);
  if (window.scrollX !== pageX || window.scrollY !== pageY) window.scrollTo(pageX, pageY);
}
async function refresh() {
  try {
    const key = $("key").value.trim();
    const response = await fetch("/v1/monitor", {cache: "no-store", headers: key ? {Authorization: `Bearer ${key}`} : {}});
    if (!response.ok) throw new Error(response.status === 401 ? tr("ui_0") : `HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("state").textContent = tr("ui_6"); $("state").className = "pill bad";
    $("updated").textContent = error.message;
  }
}
async function init() {
  let config;
  try {
    const response = await fetch("/ui/config", {cache: "no-store"});
    if (response.ok) { config = await response.json(); $("access").hidden = !config.auth_required; }
  } catch { /* Show the key field if the public configuration is unavailable. */ }
  try { await window.HailoUI.load(window.HailoUI.savedChoice(), config); } catch {}
  await refresh();
}
window.HailoUI.subscribe(() => { if (lastData) render(lastData); });
$("apply").addEventListener("click", refresh);
$("refresh").addEventListener("click", refresh);
$("key").addEventListener("keydown", event => { if (event.key === "Enter") refresh(); });
void init(); setInterval(refresh, 5000);

for (const name of Object.keys(logFilters)) $(`log-${name}`)?.addEventListener("change", event => {
  logFilters[name] = event.target.value; if (lastData) renderLogs(lastData.logs || []);
});
$("restart").addEventListener("click", async () => {
  $("restart").disabled = true;
  try {
    const key = $("key").value.trim();
    const response = await fetch("/v1/settings/restart", {method: "POST", headers: key ? {Authorization: `Bearer ${key}`} : {}});
    if (!response.ok) { const body = await response.json(); throw new Error(body.detail || `HTTP ${response.status}`); }
    $("updated").textContent = tr("settings.restarting");
  } catch (error) { $("updated").textContent = error.message; $("restart").disabled = false; }
});
