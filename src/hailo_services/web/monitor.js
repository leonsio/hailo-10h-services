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
const locale = navigator.language || "de";
const fmt = value => typeof value === "number" ? new Intl.NumberFormat(locale, {maximumFractionDigits: 1}).format(value) : "—";
const bytes = value => typeof value === "number" ? `${fmt(value / 1048576)} MiB` : "—";
const when = value => value ? new Date(value).toLocaleString(locale) : "Noch keine Anfrage";
const item = (label, value) => {
  const row = document.createElement("div"), name = document.createElement("span"), content = document.createElement("strong");
  name.textContent = label; content.textContent = String(value ?? "—"); row.append(name, content); return row;
};
let selectedProcess = "gateway", selectedService = "Alle";
const processNames = {gateway: "Gateway", llm: "LLM", hailo: "Hailo", piper: "Piper"};
function tab(label, selected, onClick) {
  const button = document.createElement("button");
  button.type = "button"; button.role = "tab"; button.textContent = label;
  button.setAttribute("aria-selected", String(selected)); button.addEventListener("click", onClick);
  return button;
}
function serviceOf(line) {
  const match = line.match(/\bservice=([^ ]+)/) || line.match(/\boperation=([^ ]+)/);
  return match ? match[1] : null;
}
function renderLogs(groups) {
  const ordered = ["gateway", "llm", "hailo", "piper"].map(name => groups.find(group => group.name === name)).filter(Boolean);
  if (!ordered.some(group => group.name === selectedProcess)) selectedProcess = ordered[0]?.name || "gateway";
  $("log-tabs").replaceChildren(...ordered.map(group => tab(`${processNames[group.name] || group.name} · ${group.pid ?? "—"}`, group.name === selectedProcess, () => {selectedProcess = group.name; selectedService = "Alle"; renderLogs(groups);} )));
  const group = ordered.find(entry => entry.name === selectedProcess);
  const services = ["Alle", ...new Set((group?.lines || []).map(serviceOf).filter(Boolean))];
  if (!services.includes(selectedService)) selectedService = "Alle";
  $("service-tabs").replaceChildren(...services.map(service => tab(service, selectedService === service, () => {selectedService = service; renderLogs(groups);} )));
  const article = document.createElement("article"), heading = document.createElement("h3"), lines = document.createElement("pre");
  const selected = (group?.lines || []).filter(line => selectedService === "Alle" || serviceOf(line) === selectedService);
  heading.textContent = `${processNames[group?.name] || group?.name || "Prozess"} · PID ${group?.pid ?? "—"} · ${selected.length}/${group?.lines.length || 0} Zeilen`;
  lines.textContent = selected.join("\n") || "Keine passenden Logeinträge. Dienstfilter zeigen nur Einträge mit service= oder operation= an.";
  article.append(heading, lines); $("logs").replaceChildren(article);
}
function render(data) {
  const h = data.health || {}, r = data.resources || {};
  const ready = h.ready && (!h.vision?.enabled || h.vision?.ready);
  $("state").textContent = ready ? "Bereit" : "Eingeschränkt";
  $("state").className = `pill ${ready ? "good" : "bad"}`;
  $("updated").textContent = `Stand: ${when(data.generated_at)} · Start: ${when(data.started_at)}`;
  const cards = Object.entries(data.services || {}).map(([name, s]) => {
    const card = document.createElement("article"), title = document.createElement("h3");
    title.textContent = name; card.append(title);
    card.append(item("Anfragen", fmt(s.requests_1h)), item("Fehler", fmt(s.errors_1h)),
      item("Aktiv", fmt(s.active)), item("Ø Antwort", s.average_ms == null ? "—" : `${fmt(s.average_ms)} ms`),
      item("Längste Antwort", s.maximum_ms == null ? "—" : `${fmt(s.maximum_ms)} ms`));
    if (["LLM", "VLM", "Hailo LLM", "HA-Assist", "Frigate-Assist"].includes(name))
      card.append(item("Ø TTFT", s.average_ttft_ms == null ? "—" : `${fmt(s.average_ttft_ms)} ms (${fmt(s.ttft_samples_1h)} Messungen)`));
    card.append(item("Letzte Anfrage", when(s.last_request_at)));
    return card;
  });
  $("services").replaceChildren(...cards);
  const memory = r.host_memory || {};
  $("resources").replaceChildren(item("CPU-Kerne", fmt(r.cpu_cores)), item("Systemlast (1 Min.)", fmt(r.load_average)),
    item("CPU Service-cgroup", r.cgroup_cpu_percent == null ? "Messung läuft …" : `${fmt(r.cgroup_cpu_percent)} %`),
    item("CPU Hauptprozess", r.process_cpu_percent == null ? "Messung läuft …" : `${fmt(r.process_cpu_percent)} %`),
    item("RAM Hauptprozess", bytes(r.process_rss_bytes)), item("RAM Service-cgroup", bytes(r.cgroup_memory_bytes)),
    item("cgroup Limit", bytes(r.cgroup_memory_limit_bytes)), item("RAM Host verfügbar", bytes(memory.available_bytes)),
    item("RAM Host gesamt", bytes(memory.total_bytes)));
  $("processes").replaceChildren(...(r.processes || []).map(process => {
    const row = document.createElement("tr");
    for (const value of [process.name, process.pid ?? "—", process.cpu_percent == null ? "—" : `${fmt(process.cpu_percent)} %`,
      process.rss_bytes == null ? process.status : bytes(process.rss_bytes)]) {
      const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell);
    }
    return row;
  }));
  $("health").replaceChildren(item("Hailo Runtime", h.ready ? "Bereit" : "Nicht bereit"),
    item("Gruppe", h.group_id), item("Modelle", (h.models || []).join(", ") || "—"),
    item("Aufträge in Warteschlange", h.pending), item("Vision", h.vision?.ready ? "Bereit" : h.vision?.enabled ? "Nicht bereit" : "Deaktiviert"),
    item("Piper", h.piper?.ready ? "Bereit" : h.piper?.enabled ? "Nicht bereit" : "Deaktiviert"),
    item("MQTT", h.mqtt_connected ? "Verbunden" : "Nicht verbunden"));
  $("health-raw").textContent = JSON.stringify(h, null, 2);
  renderLogs(data.logs || []);
}
async function refresh() {
  try {
    const key = $("key").value.trim();
    const response = await fetch("/v1/monitor", {cache: "no-store", headers: key ? {Authorization: `Bearer ${key}`} : {}});
    if (!response.ok) throw new Error(response.status === 401 ? "API-Schlüssel erforderlich oder ungültig" : `HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("state").textContent = "Nicht erreichbar"; $("state").className = "pill bad";
    $("updated").textContent = error.message;
  }
}
async function init() {
  try {
    const response = await fetch("/ui/config", {cache: "no-store"});
    if (response.ok) $("access").hidden = !(await response.json()).auth_required;
  } catch { /* Show the key field if the public configuration is unavailable. */ }
  await refresh();
}
$("apply").addEventListener("click", refresh);
$("refresh").addEventListener("click", refresh);
$("key").addEventListener("keydown", event => { if (event.key === "Enter") refresh(); });
void init(); setInterval(refresh, 5000);
