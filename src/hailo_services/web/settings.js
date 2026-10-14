"use strict";
const $ = id => document.getElementById(id);
let version = null, current = [], restartAvailable = false;
try { $("key").value = sessionStorage.getItem("hailo-api-key") || ""; } catch {}
$("key").addEventListener("input", () => { try { const key = $("key").value.trim(); if (key) sessionStorage.setItem("hailo-api-key", key); else sessionStorage.removeItem("hailo-api-key"); } catch {} });
const auth = () => { const key = $("key").value.trim(); return key ? {Authorization: `Bearer ${key}`} : {}; };
const say = (value, bad = false) => { $("notice").textContent = value; $("notice").className = bad ? "bad" : "good"; };
async function request(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options, headers: {...auth(), ...options.headers}});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || (response.status === 401 ? "API-Schlüssel erforderlich oder ungültig" : `HTTP ${response.status}`));
  return body;
}
function render(data) {
  version = data.version; current = data.fields; restartAvailable = data.restart_available;
  $("config-path").textContent = `Datei: ${data.path}`;
  const groups = new Map();
  for (const field of current) {
    const [category] = field.path.split(".");
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(field);
  }
  const names = {settings: "Allgemein, Integrationen & Zugang", models: "Modelle", vision: "Vision / YOLO"};
  $("fields").replaceChildren(...Array.from(groups, ([category, fields]) => {
    const section = document.createElement("section"), title = document.createElement("h2"), grid = document.createElement("div");
    title.textContent = names[category] || category; grid.className = "setting-grid";
    for (const field of fields) {
      const label = document.createElement("label"), input = document.createElement(field.type === "bool" ? "select" : "input");
      label.textContent = field.path; label.append(document.createElement("br"));
      input.dataset.path = field.path; input.dataset.type = field.type;
      if (field.type === "bool") { input.append(new Option("Aktiviert", "true"), new Option("Deaktiviert", "false")); input.value = String(field.value); }
      else { input.type = field.secret_set || ["settings.api_key", "settings.mqtt_password"].includes(field.path) ? "password" : field.type === "int" || field.type === "float" ? "number" : "text"; if (field.type === "float") input.step = "any"; if (field.type === "int") input.step = "1"; input.value = field.value ?? ""; if (["settings.api_key", "settings.mqtt_password"].includes(field.path)) input.placeholder = field.secret_set ? "Gesetzt · leer lassen zum Beibehalten" : "Leer"; }
      label.append(input);
      if (["settings.api_key", "settings.mqtt_password"].includes(field.path) && field.secret_set) { const clear = document.createElement("label"), checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.dataset.clear = field.path; clear.append(checkbox, document.createTextNode(" Gespeicherten Wert löschen")); label.append(clear); }
      grid.append(label);
    }
    section.append(title, grid); return section;
  }));
  $("save").disabled = false;
}
async function load() {
  try { render(await request("/v1/settings")); say("Konfiguration geladen. Änderungen benötigen einen Neustart."); }
  catch (error) { say(error.message, true); }
}
$("form").addEventListener("submit", async event => {
  event.preventDefault();
  const changes = {};
  for (const field of current) {
    const input = Array.from($("fields").querySelectorAll("[data-path]")).find(node => node.dataset.path === field.path);
    let value = input.value;
    if (field.type === "bool") value = value === "true";
    else if (field.type === "int" || field.type === "float") value = Number(value);
    if (field.secret_set && $("fields").querySelector(`[data-clear="${field.path}"]`)?.checked) { changes[field.path] = null; continue; }
    if (field.secret_set && value === "") continue;
    if (value !== field.value && !(field.value === null && value === "")) changes[field.path] = value;
  }
  if (!Object.keys(changes).length) { say("Keine Änderungen vorhanden."); return; }
  $("save").disabled = true;
  try {
    render(await request("/v1/settings", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({version, changes})}));
    say("Gespeichert. Neustart erforderlich, damit die Änderungen wirksam werden.");
    $("restart").hidden = !restartAvailable;
  } catch (error) { say(error.message, true); $("save").disabled = false; }
});
$("restart").addEventListener("click", async () => {
  try { await request("/v1/settings/restart", {method: "POST"}); say("Neustart eingeleitet. Seite nach einigen Sekunden neu laden."); $("restart").hidden = true; }
  catch (error) { say(error.message, true); }
});
$("apply").addEventListener("click", load);
(async () => { try { const response = await fetch("/ui/config", {cache: "no-store"}); if (response.ok) $("access").hidden = !(await response.json()).auth_required; } catch {} await load(); })();
