"use strict";
const $ = id => document.getElementById(id), tr = (key, values) => window.HailoUI.t(key, values);
let version = null, current = [], restartAvailable = false, configPath = "", notice = null;
try { $("key").value = sessionStorage.getItem("hailo-api-key") || ""; } catch {}
$("key").addEventListener("input", () => { try { const key = $("key").value.trim(); if (key) sessionStorage.setItem("hailo-api-key", key); else sessionStorage.removeItem("hailo-api-key"); } catch {} });
const auth = () => { const key = $("key").value.trim(); return key ? {Authorization: `Bearer ${key}`} : {}; };
function say(key, bad = false, values = {}) {
  notice = {key, bad, values}; $("notice").textContent = tr(key, values); $("notice").className = bad ? "bad" : "good";
}
async function request(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options, headers: {...auth(), ...options.headers}});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || (response.status === 401 ? tr("ui_0") : `HTTP ${response.status}`));
  return body;
}
const groups = [
  {id: "http", matches: path => ["settings.host", "settings.port", "settings.api_key", "settings.api_no_auth_networks"].includes(path)},
  {id: "mcp", matches: path => path.startsWith("settings.mcp_")},
  {id: "wyoming", matches: path => path.startsWith("settings.wyoming_")},
  {id: "mqtt", matches: path => path.startsWith("settings.mqtt_")},
  {id: "zmq", matches: path => path.startsWith("vision.zmq.") || path.startsWith("settings.vision_zmq_")},
  {id: "vlm", matches: path => path.startsWith("models.vlm.") || path.startsWith("settings.vlm_")},
  {id: "hailo_llm", matches: path => path.startsWith("models.hailo_llm.") || path.startsWith("settings.hailo_llm_")},
  {id: "whisper", matches: path => path.startsWith("models.whisper.") || path.startsWith("settings.whisper_") || ["settings.language", "settings.max_audio_seconds"].includes(path)},
  {id: "vision", matches: path => path.startsWith("vision.") || path.startsWith("models.vision.") || path.startsWith("settings.vision_")},
  {id: "minilm", matches: path => path.startsWith("models.minilm.") || path.startsWith("settings.minilm_")},
  {id: "gemma", matches: path => path.startsWith("models.gemma.") || path.startsWith("settings.litert_")},
  {id: "piper", matches: path => path.startsWith("settings.piper_")},
  {id: "ha", matches: path => path.startsWith("settings.ha_assist_") || path === "settings.ha_wait_messages"},
  {id: "frigate", matches: path => path.startsWith("settings.frigate_assist_")},
  {id: "catalog", matches: path => path.startsWith("settings.model_")},
  {id: "runtime", matches: () => true},
];
const fieldLabels = new Set("host port api_key api_no_auth_networks no_auth_networks hosts enabled model path release max_input_tokens language service_language max_audio_seconds zmq_enabled endpoint confidence iou_threshold max_detections queue_size scheduler_priority voice voice_dir max_input_chars text_model vision_model fuzzy_enabled fuzzy_threshold fuzzy_margin sentence_fuzzy_enabled sentence_threshold verify_attempts verify_delay wait_messages max_events text_chars vision_chars vision_max_tokens away_profiles max_num_tokens model_path hef_path model_store model_release model_catalog request_timeout max_body debug_log username password prefix tls".split(" "));
function fieldKey(field) {
  let leaf = field.path.split(".").pop().replace(/^(ha_assist_|frigate_assist_|hailo_llm_|wyoming_|whisper_|piper_|litert_|mqtt_|vlm_|vision_|mcp_|minilm_)/, "");
  if (leaf === "zmq_endpoint") leaf = "endpoint";
  if (field.options && !["text_model", "vision_model"].includes(leaf)) leaf = "model";
  if (field.path === "settings.ha_wait_messages") leaf = "wait_messages";
  if (["model_store", "model_release", "model_catalog"].includes(field.path.split(".").pop())) leaf = field.path.split(".").pop();
  return fieldLabels.has(leaf) ? `settings.field.${leaf}` : null;
}
function translated(tag, key, text = "") {
  const node = document.createElement(tag); node.dataset.i18n = key; node.textContent = text || tr(key); return node;
}
function fieldRow(field, index) {
  const row = document.createElement("div"), label = document.createElement("label"), controls = document.createElement("div");
  row.className = "setting-row"; controls.className = "setting-control";
  const key = fieldKey(field), title = key ? translated("span", key) : document.createElement("span");
  if (!key) title.textContent = field.path.split(".").pop().replaceAll("_", " ");
  const path = document.createElement("code"); path.textContent = field.path;
  label.append(title, path); label.htmlFor = `setting-${index}`;
  const input = document.createElement(field.options || field.type === "bool" ? "select" : "input");
  input.id = label.htmlFor; input.dataset.path = field.path; input.dataset.type = field.type;
  if (field.options) {
    if (field.allow_empty) { const option = new Option(tr("settings.default_model"), ""); option.dataset.i18n = "settings.default_model"; input.append(option); }
    for (const model of field.options) input.append(new Option(model, model));
    const value = field.value ?? "";
    if (![...input.options].some(option => option.value === value)) input.append(new Option(value || "—", value));
    input.value = value;
  } else if (field.type === "bool") {
    for (const [value, key] of [["true", "common.on"], ["false", "common.off"]]) {
      const option = new Option(tr(key), value); option.dataset.i18n = key; input.append(option);
    }
    input.value = String(field.value);
  } else {
    const secret = ["settings.api_key", "settings.mqtt_password"].includes(field.path);
    input.type = secret ? "password" : ["int", "float"].includes(field.type) ? "number" : "text";
    if (input.type === "number") { input.step = field.type === "float" ? "any" : "1"; input.required = true; }
    input.value = field.value ?? "";
    if (secret) { input.autocomplete = "new-password"; input.dataset.i18nPlaceholder = field.secret_set ? "settings.secret_keep" : "settings.secret_empty"; input.placeholder = tr(input.dataset.i18nPlaceholder); }
  }
  controls.append(input);
  if (["settings.api_key", "settings.mqtt_password"].includes(field.path) && field.secret_set) {
    const clear = document.createElement("label"), checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.dataset.clear = field.path;
    clear.className = "secret-clear"; clear.append(checkbox, translated("span", "settings.secret_clear")); controls.append(clear);
  }
  row.append(label, controls); return row;
}
function render(data) {
  version = data.version; current = data.fields; restartAvailable = data.restart_available; configPath = data.path;
  $("config-path").textContent = tr("settings.file", {path: configPath});
  const grouped = new Map();
  for (const field of current) {
    const group = groups.find(group => group.matches(field.path));
    if (!grouped.has(group.id)) grouped.set(group.id, []);
    grouped.get(group.id).push(field);
  }
  const sections = [], links = [];
  for (const group of groups) {
    const fields = grouped.get(group.id); if (!fields) continue;
    const section = document.createElement("section"), details = document.createElement("details"), summary = document.createElement("summary"), list = document.createElement("div");
    section.className = "setting-group"; section.id = `settings-${group.id}`; details.open = true; list.className = "setting-list";
    summary.append(translated("span", `settings.group.${group.id}`), translated("small", "settings.needs_restart"));
    fields.forEach(field => list.append(fieldRow(field, current.indexOf(field))));
    details.append(summary, list); section.append(details); sections.push(section);
    const link = translated("a", `settings.group.${group.id}`); link.href = `#${section.id}`; links.push(link);
  }
  $("fields").replaceChildren(...sections); $("settings-groups").replaceChildren(...links);
  $("catalog-note").hidden = !data.catalog_error;
  $("catalog-note").textContent = data.catalog_error ? tr("settings.catalog_error", {error: data.catalog_error}) : "";
  $("save").disabled = false;
}
async function load() {
  try { render(await request("/v1/settings")); say("settings.loaded"); }
  catch (error) { say(error.message, true); }
}
$("form").addEventListener("submit", async event => {
  event.preventDefault();
  const changes = {}, inputs = new Map(Array.from($("fields").querySelectorAll("[data-path]")).map(input => [input.dataset.path, input]));
  for (const field of current) {
    const input = inputs.get(field.path); let value = input.value;
    if (field.type === "bool") value = value === "true";
    else if (field.type === "int" || field.type === "float") value = Number(value);
    if (field.secret_set && $("fields").querySelector(`[data-clear="${field.path}"]`)?.checked) { changes[field.path] = null; continue; }
    if (field.secret_set && value === "") continue;
    if (value !== field.value && !(field.value === null && value === "")) changes[field.path] = value;
  }
  if (!Object.keys(changes).length) { say("settings.unchanged"); return; }
  $("save").disabled = true;
  try {
    render(await request("/v1/settings", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({version, changes})}));
    say("settings.saved"); $("restart").hidden = !restartAvailable;
  } catch (error) { say(error.message, true); $("save").disabled = false; }
});
$("restart").addEventListener("click", async () => {
  try { await request("/v1/settings/restart", {method: "POST"}); say("settings.restarting"); $("restart").hidden = true; }
  catch (error) { say(error.message, true); }
});
$("ui-language").addEventListener("change", async () => {
  try { await window.HailoUI.setChoice($("ui-language").value); }
  catch (error) { say(error.message, true); }
});
window.HailoUI.subscribe(() => {
  $("config-path").textContent = configPath ? tr("settings.file", {path: configPath}) : "";
  if (notice) say(notice.key, notice.bad, notice.values);
});
$("apply").addEventListener("click", load);
(async () => {
  let config;
  try { const response = await fetch("/ui/config", {cache: "no-store"}); if (response.ok) { config = await response.json(); $("access").hidden = !config.auth_required; } } catch {}
  try { await window.HailoUI.load(window.HailoUI.savedChoice(), config); } catch {}
  await load();
})();
