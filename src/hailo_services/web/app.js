"use strict";
const $ = (id) => document.getElementById(id);
let config, history = [], chatBusy = false, audioBusy = false;
let audioBlob = null, audioName = "aufnahme.wav", audioURL = null, imageURL = null;
let recording = null, starting = false, stopping = false, pageHidden = false;

let uiLanguage = "de", uiStrings = {}, uiChoice = "auto", languageNames = {};
function tr(key, values = {}) {
  return (uiStrings[key] || key).replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ""));
}
function savedChoice() {
  try { return localStorage.getItem("hailo-ui-language") || "auto"; } catch { return "auto"; }
}
function chooseLanguage(choice) {
  const supported = config?.ui_languages || ["de", "en", "ru"];
  if (supported.includes(choice)) return choice;
  for (const value of navigator.languages || [navigator.language || ""]) {
    const code = value.toLowerCase().split("-")[0];
    if (supported.includes(code)) return code;
  }
  return config?.service_language || "de";
}
async function loadLanguage(choice) {
  const language = chooseLanguage(choice);
  const response = await fetch(`ui/locales/${language}.json`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  uiLanguage = language; uiStrings = data.ui || {}; uiChoice = choice;
  languageNames = data.language_names || {};
  const selectedSTT = $("language").value;
  $("language").replaceChildren(new Option(tr("ui_71"), ""),
    ...(config?.stt_languages || ["de", "en", "ru"]).map(code => new Option(languageNames[code] || code, code)));
  $("language").value = selectedSTT;
  document.documentElement.lang = language;
  for (const element of document.querySelectorAll("[data-i18n]")) element.textContent = tr(element.dataset.i18n);
  for (const element of document.querySelectorAll("[data-i18n-placeholder]")) element.placeholder = tr(element.dataset.i18nPlaceholder);
  for (const element of document.querySelectorAll("[data-i18n-alt]")) element.alt = tr(element.dataset.i18nAlt);
  for (const element of document.querySelectorAll("[data-i18n-aria]")) element.setAttribute("aria-label", tr(element.dataset.i18nAria));
}
$("ui-language").addEventListener("change", async () => {
  const choice = $("ui-language").value;
  try { await loadLanguage(choice); updateChatMode(); localStorage.setItem("hailo-ui-language", choice); await refreshStatus(); }
  catch (error) { note("status-note", error.message, true); }
});

function note(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle("error", error);
}
function headers() {
  const key = $("api-key").value.trim();
  return key ? { Authorization: `Bearer ${key}` } : {};
}
async function request(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...headers(), ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.error || data.detail;
    throw new Error(response.status === 401 ? tr("ui_0") :
      (typeof detail === "string" ? detail : JSON.stringify(detail || `HTTP ${response.status}`)));
  }
  return data;
}
function canRecord() {
  return window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.AudioContext && window.AudioWorkletNode;
}
function controls() {
  $("send-chat").disabled = chatBusy || !config;
  $("clear-chat").disabled = chatBusy;
  $("prompt").disabled = chatBusy;
  $("chat-model").disabled = chatBusy;
  $("chat-mode").disabled = chatBusy || $("chat-model").value === config?.llm_model;
  $("image").disabled = chatBusy || $("chat-mode").value === "text" || $("chat-model").value === config?.llm_model;
  $("record").disabled = !config || !canRecord() || !!recording || starting || stopping || audioBusy;
  $("stop").disabled = !recording || stopping;
  $("audio-file").disabled = !!recording || starting || stopping || audioBusy;
  $("transcribe").disabled = !config || !audioBlob || !!recording || starting || stopping || audioBusy;
}
async function refreshStatus() {
  try {
    const response = await fetch("health", { cache: "no-store" });
    const data = await response.json();
    $("connection").textContent = response.ok && data.ready ? tr("ui_1") : tr("ui_2");
    $("connection").className = `badge ${response.ok && data.ready ? "ready" : "error"}`;
    $("group").textContent = data.group_id || "—";
    $("pending").textContent = data.pending ?? "—";
    $("models").textContent = (data.models || []).join(" · ") || "—";
    $("mqtt").textContent = data.mqtt_connected ? tr("ui_3") : tr("ui_4");
    $("raw-status").textContent = JSON.stringify(data, null, 2);
    note("status-note", tr("ui_5", {v0: new Date().toLocaleTimeString(uiLanguage)}));
  } catch (error) {
    $("connection").textContent = tr("ui_6");
    $("connection").className = "badge error";
    for (const id of ["group", "pending", "models", "mqtt"]) $(id).textContent = "—";
    note("status-note", tr("ui_7", {v0: error.message}), true);
  }
}
function bubble(role, text, image) {
  $("conversation").querySelector(".empty")?.remove();
  const node = document.createElement("div"); node.className = `message ${role}`;
  const title = document.createElement("strong"); title.textContent = role === "user" ? tr("ui_8") : (config?.model_labels?.[$("chat-model").value] || "ASSISTANT");
  const content = document.createElement("span"); content.textContent = text;
  node.append(title, content);
  if (image) { const img = document.createElement("img"); img.src = image; img.alt = tr("ui_9"); node.append(img); }
  $("conversation").append(node); node.scrollIntoView({ block: "nearest" });
  return node;
}
function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(tr("ui_10")));
    reader.readAsDataURL(file);
  });
}
$("image").addEventListener("change", () => {
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = null;
  const file = $("image").files[0];
  if (file) { imageURL = URL.createObjectURL(file); $("image-preview").src = imageURL; }
  $("image-preview").hidden = !file;
});
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (chatBusy || !config) return;
  const prompt = $("prompt").value.trim(); if (!prompt) return;
  chatBusy = true; controls(); note("chat-note", tr("ui_11"));
  let userNode;
  try {
    const model = $("chat-model").value;
    const file = $("chat-mode").value === "text" ? null : $("image").files[0];
    if (file && model === config.llm_model) throw new Error(tr("ui_12"));
    if (file && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) throw new Error(tr("ui_13"));
    if (file && file.size * 4 / 3 > config.max_body - 1024) throw new Error(tr("ui_14"));
    const image = file ? await readImage(file) : null;
    const content = image ? [{ type: "text", text: prompt }, { type: "image_url", image_url: { url: image } }] : prompt;
    const messages = [...history, { role: "user", content }];
    if (messages.length > 31) throw new Error(tr("ui_15"));
    const images = messages.flatMap(m => Array.isArray(m.content) ? m.content : []).filter(p => p.type === "image_url");
    if (images.length > (config.vlm_max_images || 4)) throw new Error(tr("ui.image_limit", {limit: config.vlm_max_images || 4}));
    const body = JSON.stringify({ model, messages, language: uiLanguage, max_tokens: Number($("max-tokens").value) });
    if (new Blob([body]).size > config.max_body) throw new Error(tr("ui_17"));
    userNode = bubble("user", prompt, image);
    const start = performance.now();
    const data = await request("v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body });
    const text = data.choices?.[0]?.message?.content;
    if (typeof text !== "string") throw new Error(tr("ui_18"));
    history = [...messages, { role: "assistant", content: text }];
    bubble("assistant", text); $("prompt").value = ""; $("image").value = "";
    $("image-preview").hidden = true;
    if (imageURL) URL.revokeObjectURL(imageURL); imageURL = null;
    note("chat-note", tr("ui_19", {v0: ((performance.now() - start) / 1000).toFixed(1)}));
  } catch (error) { userNode?.remove(); note("chat-note", error.message, true); }
  finally { chatBusy = false; controls(); }
});
$("clear-chat").addEventListener("click", () => {
  history = []; $("conversation").replaceChildren(); note("chat-note", tr("ui_20"));
});
function setAudio(blob, name) {
  if (audioURL) URL.revokeObjectURL(audioURL);
  audioBlob = blob; audioName = name;
  audioURL = URL.createObjectURL(blob); $("audio-preview").src = audioURL;
  $("audio-preview").hidden = false;
  $("audio-name").textContent = `${name} · ${(blob.size / 1024).toFixed(0)} KB`;
  $("transcript").value = ""; controls();
}
function encodeWav(chunks, count, rate) {
  const buffer = new ArrayBuffer(44 + count * 2), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + count * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, count * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true); offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}
function closeMicrophone(rec) {
  clearInterval(rec.timer);
  rec.stream?.getTracks().forEach(track => track.stop());
  rec.source?.disconnect(); rec.node?.disconnect();
  if (rec.node) rec.node.port.onmessage = null;
  return rec.context?.close().catch(() => {});
}
async function stopRecording() {
  if (!recording || stopping) return;
  stopping = true; const rec = recording; recording = null; controls();
  await closeMicrophone(rec);
  try {
    if (!rec.count) throw new Error(tr("ui_21"));
    setAudio(encodeWav(rec.chunks, rec.count, rec.context.sampleRate), "aufnahme.wav");
    note("whisper-note", tr("ui_22", {v0: (rec.count / rec.context.sampleRate).toFixed(1)}));
  } catch (error) { note("whisper-note", error.message, true); }
  finally { stopping = false; controls(); }
}
$("record").addEventListener("click", async () => {
  if (starting || recording || !config) return;
  starting = true; controls(); note("whisper-note", tr("ui_23"));
  const rec = { chunks: [], count: 0 };
  try {
    // Create/resume during the click gesture, including Safari on mobile.
    rec.context = new AudioContext(); await rec.context.resume();
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true }, video: false });
    await rec.context.resume();
    if (pageHidden || document.hidden) throw new Error(tr("ui_24"));
    await rec.context.audioWorklet.addModule("ui/recorder-worklet.js");
    if (pageHidden || document.hidden) throw new Error(tr("ui_24"));
    rec.source = rec.context.createMediaStreamSource(rec.stream);
    rec.node = new AudioWorkletNode(rec.context, "pcm-recorder");
    const maxSamples = Math.min(Math.floor(rec.context.sampleRate * config.recording_seconds), Math.floor((config.max_body - 2048) / 2));
    if (maxSamples < 128) throw new Error(tr("ui_25"));
    rec.node.port.onmessage = ({ data }) => {
      if (recording !== rec) return;
      const chunk = data.subarray(0, maxSamples - rec.count);
      if (chunk.length) { rec.chunks.push(chunk); rec.count += chunk.length; }
      const seconds = rec.count / rec.context.sampleRate;
      $("record-time").textContent = `00:${String(Math.floor(seconds)).padStart(2, "0")}`;
      $("record-progress").value = seconds;
      if (rec.count >= maxSamples) void stopRecording();
    };
    recording = rec;
    rec.source.connect(rec.node); rec.node.connect(rec.context.destination);
    $("record-time").textContent = "00:00"; $("record-progress").value = 0;
    rec.timer = setInterval(() => {
      if (performance.now() - rec.started > config.recording_seconds * 1000) void stopRecording();
    }, 100);
    rec.started = performance.now();
    rec.stream.getTracks().forEach(track => track.addEventListener("ended", () => void stopRecording()));
    note("whisper-note", tr("ui_26"));
  } catch (error) {
    if (recording === rec) recording = null;
    await closeMicrophone(rec);
    const message = error.name === "NotAllowedError" ? tr("ui_27") : error.message;
    note("whisper-note", message, true);
  } finally { starting = false; controls(); }
});
$("stop").addEventListener("click", () => void stopRecording());
$("audio-file").addEventListener("change", () => {
  const file = $("audio-file").files[0]; if (!file || !config) return;
  if (file.size > config.max_body - 2048) { note("whisper-note", tr("ui_28"), true); $("audio-file").value = ""; return; }
  setAudio(file, file.name); note("whisper-note", tr("ui_29"));
});
$("transcribe").addEventListener("click", async () => {
  if (audioBusy || !audioBlob || !config) return;
  audioBusy = true; controls(); note("whisper-note", tr("ui_30"));
  try {
    const form = new FormData(); form.append("file", audioBlob, audioName); form.append("model", config.whisper_model);
    if ($("language").value) form.append("language", $("language").value);
    const start = performance.now();
    const data = await request("v1/audio/transcriptions", { method: "POST", body: form });
    $("transcript").value = data.text;
    note("whisper-note", tr("ui_31", {v0: ((performance.now() - start) / 1000).toFixed(1)}));
  } catch (error) { note("whisper-note", error.message, true); }
  finally { audioBusy = false; controls(); }
});
$("check-key").addEventListener("click", async () => {
  try { await request("v1/models"); note("auth-note", tr("ui_32")); }
  catch (error) { note("auth-note", error.message, true); }
});
$("refresh").addEventListener("click", refreshStatus);
document.addEventListener("visibilitychange", () => { if (document.hidden) void stopRecording(); });
window.addEventListener("pagehide", () => { pageHidden = true; void stopRecording(); });
window.addEventListener("pageshow", () => { pageHidden = false; controls(); });
function updateChatMode() {
  if ($("chat-mode").value === "text" || $("chat-model").value === config?.llm_model) {
    $("image").value = ""; $("image-preview").hidden = true;
    if (imageURL) URL.revokeObjectURL(imageURL); imageURL = null;
  }
  const limits = config?.model_limits?.[$("chat-model").value];
  note("input-limit", limits ? tr("ui.input_limit", {limit: limits.max_input_tokens, context: limits.context_length}) : "");
}
async function init() {

  try {
    config = await request("ui/config");
    $("ui-language").value = savedChoice();
    await loadLanguage($("ui-language").value);
    await refreshStatus(); setInterval(refreshStatus, 5000);
    $("chat-model").replaceChildren();
    for (const model of config.chat_models || [config.vlm_model]) {
      $("chat-model").add(new Option(`${model} · ${model === config.vlm_model ? "Hailo" : "LiteRT-LM / CPU"}`, model));
    }
    $("chat-model").value = config.default_text_model || config.vlm_model;
    config.model_labels = { [config.vlm_model]: config.vlm_model, [config.llm_model]: "GEMMA 4 E2B" };
    $("chat-mode").value = "text";
    updateChatMode();
    $("auth-section").hidden = !config.auth_required;
    $("record-progress").max = config.recording_seconds;
    if ([...$("language").options].some(o => o.value === config.language)) $("language").value = config.language;
    else { const option = new Option(config.language, config.language); $("language").add(option); $("language").value = config.language; }
    note("mic-hint", canRecord() ? tr("ui_33", {v0: config.recording_seconds}) :
      tr("ui_extra_0"), !canRecord());
    note("audio-name", tr("ui_extra_1", {v0: config.max_audio_seconds, v1: (config.max_body / 1024 / 1024).toFixed(1)}));
    $("chat-model").addEventListener("change", () => {
      history = []; $("conversation").replaceChildren();
      if ($("chat-model").value === config.llm_model) $("chat-mode").value = "text";
      updateChatMode(); controls();
    });
    $("chat-mode").addEventListener("change", () => {
      history = []; $("conversation").replaceChildren();
      updateChatMode(); controls();
    });
  } catch (error) { note("status-note", tr("ui_extra_4", {v0: error.message}), true); }
  controls();
}
void init();
