"use strict";
const $ = (id) => document.getElementById(id);
let config, history = [], chatBusy = false, audioBusy = false;
let audioBlob = null, audioName = "aufnahme.wav", audioURL = null, imageURL = null;
let recording = null, starting = false, stopping = false, pageHidden = false;

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
    throw new Error(response.status === 401 ? "API-Schlüssel fehlt oder ist ungültig." :
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
  $("image").disabled = chatBusy || $("chat-model").value === config?.llm_model;
  $("record").disabled = !config || !canRecord() || !!recording || starting || stopping || audioBusy;
  $("stop").disabled = !recording || stopping;
  $("audio-file").disabled = !!recording || starting || stopping || audioBusy;
  $("transcribe").disabled = !config || !audioBlob || !!recording || starting || stopping || audioBusy;
}
async function refreshStatus() {
  try {
    const response = await fetch("health", { cache: "no-store" });
    const data = await response.json();
    $("connection").textContent = response.ok && data.ready ? "Bereit" : "Nicht bereit";
    $("connection").className = `badge ${response.ok && data.ready ? "ready" : "error"}`;
    $("group").textContent = data.group_id || "—";
    $("pending").textContent = data.pending ?? "—";
    $("models").textContent = (data.models || []).join(" · ") || "—";
    $("mqtt").textContent = data.mqtt_connected ? "Verbunden" : "Nicht verbunden";
    $("raw-status").textContent = JSON.stringify(data, null, 2);
    note("status-note", `Stand: ${new Date().toLocaleTimeString("de-DE")} · Aktualisierung alle 5 Sekunden.`);
  } catch (error) {
    $("connection").textContent = "Nicht erreichbar";
    $("connection").className = "badge error";
    for (const id of ["group", "pending", "models", "mqtt"]) $(id).textContent = "—";
    note("status-note", `Service nicht erreichbar: ${error.message}`, true);
  }
}
function bubble(role, text, image) {
  $("conversation").querySelector(".empty")?.remove();
  const node = document.createElement("div"); node.className = `message ${role}`;
  const title = document.createElement("strong"); title.textContent = role === "user" ? "DU" : (config?.model_labels?.[$("chat-model").value] || "ASSISTANT");
  const content = document.createElement("span"); content.textContent = text;
  node.append(title, content);
  if (image) { const img = document.createElement("img"); img.src = image; img.alt = "Gesendetes Bild"; node.append(img); }
  $("conversation").append(node); node.scrollIntoView({ block: "nearest" });
  return node;
}
function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Bild konnte nicht gelesen werden."));
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
  chatBusy = true; controls(); note("chat-note", "Antwort wird erzeugt …");
  let userNode;
  try {
    const model = $("chat-model").value;
    const file = $("image").files[0];
    if (file && model === config.llm_model) throw new Error("Gemma verarbeitet Text. Für Bilder bitte Qwen2-VL wählen.");
    if (file && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) throw new Error("Bitte JPEG, PNG oder WebP auswählen.");
    if (file && file.size * 4 / 3 > config.max_body - 1024) throw new Error("Bild ist zu groß für den Service.");
    const image = file ? await readImage(file) : null;
    const content = image ? [{ type: "text", text: prompt }, { type: "image_url", image_url: { url: image } }] : prompt;
    const messages = [...history, { role: "user", content }];
    if (messages.length > 31) throw new Error("Chat ist voll. Bitte einen neuen Chat starten.");
    const images = messages.flatMap(m => Array.isArray(m.content) ? m.content : []).filter(p => p.type === "image_url");
    if (images.length > 4) throw new Error("Maximal vier Bilder pro Chat. Bitte einen neuen Chat starten.");
    const body = JSON.stringify({ model, messages, max_tokens: Number($("max-tokens").value) });
    if (new Blob([body]).size > config.max_body) throw new Error("Chat ist zu groß. Bitte einen neuen Chat starten.");
    userNode = bubble("user", prompt, image);
    const start = performance.now();
    const data = await request("v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body });
    const text = data.choices?.[0]?.message?.content;
    if (typeof text !== "string") throw new Error("Service lieferte keine Chat-Antwort.");
    history = [...messages, { role: "assistant", content: text }];
    bubble("assistant", text); $("prompt").value = ""; $("image").value = "";
    $("image-preview").hidden = true;
    if (imageURL) URL.revokeObjectURL(imageURL); imageURL = null;
    note("chat-note", `Antwort in ${((performance.now() - start) / 1000).toFixed(1)} s.`);
  } catch (error) { userNode?.remove(); note("chat-note", error.message, true); }
  finally { chatBusy = false; controls(); }
});
$("clear-chat").addEventListener("click", () => {
  history = []; $("conversation").replaceChildren(); note("chat-note", "Neuer Chat gestartet.");
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
    if (!rec.count) throw new Error("Keine Audiodaten aufgenommen. Bitte erneut versuchen.");
    setAudio(encodeWav(rec.chunks, rec.count, rec.context.sampleRate), "aufnahme.wav");
    note("whisper-note", `${(rec.count / rec.context.sampleRate).toFixed(1)} Sekunden aufgenommen. Jetzt anhören oder transkribieren.`);
  } catch (error) { note("whisper-note", error.message, true); }
  finally { stopping = false; controls(); }
}
$("record").addEventListener("click", async () => {
  if (starting || recording || !config) return;
  starting = true; controls(); note("whisper-note", "Mikrofon wird geöffnet …");
  const rec = { chunks: [], count: 0 };
  try {
    // Create/resume during the click gesture, including Safari on mobile.
    rec.context = new AudioContext(); await rec.context.resume();
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true }, video: false });
    await rec.context.resume();
    if (pageHidden || document.hidden) throw new Error("Aufnahme abgebrochen: Seite ist im Hintergrund.");
    await rec.context.audioWorklet.addModule("ui/recorder-worklet.js");
    if (pageHidden || document.hidden) throw new Error("Aufnahme abgebrochen: Seite ist im Hintergrund.");
    rec.source = rec.context.createMediaStreamSource(rec.stream);
    rec.node = new AudioWorkletNode(rec.context, "pcm-recorder");
    const maxSamples = Math.min(Math.floor(rec.context.sampleRate * config.recording_seconds), Math.floor((config.max_body - 2048) / 2));
    if (maxSamples < 128) throw new Error("Service-Größenlimit zu klein für eine Aufnahme.");
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
    note("whisper-note", "Aufnahme läuft …");
  } catch (error) {
    if (recording === rec) recording = null;
    await closeMicrophone(rec);
    const message = error.name === "NotAllowedError" ? "Mikrofonzugriff abgelehnt. Bitte in den Browser-Einstellungen erlauben." : error.message;
    note("whisper-note", message, true);
  } finally { starting = false; controls(); }
});
$("stop").addEventListener("click", () => void stopRecording());
$("audio-file").addEventListener("change", () => {
  const file = $("audio-file").files[0]; if (!file || !config) return;
  if (file.size > config.max_body - 2048) { note("whisper-note", "Audiodatei ist zu groß.", true); $("audio-file").value = ""; return; }
  setAudio(file, file.name); note("whisper-note", "Audiodatei bereit. Dauer und Format werden vom Service geprüft.");
});
$("transcribe").addEventListener("click", async () => {
  if (audioBusy || !audioBlob || !config) return;
  audioBusy = true; controls(); note("whisper-note", "Sprache wird transkribiert …");
  try {
    const form = new FormData(); form.append("file", audioBlob, audioName); form.append("model", config.whisper_model);
    if ($("language").value) form.append("language", $("language").value);
    const start = performance.now();
    const data = await request("v1/audio/transcriptions", { method: "POST", body: form });
    $("transcript").value = data.text;
    note("whisper-note", `Transkription in ${((performance.now() - start) / 1000).toFixed(1)} s abgeschlossen.`);
  } catch (error) { note("whisper-note", error.message, true); }
  finally { audioBusy = false; controls(); }
});
$("check-key").addEventListener("click", async () => {
  try { await request("v1/models"); note("auth-note", "API-Zugang erfolgreich geprüft. Schlüssel bleibt nur im Speicher dieser Seite."); }
  catch (error) { note("auth-note", error.message, true); }
});
$("refresh").addEventListener("click", refreshStatus);
document.addEventListener("visibilitychange", () => { if (document.hidden) void stopRecording(); });
window.addEventListener("pagehide", () => { pageHidden = true; void stopRecording(); });
window.addEventListener("pageshow", () => { pageHidden = false; controls(); });
async function init() {
  await refreshStatus(); setInterval(refreshStatus, 5000);
  try {
    config = await request("ui/config");
    for (const model of config.chat_models || [config.vlm_model]) {
      if (model === config.vlm_model) continue;
      $("chat-model").add(new Option(`${model} · LiteRT-LM / CPU`, model));
    }
    config.model_labels = { [config.vlm_model]: "QWEN2-VL", [config.llm_model]: "GEMMA 4 E2B" };
    $("auth-section").hidden = !config.auth_required;
    $("record-progress").max = config.recording_seconds;
    if ([...$("language").options].some(o => o.value === config.language)) $("language").value = config.language;
    else { const option = new Option(config.language, config.language); $("language").add(option); $("language").value = config.language; }
    note("mic-hint", canRecord() ? `Maximal ${config.recording_seconds} Sekunden. Aufnahme als Mono-WAV; kein Zusatzprogramm nötig.` :
      "Mikrofonaufnahme braucht HTTPS oder localhost und einen Browser mit AudioWorklet. Bei HTTP über eine LAN-IP bitte Audiodatei hochladen oder SSH-Tunnel verwenden.", !canRecord());
    note("audio-name", `WAV / FLAC / OGG · maximal ${config.max_audio_seconds} s und ${(config.max_body / 1024 / 1024).toFixed(1)} MB.`);
    $("chat-model").addEventListener("change", () => {
      if ($("chat-model").value === config.llm_model && $("image").files.length) {
        $("image").value = ""; $("image-preview").hidden = true;
      }
      note("chat-note", $("chat-model").value === config.llm_model
        ? "Gemma 4 E2B läuft über LiteRT-LM auf der CPU. Bildanalyse erfolgt mit Qwen2-VL."
        : "Qwen2-VL verarbeitet Text und Bilder über Hailo.");
      controls();
    });
  } catch (error) { note("status-note", `Seitenkonfiguration konnte nicht geladen werden: ${error.message}`, true); }
  controls();
}
void init();
