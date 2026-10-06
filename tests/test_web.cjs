// Run with: node --test tests/test_web.cjs (Node >=18, no npm dependencies).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../src/hailo_services/web');

function page(overrides = {}) {
  class Element {
    constructor() { this.value = ''; this.children = []; this.events = {}; this.disabled = false; this.hidden = false; this.files = []; this.options = []; this.classList = { toggle() {} }; }
    addEventListener(type, fn) { this.events[type] = fn; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    querySelector() { return null; }
    scrollIntoView() {}
    remove() { this.removed = true; }
    add(option) { this.options.push(option); }
  }
  const elements = Object.fromEntries([...fs.readFileSync(path.join(web, 'index.html'), 'utf8').matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  elements.language.options = [{ value: 'de' }, { value: '' }]; elements.language.value = 'de'; elements['max-tokens'].value = '256';
  const tracks = [{ stopped: false, stop() { this.stopped = true; }, addEventListener() {} }];
  const contexts = [], nodes = [], calls = [];
  class AudioContext {
    constructor() { this.sampleRate = 48000; this.audioWorklet = { addModule: async () => {} }; contexts.push(this); }
    async resume() {}
    async close() { this.closed = true; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  }
  class AudioWorkletNode {
    constructor() { this.port = {}; nodes.push(this); }
    connect() {} disconnect() { this.disconnected = true; }
  }
  const config = { auth_required: true, vlm_model: 'Qwen2-VL-2B-Instruct', whisper_model: 'whisper-base', max_body: 16 * 1024 * 1024, recording_seconds: 1, max_audio_seconds: 120, language: 'de', ...overrides };
  const context = vm.createContext({
    localStorage: { getItem() { return null; }, setItem() {} },
    document: { documentElement: {}, querySelectorAll() { return []; }, hidden: false, getElementById: id => elements[id], createElement: () => new Element(), addEventListener() {} },
    window: { isSecureContext: true, AudioContext, AudioWorkletNode, addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => tracks }) } },
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    AudioContext, AudioWorkletNode, Blob, URL, FormData, performance, console, setInterval: () => 1, clearInterval() {},
    fetch: async (url, options = {}) => {
      calls.push({ url, options });
      return { ok: true, json: async () => url.startsWith('ui/locales/') ? JSON.parse(fs.readFileSync(path.join(web, '../locales', url.split('/').pop()), 'utf8')) : url === 'ui/config' ? config : url === 'health' ? { ready: true, group_id: 'SHARED', models: ['qwen', 'whisper'], pending: 0 } : url === 'v1/audio/transcriptions' ? { text: 'Hallo Welt' } : { choices: [{ message: { content: 'Antwort' } }] } };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(web, 'app.js'), 'utf8'), context);
  return { context, elements, tracks, contexts, nodes, calls, ready: () => new Promise(resolve => setImmediate(resolve)), run: code => vm.runInContext(code, context) };
}

test('PCM WAV header, rate, length, signed samples and clipping', async () => {
  const p = page(); await p.ready();
  const blob = p.run('encodeWav([new Float32Array([-2,-0.5,0,0.5,2])], 5, 48000)');
  const data = Buffer.from(await blob.arrayBuffer());
  assert.equal(data.toString('ascii', 0, 4), 'RIFF');
  assert.equal(data.toString('ascii', 8, 12), 'WAVE');
  assert.equal(data.readUInt32LE(4), data.length - 8);
  assert.equal(data.readUInt16LE(22), 1);
  assert.equal(data.readUInt32LE(24), 48000);
  assert.equal(data.readUInt16LE(34), 16);
  assert.equal(data.readUInt32LE(40), 10);
  assert.deepEqual([44,46,48,50,52].map(i => data.readInt16LE(i)), [-32768,-16384,0,16384,32767]);
});

test('AudioWorklet averages channels and leaves playback output silent', () => {
  let Processor; const sent = [];
  const context = vm.createContext({ AudioWorkletProcessor: class { constructor() { this.port = { postMessage: data => sent.push(data) }; } }, registerProcessor: (_, cls) => { Processor = cls; }, Float32Array });
  vm.runInContext(fs.readFileSync(path.join(web, 'recorder-worklet.js'), 'utf8'), context);
  const output = new Float32Array(2);
  assert.equal(new Processor().process([[new Float32Array([1,-1]),new Float32Array([-1,0.5])]], [[output]]), true);
  assert.deepEqual(Array.from(sent[0]), [0,-0.25]);
  assert.deepEqual(Array.from(output), [0,0]);
});

test('Recording stops at duration limit, releases microphone and uploads a WAV with Bearer auth', async () => {
  const p = page(); await p.ready(); p.elements['api-key'].value = 'test-key';
  await p.elements.record.events.click();
  assert.equal(p.elements.stop.disabled, false);
  p.nodes[0].port.onmessage({ data: new Float32Array(48128) });
  await p.ready();
  assert.equal(p.tracks[0].stopped, true); assert.equal(p.contexts[0].closed, true);
  assert.equal(p.elements.transcribe.disabled, false);
  const blob = p.run('audioBlob'); const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.readUInt32LE(40), 48000 * 2);
  await p.elements.transcribe.events.click();
  const upload = p.calls.find(c => c.url === 'v1/audio/transcriptions');
  assert.equal(upload.options.headers.Authorization, 'Bearer test-key');
  assert.equal(upload.options.body.get('file').name, 'aufnahme.wav');
  assert.equal(upload.options.body.get('file').type, 'audio/wav');
  assert.equal(upload.options.body.get('language'), 'de');
  assert.equal(p.elements.transcript.value, 'Hallo Welt');
});

test('Denied microphone permission closes AudioContext and restores controls', async () => {
  const p = page(); await p.ready();
  p.context.navigator.mediaDevices.getUserMedia = async () => { const error = new Error('denied'); error.name = 'NotAllowedError'; throw error; };
  await p.elements.record.events.click();
  assert.equal(p.contexts[0].closed, true);
  assert.equal(p.elements.record.disabled, false);
  assert.match(p.elements['whisper-note'].textContent, /abgelehnt/);
});

test('Chat sends history safely; failed requests do not enter history or lose the prompt', async () => {
  const p = page(); await p.ready(); p.elements['api-key'].value = 'test-key'; p.elements.prompt.value = '<script>alert(1)</script>';
  await p.elements['chat-form'].events.submit({ preventDefault() {} });
  const call = p.calls.find(c => c.url === 'v1/chat/completions');
  assert.equal(call.options.headers.Authorization, 'Bearer test-key');
  assert.equal(JSON.parse(call.options.body).messages[0].content, '<script>alert(1)</script>');
  assert.equal(p.elements.conversation.children[0].children[1].textContent, '<script>alert(1)</script>');
  assert.equal(p.run('history.length'), 2);
  p.context.fetch = async () => ({ ok: false, status: 401, json: async () => ({ error: 'Unauthorized' }) });
  p.elements.prompt.value = 'Retry me';
  await p.elements['chat-form'].events.submit({ preventDefault() {} });
  assert.equal(p.run('history.length'), 2);
  assert.equal(p.elements.prompt.value, 'Retry me');
  assert.equal(p.elements['send-chat'].disabled, false);
  assert.match(p.elements['chat-note'].textContent, /ungültig/);
});

test('Browser language is primary, explicit override wins, unsupported languages use service fallback', async () => {
  const p = page(); await p.ready();
  p.context.navigator.languages = ['fr-FR', 'en-US', 'de-DE'];
  assert.equal(p.run('chooseLanguage("auto")'), 'en');
  await p.run('loadLanguage("auto")');
  assert.equal(p.run('uiLanguage'), 'en');
  assert.equal(p.run('tr("ui_1")'), 'Ready');
  await p.run('loadLanguage("ru")');
  assert.equal(p.run('uiLanguage'), 'ru');
  assert.equal(p.run('tr("ui_1")'), 'Готово');
  p.context.navigator.languages = ['zh-CN'];
  p.run('config.service_language = "en"');
  assert.equal(p.run('chooseLanguage("auto")'), 'en');
  assert.equal(p.run('chooseLanguage("de")'), 'de');
});

test('Every translated DOM key exists in every locale', () => {
  const source = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
  const keys = [...source.matchAll(/data-i18n(?:-[a-z]+)?="([^"]+)"/g)].map(m => m[1]);
  assert.ok(keys.length > 40);
  for (const language of ['de', 'en', 'ru']) {
    const data = JSON.parse(fs.readFileSync(path.join(web, `../locales/${language}.json`), 'utf8'));
    for (const key of keys) assert.equal(typeof data.ui[key], 'string', `${language}:${key}`);
  }
});


test('Selected Qwen3 is shown with text-only mode, input limit and no image upload', async () => {
  const p = page({vlm_model: 'Qwen3-VL-2B-Instruct', chat_models: ['Qwen3-VL-2B-Instruct'],
    default_text_model: 'Qwen3-VL-2B-Instruct', vlm_max_images: 1,
    model_limits: {'Qwen3-VL-2B-Instruct': {max_input_tokens: 2048, context_length: 2048}}});
  await p.ready();
  assert.equal(p.elements['chat-model'].value, 'Qwen3-VL-2B-Instruct');
  assert.equal(p.elements['chat-mode'].value, 'text');
  assert.equal(p.elements.image.disabled, true);
  assert.match(p.elements['input-limit'].textContent, /2048/);
  p.elements.prompt.value = 'Hallo';
  p.elements.image.files = [{type: 'image/png', size: 20}];
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const sent = JSON.parse(p.calls.find(c => c.url === 'v1/chat/completions').options.body);
  assert.equal(sent.messages[0].content, 'Hallo');
  assert.equal(sent.model, 'Qwen3-VL-2B-Instruct');
  p.elements['chat-mode'].value = 'vision';
  p.elements['chat-mode'].events.change();
  assert.equal(p.elements.image.disabled, false);
  assert.equal(p.run('history.length'), 2);
});

function contents(element) {
  return [element.textContent || '', ...element.children.map(contents)].join(' ');
}

test('Per-request metrics show millisecond timestamps, measured duration and native usage', async () => {
  const p = page(); await p.ready();
  const NativeDate = Date; let dateIndex = 0;
  p.context.Date = class extends NativeDate {
    constructor(...args) { super(...(args.length ? args : [Date.UTC(2026, 9, 5, 20, 0, 0, 123) + dateIndex++ * 1250])); }
  };
  let clock = 0; p.context.performance = { now: () => clock++ * 1250 };
  p.context.fetch = async () => ({ok: true, json: async () => ({
    choices: [{message: {content: 'Antwort'}}], usage: {prompt_tokens: 900, completion_tokens: 30},
    metrics: {processing_ms: 1200, inference_ms: 1100, ttft_ms: 350, ttft_source: 'native',
      input_tokens: 900, input_tokens_source: 'native', output_tokens: 30, output_tokens_source: 'native',
      decode_tokens_per_second: 15},
  })});
  p.elements.prompt.value = 'Hallo';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const text = contents(p.elements.conversation.children[0]);
  assert.match(text, /[.,]123/); assert.match(text, /[.,]373/);
  assert.match(text, /1\.250 s/); assert.match(text, /1\.200 s/);
  assert.match(text, /0\.350 s/); assert.match(text, /900 · Modell/);
  assert.match(text, /30 · Modell/); assert.match(text, /15\.00 token\/s/);
});

test('Unknown usage and TTFT stay unavailable; failed requests retain measured diagnostics', async () => {
  const p = page(); await p.ready(); p.elements.prompt.value = 'Hallo';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  assert.match(contents(p.elements.conversation.children[0]), /nicht verfügbar/);
  p.context.fetch = async () => { throw new Error('offline'); };
  p.elements.prompt.value = 'Noch einmal';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const failed = p.elements.conversation.children[2];
  assert.equal(failed.removed, undefined);
  assert.match(contents(failed), /Anfrage fehlgeschlagen: offline/);
  assert.equal(p.run('history.length'), 2);
});

test('Model switches retain visible chat and text context; Gemma excludes images without erasing them', async () => {
  const p = page({llm_model: 'gemma-4-E2B-it'}); await p.ready();
  p.elements.prompt.value = 'Erste Frage';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  p.run('history[0].content = [{type: "text", text: "Erste Frage"}, {type: "image_url", image_url: {url: "data:image/png;base64,AA=="}}]');
  p.elements['chat-model'].value = 'gemma-4-E2B-it';
  p.elements['chat-model'].events.change();
  assert.equal(p.elements.conversation.children.length, 2);
  assert.equal(p.run('history.length'), 2);
  p.elements.prompt.value = 'Zweite Frage';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const calls = p.calls.filter(c => c.url === 'v1/chat/completions');
  const second = JSON.parse(calls[1].options.body);
  assert.equal(second.messages.length, 3);
  assert.equal(second.messages[0].content, 'Erste Frage');
  assert.equal(second.model, 'gemma-4-E2B-it');
  assert.equal(p.run('history[0].content[1].type'), 'image_url');
  p.elements['chat-model'].value = 'Qwen2-VL-2B-Instruct';
  p.elements['chat-model'].events.change();
  assert.equal(p.elements.conversation.children.length, 4);
  p.elements.prompt.value = 'Dritte Frage';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const third = JSON.parse(p.calls.filter(c => c.url === 'v1/chat/completions')[2].options.body);
  assert.equal(third.messages[0].content[1].type, 'image_url');
  assert.equal(p.elements.conversation.children.length, 6);
  p.elements['clear-chat'].events.click();
  assert.equal(p.run('history.length'), 0); assert.equal(p.elements.conversation.children.length, 0);
});

test('Repeated Whisper requests retain transcripts and individual metrics', async () => {
  const p = page(); await p.ready(); p.run('setAudio(new Blob(["wav"]), "test.wav")');
  await p.elements.transcribe.events.click(); await p.elements.transcribe.events.click();
  assert.equal(p.elements['transcription-history'].children.length, 2);
  for (const item of p.elements['transcription-history'].children) {
    assert.match(contents(item), /Hallo Welt/); assert.match(contents(item), /Gesamtdauer/);
  }
});

test('Long chats keep all visible requests while sending bounded recent context', async () => {
  const p = page(); await p.ready();
  for (let i = 0; i < 18; i++) {
    p.elements.prompt.value = `Frage ${i}`;
    await p.elements['chat-form'].events.submit({preventDefault() {}});
  }
  assert.equal(p.elements.conversation.children.length, 36);
  assert.equal(p.run('history.length'), 36);
  const last = JSON.parse(p.calls.filter(c => c.url === 'v1/chat/completions').at(-1).options.body);
  assert.equal(last.messages.length, 31);
  assert.equal(last.messages[0].role, 'user');
  assert.equal(last.messages.at(-1).content, 'Frage 17');
});

test('Hailo LLM is labeled as Hailo, disallows images and preserves text history across Gemma/VLM switches', async () => {
  const llm = 'Qwen2.5-1.5B-Instruct', gemma = 'gemma-4-E2B-it', vlm = 'Qwen2-VL-2B-Instruct';
  const p = page({llm_model: gemma, hailo_llm_model: llm, vision_models: [vlm],
    chat_models: [vlm, llm, gemma], default_text_model: llm,
    model_limits: {[llm]: {max_input_tokens: 2048, context_length: 2048}}});
  await p.ready();
  assert.equal(p.elements['chat-model'].value, llm);
  assert.equal(p.elements['chat-model'].options[1].text, `${llm} · Hailo`);
  assert.equal(p.elements['chat-mode'].disabled, true);
  assert.equal(p.elements.image.disabled, true);
  assert.match(p.elements['input-limit'].textContent, /2048/);
  p.run('history = [{role: "user", content: [{type: "text", text: "Bildfrage"}, {type: "image_url", image_url: {url: "data:image/png;base64,AA=="}}]}, {role: "assistant", content: "Bildantwort"}]');
  p.elements.prompt.value = 'Textfrage';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const sent = JSON.parse(p.calls.find(c => c.url === 'v1/chat/completions').options.body);
  assert.equal(sent.model, llm);
  assert.equal(sent.messages[0].content, 'Bildfrage');
  assert.equal(p.run('history[0].content[1].type'), 'image_url');
  p.elements['chat-model'].value = gemma;
  p.elements['chat-model'].events.change();
  assert.equal(p.run('history.length'), 4);
  p.elements['chat-model'].value = vlm;
  p.elements['chat-model'].events.change();
  p.elements['chat-mode'].value = 'vision';
  p.elements['chat-mode'].events.change();
  assert.equal(p.elements.image.disabled, false);
  p.elements['chat-model'].value = llm;
  p.elements['chat-model'].events.change();
  assert.equal(p.elements['chat-mode'].value, 'text');
  assert.equal(p.elements.image.disabled, true);
});

test('HA-Assist shows selected target limits and retains images for virtual routing', async () => {
  const ha = 'HA-Assist', gemma = 'gemma-4-E2B-it', vlm = 'Qwen2-VL-2B-Instruct';
  const p = page({ha_assist_model: ha, ha_assist: {text_model: gemma, vision_model: vlm},
    llm_model: gemma, chat_models: [vlm, gemma, ha], vision_models: [vlm, ha],
    model_limits: {[gemma]: {max_input_tokens: 4096, context_length: 16384},
      [vlm]: {max_input_tokens: 2048, context_length: 2048}}});
  await p.ready();
  assert.equal(p.elements['chat-model'].options[2].text, `${ha} · Home Assistant`);
  p.elements['chat-model'].value = ha;
  p.elements['chat-model'].events.change();
  assert.match(p.elements['input-limit'].textContent, /4096/);
  p.elements['chat-mode'].value = 'vision';
  p.elements['chat-mode'].events.change();
  assert.match(p.elements['input-limit'].textContent, /2048/);
  assert.equal(p.elements.image.disabled, false);
  p.run('history = [{role:"user", content:[{type:"text",text:"Bild"}, {type:"image_url",image_url:{url:"data:image/png;base64,AA=="}}]}, {role:"assistant", content:"Raum"}]');
  p.elements.prompt.value = 'Was ist rechts?';
  await p.elements['chat-form'].events.submit({preventDefault() {}});
  const body = JSON.parse(p.calls.find(c => c.url === 'v1/chat/completions').options.body);
  assert.equal(body.model, ha);
  assert.equal(body.messages[0].content[1].type, 'image_url');
  p.run('renderMeasurement(document.getElementById("auth-note"), {requested:new Date(),elapsed:0}, {metrics:{ha_route:{route:"vlm",backend_model:"Qwen2-VL-2B-Instruct"}}})');
  assert.match(contents(p.elements['auth-note']), /vlm · Qwen2-VL-2B-Instruct/);
});
