// Run with: node --test tests/test_web.cjs (Node >=18, no npm dependencies).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../src/hailo_services/web');

function page(overrides = {}, sessionStorage = { getItem() { return null; }, setItem() {}, removeItem() {} }, uiStorage = { getItem() { return null; }, setItem() {} }) {
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
    localStorage: uiStorage,
    sessionStorage,
    document: { documentElement: {}, querySelectorAll() { return []; }, hidden: false, getElementById: id => elements[id], createElement: () => new Element(), addEventListener() {} },
    window: { isSecureContext: true, AudioContext, AudioWorkletNode, addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => tracks }) } },
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    AudioContext, AudioWorkletNode, Blob, URL, FormData, performance, console, setInterval: () => 1, clearInterval() {},
    fetch: async (url, options = {}) => {
      calls.push({ url, options });
      return { ok: true, json: async () => /^\/?ui\/locales\//.test(url) ? JSON.parse(fs.readFileSync(path.join(web, '../locales', url.split('/').pop()), 'utf8')) : url === 'ui/config' ? config : url === 'health' ? { ready: true, group_id: 'SHARED', models: ['qwen', 'whisper'], pending: 0 } : url === 'v1/audio/transcriptions' ? { text: 'Hallo Welt' } : { choices: [{ message: { content: 'Antwort' } }] } };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(web, 'common.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(web, 'app.js'), 'utf8'), context);
  return { context, elements, tracks, contexts, nodes, calls, ready: () => new Promise(resolve => setImmediate(resolve)), run: code => vm.runInContext(code, context) };
}

test('API key survives navigation between Playground and monitoring in one browser tab', async () => {
  assert.match(fs.readFileSync(path.join(web, 'index.html'), 'utf8'), /href="\/monitor"/);
  assert.match(fs.readFileSync(path.join(web, 'monitor.html'), 'utf8'), /href="\/"[^>]*>Playground/);
  const values = new Map();
  const sessionStorage = {
    getItem: key => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  };
  const playground = page({}, sessionStorage); await playground.ready();
  playground.elements['api-key'].value = 'shared-secret';
  playground.elements['api-key'].events.input();
  assert.equal(values.get('hailo-api-key'), 'shared-secret');

  const monitorElements = Object.fromEntries([...fs.readFileSync(path.join(web, 'monitor.html'), 'utf8').matchAll(/id="([^"]+)"/g)]
    .map(m => [m[1], { value: '', events: {}, addEventListener(type, fn) { this.events[type] = fn; } }]));
  const monitorCalls = [];
  const monitor = vm.createContext({
    sessionStorage, localStorage: {getItem() {return 'en';}}, window: {addEventListener() {}}, document: {documentElement: {}, querySelectorAll() {return [];}, getElementById: id => monitorElements[id] },
    navigator: { language: 'de' }, setInterval() {},
    fetch: async (url, options) => {
      monitorCalls.push({url, options});
      if (url.startsWith('/ui/locales/')) return {ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(web, '../locales', url.split('/').pop()), 'utf8'))};
      if (url === '/ui/config') return {ok: true, json: async () => ({auth_required: false})};
      return { ok: false, status: 401 };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(web, 'common.js'), 'utf8'), monitor);
  vm.runInContext(fs.readFileSync(path.join(web, 'monitor.js'), 'utf8'), monitor);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(monitorElements.key.value, 'shared-secret');
  assert.equal(monitorElements.access.hidden, true);
  assert.equal(monitorCalls.find(call => call.url === '/v1/monitor').options.headers.Authorization, 'Bearer shared-secret');
  monitorElements.key.value = 'changed-secret'; monitorElements.key.events.input();
  const nextPlayground = page({}, sessionStorage); await nextPlayground.ready();
  assert.equal(nextPlayground.elements['api-key'].value, 'changed-secret');
  nextPlayground.elements['api-key'].value = ''; nextPlayground.elements['api-key'].events.input();
  assert.equal(values.has('hailo-api-key'), false);
});

test('monitoring shows enabled services, native telemetry and preserves log scroll', async () => {
  class Element {
    constructor(tag = '') { this.tag = tag; this.children = []; this.value = ''; this.hidden = false; this.events = {}; this.scrollTop = 0; this._left = 0; this.scrollWidth = 2000; this.clientWidth = 400; this.scrollHeight = 200; this.clientHeight = 100; }
    get scrollLeft() { return this._left; }
    set scrollLeft(value) { this._left = Math.max(0, Math.min(value, this.scrollWidth - this.clientWidth)); }
    get textContent() { return this._text || ''; }
    set textContent(value) { this._text = value; if (this.tag === 'pre') { this.scrollWidth = Math.max(this.clientWidth, ...String(value).split('\n').map(line => line.length * 10)); this.scrollLeft = this._left; } }
    get firstElementChild() { return this.children[0] || null; }
    addEventListener(type, fn) { this.events[type] = fn; }
    setAttribute(name, value) { this[name] = value; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
  }
  const elements = Object.fromEntries([...fs.readFileSync(path.join(web, 'monitor.html'), 'utf8').matchAll(/id="([^"]+)"/g)]
    .map(m => [m[1], new Element()]));
  const context = vm.createContext({
    document: {documentElement: {}, querySelectorAll() {return [];}, getElementById: id => elements[id], createElement: tag => new Element(tag) },
    sessionStorage: { getItem() { return null; } }, localStorage: {getItem() {return 'en';}}, navigator: { language: 'de' }, setInterval() {},
    window: {addEventListener() {}, scrollX: 0, scrollY: 500, scrollTo(x, y) {this.scrollX = x; this.scrollY = y;}},
    fetch: async url => url.startsWith('/ui/locales/') ? {ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(web, '../locales', url.split('/').pop()), 'utf8'))} : url === '/ui/config'
      ? { ok: true, json: async () => ({auth_required: false}) }
      : { ok: true, json: async () => ({ health: {ready: true, workers: {hailo: {status: 'ready', pid: 102}}, model_paths: {vlm: '/models/qwen.hef'}}, hailo_device: {hailort_version: "5.4.0", sampled_at: "2026-10-14T18:01:42Z", nnc_percent: 37, cpu_percent: 12, dsp_percent: 0, ram_used_bytes: 5 * 1073741824, ram_total_bytes: 8 * 1073741824, temperature_c: 54, throttling: false, resident_models: [{name: 'Qwen3-VL-2B', state: 'loaded'}]}, services: {VLM: {enabled: true, active: 0, models: ['Qwen3-VL-2B']}, Piper: {enabled: false, active: 0}}, system_services: {host: {name: 'cm5', status: 'running'}, http: {host: '0.0.0.0', port: 8090, status: 'listening', local_address: ['192.168.2.5', 8090]}, wyoming: {host: '0.0.0.0', port: 10300, status: 'listening'}, mqtt: {host: 'broker.local', port: 1883, status: 'connected'}, zmq: {endpoint: 'tcp://127.0.0.1:5555', status: 'listening'}}, resources: {
        processes: [{name: 'gateway', pid: 101, rss_bytes: 1048576, cpu_percent: 2},
          {name: 'hailo', pid: 102, rss_bytes: 2097152, cpu_percent: 10}]},
        logs: [{name: 'gateway', pid: 101, lines: ['gateway line', 'event=monitor_request service=VLM duration_ms=1']},
          {name: 'hailo', pid: 102, lines: ['worker line']}] }) },
  });
  vm.runInContext(fs.readFileSync(path.join(web, 'common.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(web, 'monitor.js'), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.access.hidden, true);
  assert.equal(vm.runInContext('logTag("events=frame_received operation=vision_detect protocol=zmq model=YOLOv11m", "event")', context), 'frame_received');
  assert.equal(vm.runInContext('logTag("events=frame_received operation=vision_detect protocol=zmq model=YOLOv11m", "protocol")', context), 'zmq');
  assert.equal(elements.processes.children.length, 2);
  assert.equal(elements.services.children.length, 1);
  assert.equal(elements.services.children[0].children[0].textContent, 'VLM');
  assert.equal(elements.services.children[0].children[0].children[0].textContent, ' (Qwen3-VL-2B)');
  assert.match(contents(elements['hailo-device']), /NNC 37 %/);
  assert.match(elements['hailo-note'].textContent, /HailoRT 5\.4\.0/);
  assert.doesNotMatch(contents(elements['hailo-device']), /DSP/);
  assert.match(contents(elements['hailo-device']), /54 °C/);
  assert.match(contents(elements['hailo-device']), /Throttling No/);
  assert.match(contents(elements['resident-models']), /Qwen3-VL-2B loaded/);
  assert.match(contents(elements.health), /192.168.2.5:8090/);
  assert.match(contents(elements.health), /10300/);
  assert.match(contents(elements.health), /broker.local:1883/);
  assert.doesNotMatch(contents(elements.health), /qwen|Modelle/);
  assert.equal(elements['log-tabs'].children.length, 2);
  assert.equal(elements.logs.children.length, 1);
  assert.match(elements.logs.children[0].children[1].textContent, /gateway line/);
  const logPane = elements.logs.children[0].children[1];
  logPane.scrollTop = 45; logPane.scrollLeft = 23;
  context.window.scrollY = 680;
  vm.runInContext('renderLogs([{name: "gateway", pid: 101, lines: ["new gateway line", "event=monitor_request service=VLM duration_ms=2"]}, {name: "hailo", pid: 102, lines: ["worker line"]}])', context);
  assert.equal(elements.logs.children[0].children[1], logPane);
  assert.equal(logPane.scrollTop, 45);
  assert.equal(logPane.scrollLeft, 23);
  assert.equal(context.window.scrollY, 680);
  logPane.scrollTop = 100;
  vm.runInContext('renderLogs([{name: "gateway", pid: 101, lines: ["another line", "event=monitor_request service=VLM duration_ms=3"]}, {name: "hailo", pid: 102, lines: ["worker line"]}])', context);
  assert.equal(logPane.scrollTop, 200); // Following the bottom keeps new entries visible.
  assert.match(logPane.textContent, /another line/);
  assert.equal(elements['service-tabs'].children.length, 2);
  elements['service-tabs'].children[1].events.click();
  assert.doesNotMatch(elements.logs.children[0].children[1].textContent, /gateway line/);
  assert.match(elements.logs.children[0].children[1].textContent, /service=VLM/);
  elements['log-tabs'].children[1].events.click();
  assert.equal(elements.logs.children.length, 1);
  assert.match(elements.logs.children[0].children[1].textContent, /worker line/);
  assert.equal(logPane.scrollTop, 0);
  elements['log-tabs'].children[0].events.click();
  vm.runInContext('renderLogs([{name: "gateway", pid: 101, lines: ["x".repeat(400)]}, {name: "hailo", pid: 102, lines: ["worker line"]}])', context);
  logPane.scrollLeft = 500; logPane.events.scroll();
  elements['log-tabs'].scrollLeft = 70;
  vm.runInContext('renderLogs([{name: "gateway", pid: 101, lines: ["short"]}, {name: "hailo", pid: 102, lines: ["worker line"]}])', context);
  assert.equal(logPane.scrollLeft, 0); // The current batch has no horizontal overflow.
  logPane.events.scroll(); // A browser emits this after its programmatic clamp.
  assert.equal(elements['log-tabs'].scrollLeft, 70);
  vm.runInContext('renderLogs([{name: "gateway", pid: 101, lines: ["y".repeat(400)]}, {name: "hailo", pid: 102, lines: ["worker line"]}])', context);
  assert.equal(logPane.scrollLeft, 500); // The desired offset survives the shorter batch.
  elements['log-tabs'].children[1].events.click();
  elements['log-tabs'].children[0].events.click();
  assert.equal(logPane.scrollLeft, 500); // It also survives a process-tab round trip.
});

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
  const p = page({ui_languages: ['de', 'en', 'ru']}); await p.ready();
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
  const source = ['index.html', 'settings.html', 'monitor.html'].map(file => fs.readFileSync(path.join(web, file), 'utf8')).join('\n');
  const keys = [...source.matchAll(/data-i18n(?:-[a-z]+)?="([^"]+)"/g)].map(m => m[1]);
  assert.ok(keys.length > 40);
  for (const language of ['de', 'en', 'ru', 'es', 'fr', 'it', 'nl', 'pt']) {
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

test('settings group fields, preserve unsaved edits on language change and save catalogue selections', async () => {
  class Element {
    constructor(tag = 'div') { this.tag = tag; this.children = []; this.dataset = {}; this.events = {}; this.attributes = {}; this.value = ''; this._text = ''; this.classList = {add: name => {this.className = `${this.className || ''} ${name}`;}, toggle: (name, on) => {this.className = (this.className || '').split(' ').filter(item => item !== name).concat(on ? [name] : []).join(' ');}}; }
    addEventListener(type, callback) { this.events[type] = callback; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    get options() { return this.children.filter(node => node.tag === 'option'); }
    get textContent() { return this._text + this.children.map(node => node.textContent || '').join(''); }
    set textContent(text) { this._text = String(text); this.children = []; }
    getAttribute(name) { return name.startsWith('data-') ? this.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] : this.attributes[name]; }
    setAttribute(name, value) { this.attributes[name] = value; }
    querySelectorAll(selector) {
      const match = selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
      const nodes = this.children.flatMap(node => [node, ...node.querySelectorAll(selector)]);
      return nodes.filter(node => { const value = node.getAttribute(match[1]); return value !== undefined && (match[2] === undefined || value === match[2]); });
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  }
  const elements = Object.fromEntries([...fs.readFileSync(path.join(web, 'settings.html'), 'utf8').matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const storageValues = new Map(), writes = [];
  const storage = {getItem: key => storageValues.get(key) || null, setItem: (key, value) => storageValues.set(key, value)};
  let configuration = {version: 'v1', path: '/etc/config.yaml', restart_available: false, fields: [
    {path: 'settings.wyoming_host', type: 'str', value: '0.0.0.0'},
    {path: 'settings.wyoming_port', type: 'int', value: 10300},
    {path: 'settings.ha_assist_verify_attempts', type: 'int', value: 2},
    {path: 'models.vlm.model', type: 'str', value: 'Qwen2-VL-2B-Instruct', options: ['Qwen2-VL-2B-Instruct', 'Qwen3-VL-2B-Instruct']},
    {path: 'settings.whisper_hef', type: 'str', value: '/opt/models/local-whisper.hef', options: ['Whisper-Base', 'Whisper-Tiny']},
  ]};
  const root = new Element(); root.children = Object.values(elements);
  const context = vm.createContext({
    document: {documentElement: {}, getElementById: id => elements[id], querySelectorAll: selector => root.querySelectorAll(selector), createElement: tag => new Element(tag)},
    window: {addEventListener() {}}, navigator: {language: 'de'}, localStorage: storage,
    sessionStorage: {getItem() {return null;}},
    Option: class extends Element {constructor(text, value) {super('option'); this._text = text; this.value = value;}},
    fetch: async (url, options = {}) => ({ok: true, json: async () => {
      if (url.startsWith('/ui/locales/')) return JSON.parse(fs.readFileSync(path.join(web, '../locales', url.split('/').pop()), 'utf8'));
      if (url === '/ui/config') return {auth_required: false, ui_languages: ['de', 'en', 'ru', 'es', 'fr', 'it', 'nl', 'pt']};
      if (options.method === 'PUT') {
        const patch = JSON.parse(options.body); writes.push(patch);
        configuration = {...configuration, version: 'v2', fields: configuration.fields.map(field => ({...field, value: field.path in patch.changes ? patch.changes[field.path] : field.value}))};
      }
      return structuredClone(configuration);
    }}),
  });
  for (const file of ['common.js', 'settings.js']) vm.runInContext(fs.readFileSync(path.join(web, file), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.fields.children.length, 4);
  const wyoming = elements.fields.children.find(node => node.id === 'settings-wyoming');
  assert.deepEqual(wyoming.querySelectorAll('[data-path]').map(node => node.dataset.path), ['settings.wyoming_host', 'settings.wyoming_port']);
  const input = name => elements.fields.querySelector(`[data-path="${name}"]`);
  assert.equal(input('models.vlm.model').tag, 'select');
  assert.equal(input('models.vlm.model').options.length, 2);
  assert.equal(input('settings.whisper_hef').value, '/opt/models/local-whisper.hef');
  input('settings.wyoming_port').value = '10400';
  input('settings.wyoming_port').events.input();
  const portRow = wyoming.children[0].children[1].children[1];
  assert.match(portRow.className, /is-dirty/);
  input('models.vlm.model').value = 'Qwen3-VL-2B-Instruct';
  elements['ui-language'].value = 'ru'; await elements['ui-language'].events.change();
  assert.equal(storageValues.get('hailo-ui-language'), 'ru');
  assert.equal(context.document.documentElement.lang, 'ru');
  assert.equal(input('settings.wyoming_port').value, '10400');
  assert.equal(input('models.vlm.model').value, 'Qwen3-VL-2B-Instruct');
  assert.match(elements.notice.textContent, /Конфигурация/);
  await elements.form.events.submit({preventDefault() {}});
  assert.deepEqual(writes[0].changes, {'settings.wyoming_port': 10400, 'models.vlm.model': 'Qwen3-VL-2B-Instruct'});
  assert.equal(input('settings.whisper_hef').value, '/opt/models/local-whisper.hef');
  const playground = page({}, undefined, storage); await playground.ready();
  assert.equal(playground.run('uiLanguage'), 'ru');
  assert.equal(playground.context.document.documentElement.lang, 'ru');
  assert.doesNotMatch(fs.readFileSync(path.join(web, 'index.html'), 'utf8'), /id="ui-language"/);
});
