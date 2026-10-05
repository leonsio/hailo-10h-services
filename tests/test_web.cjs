// Run with: node --test tests/test_web.cjs (Node >=18, no npm dependencies).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../src/hailo_services/web');

function page() {
  class Element {
    constructor() { this.value = ''; this.children = []; this.events = {}; this.disabled = false; this.hidden = false; this.files = []; this.classList = { toggle() {} }; }
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
  const config = { auth_required: true, vlm_model: 'Qwen2-VL-2B-Instruct', whisper_model: 'whisper-base', max_body: 16 * 1024 * 1024, recording_seconds: 1, max_audio_seconds: 120, language: 'de' };
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
