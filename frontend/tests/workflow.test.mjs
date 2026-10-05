import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as core from '../core.mjs';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');
// Run the actual browser event handlers in isolated DOMs using only Node built-ins.
// The imported pure helpers are supplied to the VM; no app handlers are rewritten.
const script = new vm.Script(source.replace(/^import [^\n]+\n/, ''), { filename: 'app.js' });

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.textContent = '';
    this.disabled = false;
    this.hidden = false;
  }
  set innerHTML(_) { throw new Error('Unsafe HTML rendering'); }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...nodes) {
    for (const node of nodes) this.children.push(...(node.tag === 'fragment' ? node.children : [node]));
  }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, callback) { (this.listeners[type] ??= []).push(callback); }
  async dispatch(type) {
    for (const callback of this.listeners[type] ?? []) await callback({ preventDefault() {} });
  }
}

function harness({ blobSize = 100, decode } = {}) {
  const frame = { labels: [], boxes: [], draws: 0 };
  const context = {
    clearRect() { frame.labels = []; frame.boxes = []; },
    drawImage() { frame.draws++; },
    fillRect() {},
    strokeRect(...box) { frame.boxes.push(box); },
    fillText(label) { frame.labels.push(label); },
    measureText(text) { return { width: text.length * 10 }; },
  };
  const elements = {};
  // Seed actual page IDs and control defaults to catch markup/handler mismatches.
  for (const match of html.matchAll(/<([a-z]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = new Element(match[1]);
    element.disabled = /\sdisabled\b/.test(match[0]);
    element.hidden = /\shidden\b/.test(match[0]);
    element.value = match[0].match(/\bvalue="([^"]+)"/)?.[1] ?? '';
    elements[match[2]] = element;
  }
  elements.preview.getContext = () => context;
  const requests = [];
  const timers = new Map();
  const blobs = [];
  const revoked = [];
  let timerID = 0;
  const window = new Element();
  const document = {
    getElementById(id) { assert.ok(elements[id], `Missing DOM element ${id}`); return elements[id]; },
    createDocumentFragment() { return new Element('fragment'); },
    createElement(tag) {
      const element = new Element(tag);
      if (tag === 'canvas') {
        element.getContext = () => ({ fillRect() {}, drawImage() {} });
        element.toBlob = (callback, type) => {
          const blob = { size: blobSize, type };
          blobs.push(blob);
          callback(blob);
        };
      }
      return element;
    },
  };
  script.runInNewContext({
    ...core, document, window, AbortController, Error, TypeError,
    Image: class {
      naturalWidth = 100;
      naturalHeight = 200;
      decode() { return decode ? decode() : Promise.resolve(); }
    },
    URL: { createObjectURL: () => 'blob:test-image', revokeObjectURL: (url) => revoked.push(url) },
    FormData: class {
      entries = [];
      append(...entry) { this.entries.push(entry); }
    },
    fetch(url, options) {
      const pending = deferred();
      requests.push({ url, options, ...pending });
      return pending.promise;
    },
    setTimeout(callback, delay) { timers.set(++timerID, { callback, delay }); return timerID; },
    clearTimeout(id) { timers.delete(id); },
  });
  return {
    elements, requests, frame, timers, blobs, revoked, window,
    async upload(name = 'crop.jpg') {
      elements['image-input'].files = [{ name, size: 100, type: 'image/jpeg' }];
      await elements['image-input'].dispatch('change');
    },
    submit: () => elements['prediction-form'].dispatch('submit'),
    async filter(value) {
      elements['confidence-filter'].value = String(value);
      await elements['confidence-filter'].dispatch('input');
    },
  };
}

const detections = [
  { class_id: 0, class_name: 'Leaf spot', confidence: 0.25, bbox: [10, 20, 40, 80] },
  { class_id: 1, class_name: '<img src=x onerror=alert(1)>', confidence: 0.8, bbox: [50, 90, 90, 180] },
  { class_id: 2, class_name: 'Rust', confidence: 1, bbox: [5, 5, 25, 30] },
];
const prediction = (items = detections) => ({ image_width: 100, image_height: 200, detections: items });
const response = (data = prediction(), ok = true, status = 200) => ({ ok, status, json: async () => data });
const rows = (app) => app.elements.results.children.map((row) => row.children.map((span) => span.textContent));

test('confidence control has a label, keyboard-native range, inclusive bounds and help', () => {
  assert.match(html, /<label for="confidence-filter">Minimum model confidence<\/label>/);
  assert.match(html, /id="confidence-filter" type="range" min="25" max="100" step="1" value="25" disabled/);
  assert.match(html, /aria-describedby="confidence-help"/);
  assert.match(html, /cannot recover detections below that floor/);
  assert.match(html, /not a calibrated probability/);
});

test('upload → predict → cached filter redraws and renumbers boxes and safe text without inference', async () => {
  const app = harness();
  const { elements, requests, frame } = app;
  assert.equal(elements['confidence-filter'].disabled, true);
  await app.upload('<script>crop.jpg</script>');
  assert.equal(requests.length, 0);
  assert.equal(elements['run-button'].disabled, false);
  assert.match(elements['file-info'].textContent, /^<script>crop.jpg<\/script>/);
  assert.equal(app.revoked.length, 1);
  const pending = app.submit();
  assert.equal(elements.output.attributes['aria-busy'], 'true');
  assert.equal(elements['run-button'].disabled, true);
  assert.equal(requests[0].url, '/predict');
  const [key, blob, name] = requests[0].options.body.entries[0];
  assert.equal(key, 'file');
  assert.equal(blob, app.blobs[0]);
  assert.equal(blob.type, 'image/png');
  assert.equal(name, 'crop-normalized.png');
  requests[0].resolve(response());
  await pending;
  assert.equal(elements.output.attributes['aria-busy'], 'false');
  assert.equal(elements.count.textContent, '3 / 3 shown');
  assert.equal(elements['confidence-filter'].disabled, false);
  assert.deepEqual(rows(app), [
    ['1', 'Leaf spot', '25.0%'], ['2', detections[1].class_name, '80.0%'], ['3', 'Rust', '100.0%'],
  ]);
  await app.filter(80);
  assert.equal(elements['confidence-value'].textContent, '80%');
  assert.equal(elements['confidence-filter'].attributes['aria-valuetext'], '80% minimum model confidence');
  assert.equal(elements.count.textContent, '2 / 3 shown');
  assert.deepEqual(rows(app).map((row) => row[0]), ['1', '2']);
  assert.deepEqual(frame.labels, ['1', '2']);
  assert.deepEqual(frame.boxes[0], [50, 90, 40, 90]);
  assert.equal(frame.boxes.length, 4); // Outer and inner strokes per box.
  await app.filter(100);
  assert.deepEqual(rows(app), [['1', 'Rust', '100.0%']]);
  assert.deepEqual(frame.labels, ['1']);
  await app.filter(25);
  assert.equal(rows(app).length, 3);
  assert.equal(requests.length, 1);
  assert.equal(app.timers.size, 0);
});

test('zero filtered or returned detections clear old boxes and explain health uncertainty', async () => {
  const app = harness();
  await app.upload();
  const pending = app.submit();
  app.requests[0].resolve(response(prediction(detections.slice(0, 2))));
  await pending;
  await app.filter(100);
  assert.equal(app.elements.count.textContent, '0 / 2 shown');
  assert.deepEqual(app.frame.boxes, []);
  assert.deepEqual(rows(app), []);
  assert.match(app.elements['results-help'].textContent, /does not mean the crop is healthy/);
  await app.filter(25);
  assert.equal(rows(app).length, 2);
  const retry = app.submit();
  assert.equal(app.elements['confidence-filter'].disabled, true);
  app.requests[1].resolve(response(prediction([])));
  await retry;
  assert.equal(app.elements.count.textContent, '0 / 0 shown');
  assert.match(app.elements.status.textContent, /does not mean the crop is healthy/);
});

test('oversized normalized PNG is rejected locally even when source JPEG fits', async () => {
  const app = harness({ blobSize: core.MAX_FILE_BYTES + 1 });
  await app.upload();
  assert.match(app.elements.error.textContent, /normalized PNG exceeds the 20 MiB/);
  assert.equal(app.elements['run-button'].disabled, true);
  assert.equal(app.elements.preview.hidden, true);
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.revoked.length, 1);
});

test('new upload discards a late success or error from the previous request', async () => {
  for (const lateError of [false, true]) {
    const app = harness();
    await app.upload('first.jpg');
    const oldRequest = app.submit();
    await app.upload('new.jpg');
    assert.equal(app.requests[0].options.signal.aborted, true);
    const newRequest = app.submit();
    if (lateError) app.requests[0].reject(new Error('old failure'));
    else app.requests[0].resolve(response());
    await oldRequest;
    assert.equal(app.elements.output.attributes['aria-busy'], 'true');
    assert.equal(app.elements.error.textContent, '');
    assert.deepEqual(rows(app), []);
    app.requests[1].resolve(response(prediction([detections[1]])));
    await newRequest;
    assert.equal(app.elements.count.textContent, '1 / 1 shown');
    assert.match(app.elements['file-info'].textContent, /^new.jpg/);
  }
});

test('late response body and late image decode cannot overwrite the next upload', async () => {
  const body = deferred();
  const app = harness();
  await app.upload();
  const pending = app.submit();
  app.requests[0].resolve({ ok: true, json: () => body.promise });
  await Promise.resolve();
  await app.upload('next.jpg');
  body.resolve(prediction());
  await pending;
  assert.equal(app.elements.count.textContent, 'Ready');
  assert.deepEqual(rows(app), []);

  const oldDecode = deferred();
  let decodes = 0;
  const other = harness({ decode: () => ++decodes === 1 ? oldDecode.promise : Promise.resolve() });
  const first = other.upload('old.jpg');
  await other.upload('current.jpg');
  oldDecode.reject(new Error('old decode failed'));
  await first;
  assert.equal(other.elements.error.textContent, '');
  assert.match(other.elements['file-info'].textContent, /^current.jpg/);
  assert.equal(other.elements['run-button'].disabled, false);
});

test('HTTP errors, malformed responses, network errors and timeout retain retry state', async () => {
  const cases = [
    { value: response({ detail: '<script>failure</script>' }, false, 503), message: /HTTP 503.*<script>failure<\/script>/ },
    { value: { ok: true, status: 200, json: async () => { throw new Error('bad JSON'); } }, message: /unreadable response/ },
    { value: response({ image_width: 200, image_height: 100, detections: [] }), message: /unexpected image dimensions/ },
    { error: new TypeError('network'), message: /Could not reach/ },
    { timeout: true, error: new Error('aborted'), message: /timed out after two minutes/ },
  ];
  for (const scenario of cases) {
    const app = harness();
    await app.upload();
    const pending = app.submit();
    if (scenario.timeout) {
      const timer = [...app.timers.values()][0];
      assert.equal(timer.delay, 120000);
      timer.callback();
      assert.equal(app.requests[0].options.signal.aborted, true);
    }
    if (scenario.error) app.requests[0].reject(scenario.error);
    else app.requests[0].resolve(scenario.value);
    await pending;
    assert.match(app.elements.error.textContent, scenario.message);
    assert.equal(app.elements.error.hidden, false);
    assert.equal(app.elements.count.textContent, 'Request failed');
    assert.equal(app.elements['run-button'].disabled, false);
    assert.equal(app.elements['confidence-filter'].disabled, true);
    assert.equal(app.elements.output.attributes['aria-busy'], 'false');
    assert.equal(app.timers.size, 0);
  }
});
