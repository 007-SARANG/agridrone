import { fittedSize, validateFile, validatePreparedBlob, validatePrediction, scaleBox } from './core.mjs';

const $ = (id) => document.getElementById(id);
const input = $('image-input');
const button = $('run-button');
const preview = $('preview');
const context = preview.getContext('2d');
const confidence = $('confidence-filter');
let generation = 0;
let controller = null;
let prepared = null;
let busy = false;
let prediction = null;

function status(message) { $('status').textContent = message; }
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function setBusy(value) {
  busy = value;
  $('output').setAttribute('aria-busy', String(value));
  button.disabled = value || !prepared;
  button.textContent = value ? 'Running detection…' : 'Run detection ↗';
}
function clearResults() {
  prediction = null;
  confidence.disabled = true;
  $('results').replaceChildren();
  $('results-help').textContent = 'Run detection to explore the model’s output. Confidence is a model score, not a calibrated probability of disease.';
  $('count').textContent = 'Awaiting prediction';
  preview.setAttribute('aria-label', 'Selected crop image. No predictions yet.');
}
function drawImage() {
  context.clearRect(0, 0, preview.width, preview.height);
  context.drawImage(prepared.canvas, 0, 0, preview.width, preview.height);
}

async function prepareImage(file) {
  // Decode in the browser's displayed orientation, then encode those exact
  // pixels as PNG. Removing EXIF makes the server and preview agree even when
  // the server's image decoder does not apply EXIF orientation itself.
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const size = fittedSize(image.naturalWidth, image.naturalHeight);
    const canvas = document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Your browser could not prepare this image.');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    validatePreparedBlob(blob);
    return { canvas, blob };
  } finally {
    URL.revokeObjectURL(url);
  }
}

input.addEventListener('change', async () => {
  const current = ++generation;
  controller?.abort();
  controller = null;
  prepared = null;
  setBusy(false);
  showError('');
  clearResults();
  confidence.value = '25';
  updateConfidence();
  preview.hidden = true;
  $('placeholder').hidden = false;
  $('count').textContent = 'Awaiting image';
  $('file-info').textContent = 'No image selected';
  const file = input.files[0];
  if (!file) { status('Choose an image to begin.'); return; }
  try {
    validateFile(file);
    status('Preparing image…');
    const next = await prepareImage(file);
    if (current !== generation) return;
    prepared = next;
    const size = fittedSize(next.canvas.width, next.canvas.height, 1600);
    preview.width = size.width;
    preview.height = size.height;
    drawImage();
    preview.hidden = false;
    $('placeholder').hidden = true;
    $('file-info').textContent = `${file.name} · ${next.canvas.width} × ${next.canvas.height} px prepared`;
    $('count').textContent = 'Ready';
    button.disabled = !context;
    if (!context) throw new Error('Canvas is not available in this browser.');
    status('Image ready. Select Run detection to upload it.');
  } catch (error) {
    if (current !== generation) return;
    prepared = null;
    button.disabled = true;
    showError(error.name === 'EncodingError' ? 'This file could not be decoded. Choose a valid JPEG, PNG, or WebP image.' : error.message);
    status('Image could not be prepared. Choose another file.');
  }
});

function updateConfidence() {
  $('confidence-value').textContent = `${confidence.value}%`;
  confidence.setAttribute('aria-valuetext', `${confidence.value}% minimum model confidence`);
}

function renderPrediction(data) {
  drawImage();
  const detections = data.detections.filter((detection) => detection.confidence >= Number(confidence.value) / 100);
  const fragment = document.createDocumentFragment();
  const lineWidth = Math.max(2, preview.width / 300);
  const fontSize = Math.max(16, preview.width / 45);
  context.lineWidth = lineWidth;
  context.font = `bold ${fontSize}px system-ui, sans-serif`;
  detections.forEach((detection, index) => {
    const [x1, y1, x2, y2] = scaleBox(detection.bbox, data.image_width, data.image_height, preview.width, preview.height);
    context.strokeStyle = '#fff';
    context.lineWidth = lineWidth + 2;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.strokeStyle = '#174e37';
    context.lineWidth = lineWidth;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = String(index + 1);
    const tagWidth = context.measureText(label).width + 12;
    const tagHeight = fontSize + 10;
    const tagX = Math.max(0, Math.min(x1, preview.width - tagWidth));
    const tagY = Math.max(0, Math.min(y1 - tagHeight, preview.height - tagHeight));
    context.fillStyle = '#174e37';
    context.fillRect(tagX, tagY, tagWidth, tagHeight);
    context.fillStyle = '#fff';
    context.fillText(label, tagX + 6, tagY + fontSize + 1);
    const row = document.createElement('li');
    row.className = 'result';
    for (const [className, text] of [
      ['result-number', label],
      ['result-label', detection.class_name],
      ['result-confidence', `${(detection.confidence * 100).toFixed(1)}%`],
    ]) {
      const span = document.createElement('span');
      span.className = className;
      span.textContent = text;
      row.append(span);
    }
    fragment.append(row);
  });
  $('results').replaceChildren(fragment);
  const count = detections.length;
  const total = data.detections.length;
  $('count').textContent = `${count} / ${total} shown`;
  $('results-help').textContent = count ? 'Numbers match the boxes above. Confidence is a model score, not a calibrated probability of disease.' : 'No detections shown at this threshold. This does not mean the crop is healthy: the model may miss disease or encounter unfamiliar conditions.';
  preview.setAttribute('aria-label', `Crop image with ${count} of ${total} returned regions shown at ${confidence.value}% minimum model confidence. Labels and scores are listed below.`);
  status(`Detection complete. ${count} of ${total} returned regions shown at ${confidence.value}% minimum confidence.${count ? '' : ' No displayed boxes does not mean the crop is healthy.'}`);
}

confidence.addEventListener('input', () => {
  updateConfidence();
  if (prediction) renderPrediction(prediction);
});

$('prediction-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!prepared || busy) return;
  const current = generation;
  const request = new AbortController();
  controller = request;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; request.abort(); }, 120000);
  clearResults();
  drawImage();
  showError('');
  setBusy(true);
  status('Uploading image and running detection. The first request may take longer while the model loads.');
  try {
    const form = new FormData();
    form.append('file', prepared.blob, 'crop-normalized.png');
    const response = await fetch('/predict', { method: 'POST', body: form, signal: request.signal });
    if (current !== generation) return;
    let data;
    try { data = await response.json(); } catch { throw new Error(`The server returned an unreadable response (HTTP ${response.status}). Please try again.`); }
    if (current !== generation) return;
    if (!response.ok) {
      const detail = typeof data?.detail === 'string' ? ` ${data.detail.slice(0, 500)}` : '';
      throw new Error(`Detection failed (HTTP ${response.status}).${detail || ' Please try again later.'}`);
    }
    prediction = validatePrediction(data, prepared.canvas.width, prepared.canvas.height);
    confidence.disabled = false;
    renderPrediction(prediction);
  } catch (error) {
    if (current !== generation) return;
    showError(timedOut ? 'The request timed out after two minutes. The server may still be processing it. You can try again.' : error instanceof TypeError ? 'Could not reach the prediction service. Check your connection and try again.' : error.message);
    $('count').textContent = 'Request failed';
    status('Detection did not complete. You can retry or choose another image.');
  } finally {
    clearTimeout(timeout);
    if (current === generation) { controller = null; setBusy(false); }
  }
});

window.addEventListener('pagehide', () => { ++generation; controller?.abort(); setBusy(false); });
