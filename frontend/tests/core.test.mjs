import test from 'node:test';
import assert from 'node:assert/strict';
import { fittedSize, validateFile, validatePreparedBlob, validatePrediction, scaleBox, MAX_FILE_BYTES } from '../core.mjs';

test('file validation permits supported images and rejects empty, oversized, and unsupported files', () => {
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) {
    assert.doesNotThrow(() => validateFile({ type, size: 10 }));
  }
  assert.doesNotThrow(() => validateFile({ type: '', name: 'crop.JPEG', size: 10 }));
  for (const file of [{ size: 0 }, { size: MAX_FILE_BYTES + 1 }, { size: 1, type: 'image/svg+xml' }, { size: 1, type: '', name: 'file.txt' }]) {
    assert.throws(() => validateFile(file));
  }
});

test('normalized PNG must be non-empty and fit the exact file limit before upload', () => {
  assert.doesNotThrow(() => validatePreparedBlob({ size: MAX_FILE_BYTES }));
  assert.throws(() => validatePreparedBlob({ size: MAX_FILE_BYTES + 1 }), /normalized PNG exceeds/);
  for (const blob of [null, { size: 0 }]) assert.throws(() => validatePreparedBlob(blob), /encode/);
});

test('image preparation preserves portrait and landscape aspect ratio without upscaling', () => {
  assert.deepEqual(fittedSize(8000, 4000), { width: 4096, height: 2048 });
  assert.deepEqual(fittedSize(4000, 8000), { width: 2048, height: 4096 });
  assert.deepEqual(fittedSize(400, 300), { width: 400, height: 300 });
  assert.deepEqual(fittedSize(1, 10000), { width: 1, height: 4096 });
});

test('boxes scale independently on both axes into the preview canvas', () => {
  assert.deepEqual(scaleBox([100, 200, 800, 900], 1000, 2000, 500, 500), [50, 50, 400, 225]);
});

const detection = { class_id: 2, class_name: '<img onerror=alert(1)>', confidence: 0.81, bbox: [10, 20, 90, 180] };
const response = (detections) => ({ image_width: 100, image_height: 200, detections });
test('valid predictions and empty detections preserve the API contract', () => {
  assert.deepEqual(validatePrediction(response([detection]), 100, 200).detections, [detection]);
  assert.deepEqual(validatePrediction(response([]), 100, 200).detections, []);
});

test('mismatched orientation, malformed scores and invalid boxes cannot be displayed', () => {
  assert.throws(() => validatePrediction(response([]), 200, 100));
  for (const patch of [
    { confidence: NaN }, { confidence: 1.1 }, { class_name: null }, { class_id: 1.5 },
    { bbox: [1, 2, 3] }, { bbox: [90, 20, 10, 180] }, { bbox: [-1, 20, 90, 180] },
    { bbox: [10, 20, 101, 180] }, { bbox: [10, 20, 90, Infinity] },
  ]) assert.throws(() => validatePrediction(response([{ ...detection, ...patch }]), 100, 200));
});
