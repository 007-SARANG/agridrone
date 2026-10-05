export const MAX_FILE_BYTES = 20 * 1024 * 1024;
export const MAX_EDGE = 4096;

export function validateFile(file) {
  if (!file || !file.size) throw new Error('Choose a non-empty JPEG, PNG, or WebP image.');
  if (file.size > MAX_FILE_BYTES) throw new Error('This image exceeds 20 MiB. Choose a smaller image.');
  const accepted = ['image/jpeg', 'image/png', 'image/webp'];
  if (file.type ? !accepted.includes(file.type) : !/\.(jpe?g|png|webp)$/i.test(file.name)) {
    throw new Error('Choose a JPEG, PNG, or WebP image.');
  }
}

export function validatePreparedBlob(blob) {
  if (!blob || !blob.size) throw new Error('Your browser could not encode this image.');
  if (blob.size > MAX_FILE_BYTES) {
    throw new Error('The normalized PNG exceeds the 20 MiB upload limit. Choose a smaller image or reduce its dimensions.');
  }
}

export function fittedSize(width, height, limit = MAX_EDGE) {
  const ratio = Math.min(1, limit / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * ratio)), height: Math.max(1, Math.round(height * ratio)) };
}

export function validatePrediction(data, width, height) {
  if (!data || data.image_width !== width || data.image_height !== height || !Array.isArray(data.detections)) {
    throw new Error('The server returned unexpected image dimensions or prediction data. Please try again.');
  }
  for (const detection of data.detections) {
    if (!detection || !Number.isInteger(detection.class_id) || typeof detection.class_name !== 'string' ||
        !Number.isFinite(detection.confidence) || detection.confidence < 0 || detection.confidence > 1 ||
        !Array.isArray(detection.bbox) || detection.bbox.length !== 4 || !detection.bbox.every(Number.isFinite) ||
        detection.bbox[2] <= detection.bbox[0] || detection.bbox[3] <= detection.bbox[1] ||
        detection.bbox[0] < 0 || detection.bbox[1] < 0 || detection.bbox[2] > width || detection.bbox[3] > height) {
      throw new Error('The server returned invalid detection data. Please try again.');
    }
  }
  return data;
}

export function scaleBox(box, imageWidth, imageHeight, canvasWidth, canvasHeight) {
  return box.map((coordinate, index) => coordinate * (index % 2 ? canvasHeight / imageHeight : canvasWidth / imageWidth));
}
