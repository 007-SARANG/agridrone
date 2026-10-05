"""Opt-in real-artifact smoke check; never downloads weights or measures accuracy."""

import os
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from agridrone import serving


@pytest.mark.skipif(
    os.environ.get("AGRIDRONE_REAL_MODEL_TEST") != "1",
    reason="Opt in with AGRIDRONE_REAL_MODEL_TEST=1 and a local ONNX artifact",
)
def test_real_onnx_upload_to_detection():
    assert serving.model_path().is_file(), "Provide the existing trained ONNX artifact"
    payload = BytesIO()
    sample = os.environ.get("AGRIDRONE_SMOKE_IMAGE")
    if sample:
        with Image.open(Path(sample)) as image:
            oriented = ImageOps.exif_transpose(image)
            assert oriented is not None
            source = oriented.convert("RGB")
    else:
        source = Image.new("RGB", (96, 64), color="green")
    with source:
        source.save(payload, format="PNG")
        width, height = source.size

    serving.get_model.cache_clear()
    serving._model_ready = False
    try:
        with TestClient(serving.app) as client:
            assert client.get("/").status_code == 200
            for asset in ["app.js", "core.mjs", "styles.css"]:
                assert client.get(f"/demo-assets/{asset}").status_code == 200
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            response = client.post(
                "/predict", files={"file": ("sample.png", payload.getvalue(), "image/png")}
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert (result["image_width"], result["image_height"]) == (width, height)
            for detection in result["detections"]:
                assert 0 <= detection["class_id"] < 27
                assert detection["class_name"]
                assert 0 <= detection["confidence"] <= 1
                x1, y1, x2, y2 = detection["bbox"]
                assert 0 <= x1 < x2 <= width
                assert 0 <= y1 < y2 <= height
    finally:
        serving.get_model.cache_clear()
        serving._model_ready = False
