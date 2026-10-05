"""HTTP contracts using deterministic fakes; no model downloads or ML imports."""

import asyncio
import importlib
import json
import threading
from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agridrone import serving as main


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _result(coords=None, scores=None, classes=None, names=None):
    return SimpleNamespace(
        names={3: "Tomato leaf late blight"} if names is None else names,
        boxes=SimpleNamespace(
            xyxy=_Tensor([[1.234, 2.345, 40.0, 50.0]] if coords is None else coords),
            conf=_Tensor([0.87654321] if scores is None else scores),
            cls=_Tensor([3] if classes is None else classes),
        ),
    )


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.results = [_result()]
        self.error = None

    def predict(self, **kwargs):
        self.calls.append({**kwargs, "source_size": kwargs["source"].size})
        if self.error is not None:
            raise self.error
        return self.results


@pytest.fixture()
def model(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(main, "get_model", lru_cache(maxsize=1)(lambda: fake))
    monkeypatch.setattr(main, "_model_ready", False)
    monkeypatch.setattr(main, "settings", main.settings.model_copy())
    return fake


@pytest.fixture()
def client(model):
    with TestClient(main.app) as client:
        yield client


def _image(*, size=(80, 60), format="PNG", exif=None):
    payload = BytesIO()
    with Image.new("RGB", size, color="green") as image:
        kwargs = {"exif": exif} if exif is not None else {}
        image.save(payload, format=format, **kwargs)
    return payload.getvalue()


def _upload(client, raw=None, content_type="image/png"):
    return client.post(
        "/predict", files={"file": ("leaf", _image() if raw is None else raw, content_type)}
    )


def test_compatibility_entrypoint():
    assert importlib.import_module("api.main").app is main.app


def test_default_limits_cover_browser_normalization():
    # Browser prepares images up to 4096 square and validates a 20 MiB PNG.
    assert main.settings.max_image_bytes == 20 * 1024 * 1024
    assert main.settings.max_request_bytes > main.settings.max_image_bytes
    assert main.settings.max_image_pixels >= 4096 * 4096


def test_health_is_liveness_without_loading_model(client, model):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}
    assert main.get_model.cache_info().currsize == 0
    assert model.calls == []


def test_demo_assets_and_docs(client):
    assert client.get("/").status_code == 200
    assert client.get("/demo-assets/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/predict", "/health", "/ready"} <= schema["paths"].keys()


def test_missing_demo_does_not_expose_local_path(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "FRONTEND_DIR", tmp_path / "secret-folder")
    response = client.get("/")
    assert response.status_code == 503
    assert response.json() == {"detail": "Demo frontend is unavailable"}


def test_predict_returns_pixel_boxes_and_class_name(client, model):
    response = _upload(client)
    assert response.status_code == 200
    assert response.json() == {
        "image_width": 80,
        "image_height": 60,
        "detections": [{
            "class_id": 3,
            "class_name": "Tomato leaf late blight",
            "confidence": 0.876543,
            "bbox": [1.23, 2.35, 40.0, 50.0],
        }],
    }
    call = model.calls[0]
    assert call["imgsz"] == 640
    assert call["conf"] == 0.25
    assert call["device"] == "cpu"
    assert call["source"].mode == "RGB"


def test_prediction_honors_config(client, model):
    main.settings.imgsz = 320
    main.settings.confidence = 0.4
    assert _upload(client).status_code == 200
    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["conf"] == 0.4


def test_model_environment_override(monkeypatch, tmp_path):
    path = tmp_path / "custom.onnx"
    monkeypatch.setenv("AGRIDRONE_MODEL", str(path))
    assert main.model_path() == path


def test_ready_requires_successful_warmup_and_is_cached(client, model):
    for _ in range(2):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "model_loaded": True}
    assert len(model.calls) == 1
    assert model.calls[0]["source_size"] == (640, 640)
    assert main.get_model.cache_info().misses == 1
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_successful_prediction_also_proves_readiness(client, model):
    assert _upload(client).status_code == 200
    assert client.get("/ready").status_code == 200
    assert len(model.calls) == 1


@pytest.mark.parametrize("error", [FileNotFoundError, ImportError, RuntimeError])
def test_model_load_failures_are_unavailable_without_internal_paths(client, monkeypatch, error):
    @lru_cache(maxsize=1)
    def fail():
        raise error("/private/path/secret.onnx")

    monkeypatch.setattr(main, "get_model", fail)
    for response in [client.get("/ready"), _upload(client)]:
        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}
    assert client.get("/health").status_code == 200


def test_failed_lazy_initialization_is_not_ready_and_can_recover(client, model):
    model.error = RuntimeError("ONNX load failed at /private/model.onnx")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "Inference is unavailable"}
    assert not main._model_ready
    assert client.get("/health").status_code == 200
    model.error = None
    assert client.get("/ready").status_code == 200
    assert len(model.calls) == 2


def test_inference_failure_clears_readiness_and_hides_exception(client, model):
    assert client.get("/ready").status_code == 200
    model.error = ValueError("/private/user/data")
    response = _upload(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "Inference is unavailable"}
    assert not main._model_ready
    assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("results", [[], [None]])
def test_invalid_model_contract_is_not_ready(client, model, results):
    model.results = results
    assert client.get("/ready").status_code == 503
    assert not main._model_ready


@pytest.mark.parametrize("boxes", [None, SimpleNamespace(
    xyxy=_Tensor([]), conf=_Tensor([]), cls=_Tensor([])
)])
def test_empty_detection_result_is_valid(client, model, boxes):
    model.results = [SimpleNamespace(boxes=boxes, names={})]
    assert _upload(client).json() == {
        "image_width": 80, "image_height": 60, "detections": []
    }


def test_prediction_sanitizes_invalid_rows_and_clips_boxes(client, model):
    model.results = [_result(
        coords=[[-5, -2, 200, 100], [0, 0, float("nan"), 10], [0, 0, 10, 10],
                [0, 0, 10, 10], [20, 0, 10, 10], [0, 0, 10], [0, 0, 10, 10],
                [0, 0, 10, 10], [0, 0, 10, 10]],
        scores=[0.9, 0.9, float("inf"), 1.1, 0.9, 0.9, 0.9, 0.9, 0.9],
        classes=[0, 0, 0, 0, 0, 0, -1, 0.5, float("nan")],
        names=["leaf"],
    )]
    response = _upload(client)
    assert response.status_code == 200
    assert response.json()["detections"] == [{
        "class_id": 0, "class_name": "leaf", "confidence": 0.9, "bbox": [0, 0, 80, 60]
    }]


def test_missing_class_name_uses_id_and_incomplete_rows_are_ignored(client, model):
    model.results = [_result(coords=[[0, 0, 10, 10], [1, 2, 3, 4]], names={})]
    detections = _upload(client).json()["detections"]
    assert len(detections) == 1
    assert detections[0]["class_name"] == "3"


def test_exif_orientation_defines_response_and_model_dimensions(client, model):
    exif = Image.Exif()
    exif[274] = 6
    response = _upload(client, _image(format="JPEG", exif=exif), "image/jpeg")
    assert response.status_code == 200
    assert (response.json()["image_width"], response.json()["image_height"]) == (60, 80)
    assert model.calls[0]["source_size"] == (60, 80)


@pytest.mark.parametrize("format,content_type", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_supported_image_formats(client, format, content_type):
    assert _upload(client, _image(format=format), content_type).status_code == 200


def test_wrong_content_type(client, model):
    assert _upload(client, b"anything", "text/plain").status_code == 415
    assert model.calls == []


def test_unsupported_image_format_with_spoofed_mime(client):
    assert _upload(client, _image(format="GIF"), "image/png").status_code == 415


@pytest.mark.parametrize("raw", [b"", b"not-an-image", _image()[:40]])
def test_malformed_images(client, model, raw):
    response = _upload(client, raw)
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image"}
    assert model.calls == []


def test_missing_file_and_invalid_multipart(client):
    assert client.post("/predict").status_code == 422
    assert client.post("/predict", content=b"broken", headers={
        "content-type": "multipart/form-data"
    }).status_code == 400


def test_file_byte_limit_accepts_exact_boundary(client):
    raw = _image()
    main.settings.max_image_bytes = len(raw)
    assert _upload(client, raw).status_code == 200
    main.settings.max_image_bytes -= 1
    assert _upload(client, raw).status_code == 413


def test_pixel_limit_accepts_exact_boundary(client):
    main.settings.max_image_pixels = 80 * 60
    assert _upload(client).status_code == 200
    main.settings.max_image_pixels -= 1
    assert _upload(client).status_code == 413


@pytest.mark.parametrize("pillow_limit", [3000, 1000])
def test_decompression_bomb_warnings_and_errors_are_rejected(client, monkeypatch, pillow_limit):
    raw = _image()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    assert _upload(client, raw).status_code == 413


def test_request_byte_limit_accepts_exact_boundary(client):
    request = client.build_request("POST", "/predict", files={
        "file": ("leaf.png", _image(), "image/png")
    })
    body = request.read()
    main.settings.max_request_bytes = len(body)
    assert client.post("/predict", content=body, headers=request.headers).status_code == 200
    main.settings.max_request_bytes -= 1
    assert client.post("/predict", content=body, headers=request.headers).status_code == 413


@pytest.mark.parametrize("length", [None, b"1", b"1000"])
def test_oversized_bodies_are_rejected_before_multipart_parsing(model, length):
    main.settings.max_request_bytes = 64
    consumed = []
    sent = []

    async def downstream(scope, receive, send):
        pytest.fail("Oversized body reached the multipart parser")

    async def receive():
        consumed.append(True)
        return {"type": "http.request", "body": b"x" * 40, "more_body": True}

    async def send(message):
        sent.append(message)

    headers = [(b"content-type", b"multipart/form-data; boundary=leaf")]
    if length is not None:
        headers.append((b"content-length", length))
    asyncio.run(main.RequestSizeLimit(downstream)(
        {"type": "http", "path": "/predict", "headers": headers}, receive, send
    ))
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "Request body is too large"}
    assert len(consumed) == (0 if length == b"1000" else 2)


@pytest.mark.parametrize("length", ["-1", "invalid", "1, 1"])
def test_invalid_content_length(client, length):
    response = client.post("/predict", content=b"x", headers={"content-length": length})
    assert response.status_code == 400


def test_health_and_busy_rejection_during_inference_on_same_event_loop(model, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = model.predict

    def blocking_predict(**kwargs):
        entered.set()
        assert release.wait(5), "Test failed to release inference"
        return original(**kwargs)

    monkeypatch.setattr(model, "predict", blocking_predict)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/predict", files={
                "file": ("leaf.png", _image(), "image/png")
            }))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                health = await asyncio.wait_for(client.get("/health"), timeout=1)
                assert health.status_code == 200
                busy = await asyncio.wait_for(client.post("/predict", files={
                    "file": ("leaf.png", _image(), "image/png")
                }), timeout=1)
                assert busy.status_code == 503
                assert busy.headers["retry-after"] == "1"
                assert (await client.get("/ready")).status_code == 503
            finally:
                release.set()
                result = await first
            assert result.status_code == 200
            assert (await client.get("/ready")).status_code == 200
            assert len(model.calls) == 1
            assert main.get_model.cache_info().misses == 1

    asyncio.run(exercise())


def test_cancelled_caller_keeps_inference_slot_until_worker_finishes(model):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_work():
        entered.set()
        try:
            assert release.wait(5)
        finally:
            finished.set()

    async def exercise():
        first = asyncio.create_task(main._run_exclusive(blocking_work))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            with pytest.raises(main.HTTPException) as busy:
                await main._run_exclusive(lambda: None)
            assert busy.value.status_code == 503
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            # Ensure the executor's completion callback has released admission.
            await asyncio.wrap_future(main._model_executor.submit(lambda: None))
        await main._run_exclusive(lambda: None)

    asyncio.run(exercise())
