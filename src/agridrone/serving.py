"""Bounded CPU inference service and browser demo.

The model is imported lazily and used by one worker at a time. Liveness never
loads it; readiness performs a real prediction because ONNX loading is lazy in
Ultralytics. All expensive image/model work runs outside the event loop.
"""

from __future__ import annotations

import asyncio
import io
import math
import os
import threading
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agridrone.config import load_config, project_root, resolve_path


class ServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_path: str = Field(min_length=1)
    imgsz: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    max_request_bytes: int = Field(gt=0)
    max_image_bytes: int = Field(gt=0)
    max_image_pixels: int = Field(gt=0)


settings = ServeSettings.model_validate(
    load_config(os.environ.get("AGRIDRONE_SERVE_CONFIG", "configs/serve.yaml"))
)
FRONTEND_DIR = project_root() / "frontend"
_model_lock = threading.Lock()
_model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agridrone-inference")
_model_ready = False
_T = TypeVar("_T")


class Detection(BaseModel):
    """A box in EXIF-oriented uploaded-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RequestSizeLimit:
    """Buffer at most the configured limit *before* multipart parsing.

    Counting actual ASGI bytes also covers chunked or dishonest Content-Length
    requests. No oversized request reaches the parser or creates upload files.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/predict":
            await self.app(scope, receive, send)
            return
        lengths = [value for key, value in scope["headers"] if key == b"content-length"]
        if lengths:
            try:
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError
                length = int(lengths[0])
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
                return
            if length > settings.max_request_bytes:
                await self._too_large(scope, receive, send)
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > settings.max_request_bytes:
                await self._too_large(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            nonlocal body
            message: Message = {"type": "http.request", "body": bytes(body), "more_body": False}
            body = bytearray()
            return message

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"detail": "Request body is too large"}, 413)(scope, receive, send)


app = FastAPI(title="AgriDrone inference API", version="0.1.0")
app.add_middleware(RequestSizeLimit)
app.mount(
    "/demo-assets", StaticFiles(directory=FRONTEND_DIR, check_dir=False), name="demo-assets"
)


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def demo() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(503, "Demo frontend is unavailable")
    return FileResponse(index, media_type="text/html")


def model_path() -> Path:
    return resolve_path(os.environ.get("AGRIDRONE_MODEL", settings.model_path))


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Called only while holding the inference slot; never downloads a model."""

    path = model_path()
    if not path.is_file():
        raise FileNotFoundError("Model artifact is unavailable")
    from ultralytics import YOLO

    return YOLO(str(path), task="detect")


def _prediction_to_response(result: Any, width: int, height: int) -> PredictionResponse:
    """Drop invalid detections and clip boxes to the oriented image bounds."""

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}
    detections: list[Detection] = []
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        for coords, score, class_value in zip(xyxy, conf, classes, strict=False):
            try:
                values = [float(value) for value in coords]
                score = float(score)
                class_value = float(class_value)
                if (
                    len(values) != 4
                    or not all(math.isfinite(value) for value in [*values, score, class_value])
                    or not 0 <= score <= 1
                    or class_value < 0
                    or not class_value.is_integer()
                ):
                    continue
                class_id = int(class_value)
                bounds = [width, height, width, height]
                bbox = [
                    round(min(max(value, 0), bound), 2)
                    for value, bound in zip(values, bounds, strict=True)
                ]
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue
                if isinstance(names, Mapping):
                    name = names.get(class_id, str(class_id))
                elif isinstance(names, list | tuple) and class_id < len(names):
                    name = names[class_id]
                else:
                    name = str(class_id)
                if not isinstance(name, str) or not name.strip():
                    name = str(class_id)
                detections.append(
                    Detection(
                        class_id=class_id, class_name=name, confidence=round(score, 6), bbox=bbox
                    )
                )
            except (TypeError, ValueError, OverflowError):
                continue
    return PredictionResponse(image_width=width, image_height=height, detections=detections)


def _decode_image(raw: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
                if image.width * image.height > settings.max_image_pixels:
                    raise HTTPException(413, "Image has too many pixels")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                oriented = ImageOps.exif_transpose(image)
                assert oriented is not None  # in_place=False returns an image.
                return oriented.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "Image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(400, "Uploaded file is not a valid image") from exc


def _infer(source: Image.Image, *, readiness: bool = False) -> PredictionResponse:
    global _model_ready
    try:
        model = get_model()
    except Exception as exc:
        _model_ready = False
        raise HTTPException(503, "Model is unavailable") from exc
    try:
        results = model.predict(
            source=source, imgsz=settings.imgsz, conf=settings.confidence,
            device="cpu", verbose=False,
        )
        # A valid detector returns one Results object, even with no detections.
        if len(results) != 1 or not hasattr(results[0], "boxes"):
            raise ValueError("Invalid detector result")
        response = _prediction_to_response(results[0], *source.size)
    except Exception as exc:
        _model_ready = False
        raise HTTPException(503 if readiness else 500, "Inference is unavailable") from exc
    _model_ready = True
    return response


def _predict_bytes(raw: bytes) -> PredictionResponse:
    with _decode_image(raw) as source:
        return _infer(source)


def _warmup() -> None:
    with Image.new("RGB", (settings.imgsz, settings.imgsz)) as source:
        _infer(source, readiness=True)


async def _run_exclusive(work: Callable[..., _T], *args: Any) -> _T:
    # Claim before submitting: the executor cannot accumulate a waiting queue.
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(503, "Model is busy; retry shortly", headers={"Retry-After": "1"})
    try:
        future = _model_executor.submit(work, *args)
    except Exception:
        _model_lock.release()
        raise
    # A disconnected/cancelled caller must not release the slot while its CPU
    # worker still uses the model. Release on actual worker completion instead.
    future.add_done_callback(lambda _: _model_lock.release())
    return await asyncio.wrap_future(future)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness is cheap and independent of both the model and inference slot."""

    return HealthResponse(status="ok", model_loaded=bool(get_model.cache_info().currsize))


@app.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Prove the configured model works once; failures remain retryable."""

    if not _model_ready:
        await _run_exclusive(_warmup)
    return HealthResponse(status="ready", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:  # noqa: B008
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
    raw = await file.read(settings.max_image_bytes + 1)
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(413, "Image file is too large")
    return await _run_exclusive(_predict_bytes, raw)
