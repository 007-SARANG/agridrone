# CPU-only Linux deployment. Artifacts are supplied as read-only runtime mounts.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    AGRIDRONE_MODEL=/app/models/model.onnx

# OpenCV (an Ultralytics dependency) requires these even in CPU-only mode.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app

WORKDIR /app
COPY requirements-runtime.lock.txt ./
RUN python -m pip install --no-cache-dir pip==25.3 setuptools==75.6.0 \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
       torch==2.5.1+cpu torchvision==0.20.1+cpu \
    && python -m pip install --no-cache-dir -r requirements-runtime.lock.txt \
    && python -m pip check

COPY pyproject.toml README.md ./
COPY src/ src/
COPY api/ api/
COPY configs/serve.yaml configs/serve.yaml
COPY frontend/ frontend/
COPY docker/healthcheck.py /app/docker/healthcheck.py
# Editable installation preserves repository-relative configs and frontend paths.
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && mkdir -p /app/models

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py"]
CMD ["python", "-m", "uvicorn", "agridrone.serving:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
