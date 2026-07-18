# AgriDrone — Crop Disease Detection + Inference Optimization

Fine-tune a YOLO26 object detector to **detect and localize crop diseases** in
leaf images, then prove it is genuinely deployable by exporting to ONNX,
quantizing (FP16 / INT8), and benchmarking real inference latency — not just
training accuracy.

> **Status:** Phase 1 (data pipeline + EDA) in progress. Later phases (training,
> quantization/benchmark, API, demo, docs) are tracked below.

## Honest task framing

This project uses **[PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)**,
a genuine object-detection dataset with real bounding boxes on in-the-wild
images. After conversion this project sees **2,578 images across 29 classes**
(healthy-leaf + disease categories spanning ~13 plant species; CC-BY-4.0). The
class distribution is heavily skewed — see [`reports/eda_report.md`](reports/eda_report.md).

It deliberately does **not** use PlantVillage for detection. PlantVillage is a
*classification-only* dataset — every image is a single leaf on a plain
background with no bounding boxes — so training a "detector" on it would just
learn "the leaf is in the middle of the frame." That would be a dishonest
localization story. PlantDoc is smaller and noisier, so mAP will be modest, and
that number is reported as-is rather than cherry-picked.

## Repository layout

```
src/agridrone/     # all production code (data, train, evaluate, export, infer, benchmark)
configs/           # YAML configs — no hardcoded paths or magic numbers in code
data/              # raw/ + processed/ (gitignored; created by the download script)
notebooks/         # Colab training notebooks (exploration + GPU training only)
api/               # FastAPI inference service (Phase 4)
frontend/          # demo UI (Phase 5)
tests/             # pytest suite
reports/           # EDA report + benchmark results (committed)
docker/            # Dockerfile + compose (Phase 6)
.github/workflows/ # CI
```

## Environment setup

Requires **Python 3.11+**. Training runs on **Colab GPU** (this dev machine has
no dedicated GPU); the laptop is used for CPU-side benchmarking.

### Standard setup

```bash
make setup        # create .venv and install everything (requirements-dev.txt)
```

### Debian/Ubuntu note (PEP 668 / externally-managed Python)

If `python3 -m venv` fails with an `ensurepip`/externally-managed error, install
the venv package once, then set up:

```bash
sudo apt install python3.12-venv     # one-time, needs sudo
make setup
```

If you cannot use `apt`, bootstrap pip into an isolated venv without touching
system Python:

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
.venv/bin/python -m pip install -r requirements-dev.txt
```

> **ROS users:** if your shell sources a ROS 2 workspace, it exports
> `PYTHONPATH` and leaks system packages into the venv. Every `make` target
> already clears `PYTHONPATH`; run project commands via `make` (or prefix with
> `PYTHONPATH=`) to avoid this.

### Lightweight setup (data pipeline + tests only, no PyTorch)

```bash
make setup-min    # pandas/numpy/Pillow/matplotlib/PyYAML/pytest only
```

## Phase 1 — data pipeline

```bash
make data   # download PlantDoc, convert CSV boxes -> YOLO format, split train/val/test
make eda    # write reports/eda_report.md + plots
```

- `agridrone.data` downloads the dataset, parses the Pascal-VOC-style CSV
  annotations, converts boxes to normalized YOLO `class cx cy w h` labels,
  copies images into the `images/train|val|test` + `labels/...` layout
  Ultralytics expects, carves a deterministic 15% validation split out of TRAIN
  (TEST is left untouched), and emits a `dataset.yaml`.
- `agridrone.eda` produces class-balance, image-size, boxes-per-image, and
  annotated-sample plots plus a markdown report.

Config lives in [`configs/data.yaml`](configs/data.yaml).

## Testing

```bash
make test   # runs the pytest suite
```

## Roadmap

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Data pipeline + EDA | in progress |
| 2 | YOLO26 fine-tune + YOLOv8 baseline (Colab GPU) | pending |
| 3 | ONNX export, FP16/INT8 quantization, latency benchmark | pending |
| 4 | FastAPI inference service | pending |
| 5 | Demo frontend | pending |
| 6 | Docs, tests, CI hardening | pending |

## License / attribution

PlantDoc dataset © its authors, released under **CC-BY-4.0**. See the
[dataset repo](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
for citation details.
