# AgriDrone — Crop Disease Detection + Inference Optimization

Fine-tune a YOLO26 object detector to **detect and localize crop diseases** in
leaf images, then prove it is genuinely deployable by exporting to ONNX,
quantizing (FP16 / INT8), and benchmarking real inference latency — not just
training accuracy.

> **Status:** Phase 1 (data pipeline + EDA) done. Phase 2 (training) code is
> implemented and runs on Colab GPU — see [Phase 2](#phase-2--training). No
> training run has been executed yet (this dev machine has no GPU).
> Later phases (quantization/benchmark, API, demo, docs) are tracked below.

## Honest task framing

This project uses **[PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)**,
a genuine object-detection dataset with real bounding boxes on in-the-wild
images. After conversion and class exclusion (below) this project trains on
**27 classes** (healthy-leaf + disease categories spanning ~13 plant species;
CC-BY-4.0). The class distribution is heavily skewed — see
[`reports/eda_report.md`](reports/eda_report.md).

It deliberately does **not** use PlantVillage *for detection*. PlantVillage is a
*classification-only* dataset — every image is a single leaf on a plain
background with no bounding boxes — so training a "detector" on it would just
learn "the leaf is in the middle of the frame." That would be a dishonest
localization story. (PlantVillage *is* used to pretrain the detector's backbone
as a classifier — see [Phase 2](#phase-2--training).
That's an honest use of classification data: it improves learned features
without ever pretending to supply localization labels.) PlantDoc is smaller and
noisier, so mAP will be modest, and that number is reported as-is rather than
cherry-picked.

### Excluded classes

Two of PlantDoc's 29 raw classes are dropped before training, for
**evaluation-integrity** reasons rather than mere imbalance:

| Class | Train boxes | Test boxes | Why dropped |
|-------|:-----------:|:----------:|-------------|
| `Tomato two spotted spider mites leaf` | 1 | 0 | Effectively no training signal; **zero test representation** |
| `Potato leaf` | 10 | 0 | Negligible training signal; **zero test representation** |

The deciding factor is the zero test-set count: with no test boxes there is no
honest way to measure detection performance on these classes, so reporting a
number for them would be misleading. Dropping them leaves **27 classes**. The
exclusion list lives in [`configs/data.yaml`](configs/data.yaml)
(`excluded_classes`) and the conversion counts the dropped boxes explicitly
(see the "Dropped (excluded class)" column in the EDA report). Note that the
generic `Potato leaf` is removed while the specific `Potato leaf early blight`
and `Potato leaf late blight` disease classes are kept.

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

## Phase 2 — training

Training code lives in [`agridrone.train`](src/agridrone/train.py) and is driven
by [`configs/train.yaml`](configs/train.yaml). It needs a GPU and the full ML
stack, so it is run on Colab via
[`notebooks/phase2_colab.ipynb`](notebooks/phase2_colab.ipynb). No training run
has been executed yet — this dev machine has no GPU. The module imports cleanly
without torch/ultralytics (the ML imports are guarded) so the logic stays
unit-tested here.

```bash
# On a GPU box (or Colab) after `make setup`:
make train-pretrain   # optional PlantVillage backbone pretrain
make train-finetune   # PlantDoc detector fine-tune (weighted oversampling)
make train-baseline   # vanilla YOLOv8 baseline for comparison
make train-eval       # evaluate on TEST -> reports/phase2_results.md
```

**Colab data cache.** Re-downloading + re-converting PlantDoc (~995 MB) every
Colab session is wasteful. Convert once and cache the processed dataset to Google
Drive, then point `data.drive_dataset_dir` in the config at it:

```bash
python -m agridrone.data --drive-cache /content/drive/MyDrive/agridrone/plantdoc
```

`sync_to_drive` rewrites the `dataset.yaml` `path:` line to the destination so
Ultralytics resolves images wherever the folder is mounted.

### Pretrain-then-finetune (PlantVillage → PlantDoc)

The detector backbone is first pretrained as an image **classifier** on
**PlantVillage**, then the full detection model is fine-tuned on PlantDoc.

*Why:* PlantDoc is small (~2.6k images) and noisy, which limits how good the
backbone's learned features can get from detection data alone. PlantVillage is
large and covers closely related leaf/disease imagery, so using it to warm up
the backbone gives the detector better low- and mid-level features before it
ever sees a bounding box. Fine-tuning then adapts those features to the
detection task on real, in-the-wild PlantDoc images. This is a deliberate design
choice, not a shortcut — PlantVillage supplies *feature* pretraining only and is
never treated as a source of localization labels.

### Class imbalance — weighted oversampling + per-class reporting

The kept 27 classes are heavily imbalanced (e.g. `Blueberry leaf` ~718 train
boxes vs `Corn Gray leaf spot` ~66). Approach:

- **Oversample images containing rare classes** rather than reweighting the loss.
  Ultralytics detection (8.4.x) exposes no built-in class-weighted-loss flag for
  the `Detect` task, and hard loss-weighting can cause gradient spikes when a
  rare class finally lands in a batch. Instead a `YOLODataset` subclass computes
  per-image sampling probabilities from inverse class frequency and draws rare
  images more often during training only (val/test loading is untouched, so
  evaluation stays honest). The balancing math is pure numpy in
  [`agridrone.weighting`](src/agridrone/weighting.py) and fully unit-tested; the
  dataset subclass is monkey-patched into `ultralytics.data.build` at train time.
- Report **per-class AP** in the final results table alongside aggregate
  mAP@0.5 / mAP@0.5:0.95, so tail-class performance is visible and honestly
  reported rather than hidden inside an average.

## Testing

```bash
make test   # runs the pytest suite
```

## Roadmap

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Data pipeline + EDA (27 classes after exclusion) | done |
| 2 | PlantVillage backbone pretrain → PlantDoc fine-tune, weighted oversampling, per-class AP (Colab GPU) | code done, run pending (no local GPU) |
| 3 | ONNX export, FP16/INT8 quantization, latency benchmark | pending |
| 4 | FastAPI inference service | pending |
| 5 | Demo frontend | pending |
| 6 | Docs, tests, CI hardening | pending |

## License / attribution

PlantDoc dataset © its authors, released under **CC-BY-4.0**. See the
[dataset repo](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
for citation details.
