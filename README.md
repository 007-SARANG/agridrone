# AgriDrone — Crop Disease Detection + Inference Optimization

Fine-tune a YOLO26 object detector to **detect and localize crop diseases** in
leaf images, then prove it is genuinely deployable by exporting to ONNX,
quantizing (FP16 / INT8), and benchmarking real inference latency — not just
training accuracy.

> **Status:** All six implementation phases are delivered: data, training,
> measured ONNX benchmarking, API, browser demo, and deployment/CI tooling.
> Real ONNX inference has been checked locally. Docker execution still requires
> verification on a Docker-enabled host; see [verification](reports/completion.md).

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
localization story. (The pipeline supports optional PlantVillage pretraining of
the detector's backbone as a classifier — see [Phase 2](#phase-2--training).
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
api/               # compatibility shim; implementation in src/agridrone/serving.py
frontend/          # static demo UI and Node regression tests (Phase 5)
tests/             # pytest suite
reports/           # EDA report + benchmark results (committed)
docker/            # Dockerfile + compose (Phase 6)
.github/workflows/ # CI
```

## Environment setup

Use **Python 3.11 or 3.12**, `uv`, Make, and Node.js 22+ for frontend tests.
Python 3.14 is incompatible with the pinned PyTorch stack. Training uses Colab
GPU; serving and benchmark verification use CPU.

### Arch Linux / fish quick start

Install missing tools with `sudo pacman -S uv make nodejs`. Then, from this repo:

```fish
uv python install 3.12
make setup
and make check
and make api
```

Open **http://localhost:8000/**. `make api` stays in the foreground; Ctrl-C stops it.
No environment activation is needed. `uv` supplies the interpreter independently
of Arch's system Python, and setup repairs missing pip inside the venv. An
existing wrong-version venv is never deleted: use `make setup VENV=.venv312`
and pass `VENV=.venv312` to subsequent Make commands instead.

For lightweight development without the ML stack, use `make setup-min` then
`make check`. The demo and API start, but predictions need the ML runtime and
trained model. Full setup installs the pinned training/dev lock; Docker uses a
separate, CPU-only runtime lock. Locks include transitive dependencies; update
them intentionally with `uv pip compile` rather than broad upgrades.

> **ROS users:** all Make Python commands clear `PYTHONPATH`. For direct commands
> in either fish or Bash, use `env PYTHONPATH= .venv/bin/python ...`.

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
[`notebooks/phase2_colab.ipynb`](notebooks/phase2_colab.ipynb). The trained
detector and recorded Phase 3 evaluation exist (PyTorch mAP@0.5 = 0.6015).
The module imports without torch/ultralytics so helper logic stays unit-tested.
The repository does not contain a Phase 2 per-class/baseline comparison report;
optional pretraining and baseline commands below are supported workflows, not
evidence that every experiment was run. Preserve the Colab run artifacts when
making claims about their provenance.

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

The optional workflow first pretrains the detector backbone as an image
**classifier** on **PlantVillage**, then fine-tunes the detector on PlantDoc.

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
  per-image sampling probabilities from (dampened, capped) inverse class
  frequency and draws rare images more often during training only — see the next
  bullet for the dampening/cap knobs (val/test loading is untouched, so
  evaluation stays honest). The balancing math is pure numpy in
  [`agridrone.weighting`](src/agridrone/weighting.py) and fully unit-tested; the
  dataset subclass is monkey-patched into `ultralytics.data.build` at train time.
- **Dampen and cap the weights** to avoid overfitting the tail. Pure inverse
  frequency (`weight = total / count`) fully equalizes classes, but on a
  long-tailed set that means the rarest classes are drawn from the same handful
  of images over and over — the model memorizes them instead of generalizing.
  Two config knobs (`configs/train.yaml → weighting`) control this:
  - `power` (default `0.5`) — the inverse-frequency exponent.
    `(total / count) ** power`. `1.0` is pure inverse frequency (full
    equalization, most aggressive); `0.5` is **sqrt dampening**, which still
    favors rare classes but far more gently; `0.0` disables balancing. The
    tradeoff: lower `power` reduces overfitting risk on tiny classes at the cost
    of leaving some imbalance in place.
  - `max_ratio` (default `5.0`) — a hard cap so the rarest class is never
    weighted more than `max_ratio ×` the most common present class, bounding the
    realized amplification regardless of how extreme the raw skew is. `null`
    disables the cap.

  On PlantDoc specifically, `mean` aggregation already tempers amplification
  (rare-class images usually contain common classes too), so the realized max
  drops from ~3.2× (pure, uncapped) to ~1.9× with the defaults — the `max_ratio`
  cap is a guardrail that only binds under harder skew or `agg: max`. The active
  `power`/`max_ratio` and the max realized amplification are printed in the
  results report so the settings are on record with the numbers.
- Report **per-class AP** in the final results table alongside aggregate
  mAP@0.5 / mAP@0.5:0.95, so tail-class performance is visible and honestly
  reported rather than hidden inside an average.
- **Quantify the oversampling effect**, so it is a number rather than trusted
  code. After fine-tuning, `reports/phase2_results.md` gains a *raw vs effective
  per-class exposure* table: raw box counts next to the expected boxes/epoch
  under the sampler's own probability vector (an amplification factor `x` per
  class). Because a uniform sampler would reproduce the raw counts exactly, `x`
  is derived from the very distribution `np.random.choice` samples from — rare
  classes show `x > 1`, common classes `x < 1`. Computed by
  [`expected_class_exposure`](src/agridrone/weighting.py) (pure numpy, unit-tested).

## Phase 3 — ONNX export + CPU benchmark

```bash
make export    # export best.pt → ONNX FP32 / FP16 / INT8
make benchmark # run latency + accuracy benchmark, write reports/phase3_benchmark.md
```

Config lives in [`configs/export.yaml`](configs/export.yaml).

All four precisions are benchmarked on CPU (single image, 640×640, 100 timed
runs after 10 warmup). Real measured results:

| Precision | Latency mean (ms) | Speedup | mAP@0.5 |
|-----------|------------------:|--------:|--------:|
| PyTorch FP32 | 84.35 | 1.00× | 0.6015 |
| ONNX FP32 | 32.92 | 2.56× | 0.5839 |
| ONNX FP16 | 55.63 | 1.52× | 0.5847 |
| ONNX INT8 | 74.48 | 1.13× | 0.0000 |

This INT8 export collapses to 0.0 mAP and is slower than ONNX FP32.
That makes this artifact unsuitable for deployment; it does not prove that
every YOLO26 quantization approach fails. The cause requires further investigation.
The benchmark code auto-annotates non-viable rows and emits a recommendation:
**Recommended deployment artifact: ONNX FP32** (2.56× faster than PyTorch,
accuracy within 0.018 mAP@0.5).

## Testing

```bash
make test   # runs the pytest suite
make test-frontend
make lint
make typecheck
# or all of the above:
make check
```

Opt-in real model smoke (no downloads; uses a synthetic image by default):

```fish
env PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 AGRIDRONE_REAL_MODEL_TEST=1 .venv/bin/python -m pytest tests/test_serving_integration.py
```

Set `AGRIDRONE_SMOKE_IMAGE` to a local crop photo to check that image instead.
This checks the inference contract, **not accuracy**. The regular suite uses
fakes and tests malformed uploads, EXIF, limits, readiness, concurrency and
cancellation; frontend tests exercise mocked DOM upload/filter/error flows.

## Phase 4 — FastAPI inference service

The API serves the recommended ONNX FP32 artifact and loads it lazily on the
first prediction request. Start it with:

```bash
make api
```

- `GET /health` is cheap liveness; HTTP 200 does **not** prove model availability.
- `GET /ready` warms up the model once and returns 200 only after inference works.
- `POST /predict` accepts a JPEG, PNG, or WebP upload in the `file` field and
  returns image dimensions plus detections (`class_id`, `class_name`,
  `confidence`, and pixel-space `[x1, y1, x2, y2]` boxes).
- Set `AGRIDRONE_MODEL` to override the default model artifact path.
- YAML settings live in `configs/serve.yaml`; select another with
  `AGRIDRONE_SERVE_CONFIG`. Restart the server after changing settings/artifacts.
- Defaults: 20 MiB file, 21 MiB total request, 4096² image pixels, confidence 0.25.
  Oversized input returns 413, bad image 400, unsupported format 415, missing file
  422, unavailable/busy model 503, and inference failure 500. Busy replies carry
  `Retry-After: 1`. One worker serializes inference without blocking health checks.

The exported model is intentionally not committed because model artifacts are
ignored by git. Run Phase 3 export first, or place the artifact at
`models/onnx/plantdoc_yolo26_best_fp32.onnx`.

## Phase 5 — Browser demo

Opening `http://localhost:8000/` after `make api` shows the demo UI. It supports
image preparation, loading status, scaled boxes, confidence scores, malformed
responses, and clear API/network errors. The 25–100% confidence slider filters
cached results without more inference and keeps boxes/list numbering aligned.
It cannot recover detections below the API's 0.25 floor. Keep the default API
floor/limits for this demo, or update its controls alongside custom configuration.

Photos are oriented and resized to at most 4096 pixels on the longest edge,
then encoded as PNG before upload. Coordinates refer to these prepared pixels.
Uploads are not persisted by application code. No frontend build is needed.
Scores are not calibrated disease probabilities; conflicting labels and missed
detections remain possible. An empty result does not mean a plant is healthy.
This is a portfolio demonstration, not agronomic advice or an autonomous drone.

## Phase 6 — CPU deployment and CI

With Docker Engine access and the Compose plugin, from the repository root:

```fish
docker compose version
make docker-up
docker compose -f docker/compose.yaml ps
curl --fail http://localhost:8000/ready
```

Open http://localhost:8000/ and upload a crop photo. `make docker-down` stops it.
Stop a locally running `make api` first to free port 8000. On Arch the Compose
package is `docker-compose`; Docker daemon setup/access is managed by your host
administrator (membership in the Docker group grants root-equivalent access).

Compose requires the existing ONNX file and mounts it **read-only**. Override it
in fish with `set -x AGRIDRONE_MODEL_HOST /absolute/path/model.onnx` before starting.
Missing files fail rather than creating an empty directory. Model artifacts are
deliberately excluded from git and the Docker build context; obtain your trained
artifact or run the documented training/export pipeline. Only load trusted models.

The Linux x86_64 image uses CPU Torch, a non-root user, a read-only filesystem,
bounded temporary storage and resources, and a readiness healthcheck. Compose
binds to loopback only. The app has no authentication or multi-tenant guarantees;
before Internet exposure, add TLS, authentication, rate limits, request/time
limits and monitoring at a reverse proxy. Do not expose it directly as-is.

CI runs locked lightweight Python 3.11/3.12 checks and Node tests, explicitly
requires API tests to execute, builds the CPU Docker image, and smoke-tests its
missing-model behavior (`/health=200`, `/ready=503`). CI cannot validate private
weights; run the opt-in model smoke separately. A workflow being present does
not mean a remote CI run or Docker build has passed—see the verification report.

## Roadmap

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Data pipeline + EDA (27 classes after exclusion) | done |
| 2 | Detector training; optional pretraining, balancing and evaluation tooling | trained detector available — mAP@0.5 = 0.6015; optional experiment reports not included |
| 3 | ONNX export, FP16/INT8 quantization, CPU latency benchmark | done — see [`reports/phase3_benchmark.md`](reports/phase3_benchmark.md) |
| 4 | FastAPI inference service | implemented and tested — `src/agridrone/serving.py` |
| 5 | Demo frontend | done — `frontend/`, served at `/` |
| 6 | Docs, tests, CI and CPU Docker deployment | implemented; local checks pass, Docker runtime verification pending host access |

## License / attribution

PlantDoc dataset © its authors, released under **CC-BY-4.0**. See the
[dataset repo](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
for citation details.
