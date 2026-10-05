# AgriDrone — implementation and verification handoff

## Outcome

The existing portfolio project now has an end-to-end local CPU workflow:
trained ONNX FP32 artifact → bounded HTTP upload → inference → image-space
boxes and labels → cached confidence filtering in the browser demo.
Data/training/export/benchmark workflows remain intact. No model was retrained,
no benchmark was fabricated, and recorded accuracy/latency values were not changed.

### Delivered

- `src/agridrone/serving.py`: YAML-driven FastAPI service, static demo, liveness,
  warmup-backed readiness, byte/pixel bounds before inference, EXIF correction,
  safe errors and sanitized boxes. One CPU worker with busy rejection keeps the
  event loop responsive and avoids overlapping use of the cached model.
- `configs/serve.yaml`: ONNX FP32, 640px input, confidence 0.25; limits match the
  browser's 20 MiB file / 4096² pixel preparation (21 MiB multipart body).
- `api/main.py`: compatibility shim for the earlier entrypoint.
- `frontend/`: accessible 25–100% confidence control, coherent boxes/list/count,
  cached filtering, normalized PNG size check, stale-response protection, and
  explicit confidence/diagnosis limitations.
- `Makefile`, `requirements-*.txt`: supported Python 3.11/3.12 setup via uv,
  transitive locks, lightweight/full/runtime dependency sets, pip repair without
  deleting environments, complete check/export/benchmark/deployment commands.
- `docker/`, `.dockerignore`: CPU-only non-root deployment, read-only model mount,
  loopback port, resource limits, strict context allowlist and health/smoke checks.
- `.github/workflows/ci.yml`: Python 3.11/3.12, Node regression tests, lint/types,
  required API tests, Docker build and model-free container smoke jobs.
- `tests/test_api.py`, `tests/test_serving_integration.py`, frontend and deployment
  tests; two ML typing issues resolved in training/benchmark code.
- `README.md`, `CLAUDE.md`: working Arch/fish setup and explicit deployment scope.
  Private chat attachments are gitignored and excluded from the Docker context.

## Verification actually run

| Check | Result |
|---|---|
| `make check` in existing full Python 3.12 environment | 120 Python tests passed, 5 skipped; 13 frontend tests passed; Ruff and mypy passed |
| `make check VENV=<fresh lightweight Python 3.12 environment>` | 123 Python tests passed, 2 skipped; 13 frontend tests passed; Ruff and mypy passed |
| Same fresh lightweight check on Python 3.11 | Same passing counts and checks |
| Opt-in real ONNX integration test | 1 passed with existing artifact; no model downloads |
| Real HTTP smoke with separate locked CPU runtime + Uvicorn | Demo, JS assets, readiness and crop image upload returned 200; one detection returned for a 960×720 sample |
| `git diff --check` | Passed |

The real HTTP smoke used a local PlantDoc test image only to validate transport
and output shape, not to tune thresholds or report new accuracy. Its server was
stopped after the check. Browser interaction tests use a mocked DOM; no new
full graphical-browser acceptance run was performed in this worker.

Full-env skips: four missing-dependency guard tests skip when ML dependencies
are installed; the fifth is the opt-in real-artifact test, run separately.
Lightweight skips: one ML-only test plus that opt-in test. API tests do not skip.
Dependency deprecation warnings remain (Starlette/AnyIO and Matplotlib/pyparsing);
they did not fail checks. Ruff/mypy also passed against the installed ML stack.

## Remaining verification and honest limits

- **Docker build/run is not verified locally.** `docker info` returned a daemon
  socket permission error; `docker compose version` reported the plugin missing.
  No privilege changes or daemon changes were attempted. The CI Docker job has
  been authored but no remote CI run was started or claimed successful.
- The model's predictions remain imperfect. Filtering weak scores does not
  improve the trained model or turn confidence into a diagnostic probability.
- Existing Phase 3 results support FP32 deployment; INT8 is not viable in the
  recorded experiment. Broader quantization claims require new experiments.
- No per-class/baseline Phase 2 report is present. Optional PlantVillage
  pretraining/baseline workflows are implemented, but their experimental
  completion cannot be inferred just from code. Retraining was not necessary
  for completing this deployment/demo scope and was not performed.
- This is a local portfolio demo, not a publicly hardened multi-user service,
  agricultural decision system, actual drone integration, or edge-hardware test.
- No commit, push, cloud deployment, or release publication was performed.

## Run / resume

For this user's Arch/fish environment, no activation or pip bootstrap is needed:

```fish
uv python install 3.12
make setup
and make check
and make api
```

Open http://localhost:8000/ and upload a photo. Move Minimum model confidence;
boxes and labels should change immediately with no extra prediction request.
The existing trained file must be at
`models/onnx/plantdoc_yolo26_best_fp32.onnx` (or use `AGRIDRONE_MODEL`).

For a repeatable real model check:

```fish
env PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 AGRIDRONE_REAL_MODEL_TEST=1 .venv/bin/python -m pytest tests/test_serving_integration.py
```

On a Docker-enabled host, with port 8000 free:

```fish
make docker-up
docker compose -f docker/compose.yaml ps
curl --fail http://localhost:8000/ready
```

Next verification step: run those container commands, upload a real image in the
browser, and confirm Compose reports healthy. If startup fails, inspect
`docker compose -f docker/compose.yaml logs api` and verify the read-only ONNX
bind source exists and is readable. Use `make docker-down` to stop the container.
