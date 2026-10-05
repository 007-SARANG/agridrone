# AgriDrone — Project Context

## What this is
A crop disease detection portfolio project. Resume-facing — code quality
and honest, real benchmark numbers matter more than moving fast.

## Hard constraints
- No Jetson/edge hardware — all "deployment" proof is software-side
  (ONNX Runtime quantization + CPU/GPU latency benchmarking).
- Never fabricate or estimate benchmark numbers — always run them for real.
- If PlantVillage turns out to be classification-only rather than
  detection-labeled, say so plainly rather than reframing the task.

## Conventions
- All production code lives in src/agridrone/, not in notebooks/.
- Config via YAML files in configs/, no hardcoded hyperparameters.
- Every new feature gets a test in tests/ before being considered done.

## Current phase
All six implementation phases delivered. Production serving is in
src/agridrone/serving.py; api/main.py is a compatibility shim. Static browser
assets are in frontend/. Use Python 3.11/3.12 (not system Python 3.14).
Run make setup with uv, then make check and make api. Serving is CPU ONNX FP32.
Docker and CI are configured; see reports/completion.md for actual verification
and remaining environment constraints. Never equate implemented workflows with
completed optional training experiments or unrun deployment checks.
