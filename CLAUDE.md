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
Just Starting
