# Phase 3 — Inference optimization benchmark

All numbers measured on CPU on the dev laptop (no GPU). Latency is per single image at 640×640; accuracy is on the untouched PlantDoc TEST split. FP16 is included even though CPU runtimes upcast it to FP32 — the ~flat latency is an honest finding, not an omission.

| Precision | Latency mean (ms) | median | p95 | Speedup | mAP@0.5 | mAP@0.5:0.95 | Size (MB) |
|-----------|------------------:|-------:|----:|--------:|--------:|-------------:|----------:|
| PyTorch FP32 | 84.35 | 82.64 | 93.93 | 1.00× | 0.6015 | 0.4786 | 5.40 |
| ONNX FP32 | 32.92 | 31.79 | 36.79 | 2.56× | 0.5839 | 0.4608 | 9.83 |
| ONNX FP16 | 55.63 | 53.99 | 62.90 | 1.52× | 0.5847 | 0.4611 | 4.98 |
| ONNX INT8 | 74.48 | 74.18 | 80.79 | 1.13× | 0.0000 | 0.0000 | 3.11 |

Accuracy source per row:

- `pytorch_fp32`: ultralytics.val()
- `onnx_fp32`: ultralytics.val()
- `onnx_fp16`: ultralytics.val()
- `onnx_int8`: ultralytics.val()

> Latency: 100 timed runs after 10 warmup, single-image, CPU.

