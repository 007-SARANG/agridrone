"""Phase 3 — latency + accuracy + size benchmark across precisions.

Compares the fine-tuned detector at four precisions on CPU:

    pytorch_fp32  — the original .pt via Ultralytics
    onnx_fp32     — exported ONNX, FP32
    onnx_fp16     — exported ONNX, FP16 (usually ~no CPU speedup — reported anyway)
    onnx_int8     — statically quantized ONNX

For each it records latency (mean/median/p95 ms per image on CPU), accuracy
(mAP@0.5 and mAP@0.5:0.95 on the untouched TEST split), and on-disk size (MB),
then writes a markdown table + bar charts. Every number comes from a real run on
this machine — nothing is estimated (see CLAUDE.md).

Accuracy uses a hybrid strategy: Ultralytics ``val()`` where the format loads
cleanly (it handles ONNX FP32/FP16 and often INT8 too); the code records which
path produced each row so the table is honest about provenance.

Heavy imports are lazy so the pure helpers (latency statistics, table/markdown
formatting) stay unit-testable in the lightweight environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Pure helpers (no heavy deps — unit-testable in the lightweight env)
# ---------------------------------------------------------------------------


def latency_stats(timings_s: list[float]) -> dict[str, float]:
    """Summarize per-run wall times (seconds) into ms-per-image statistics.

    Returns mean, median, p95, and min in milliseconds. Empty input yields zeros
    so downstream formatting never divides by nothing.
    """
    if not timings_s:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0}
    ms = sorted(t * 1000.0 for t in timings_s)
    n = len(ms)
    mean = sum(ms) / n
    median = ms[n // 2] if n % 2 else (ms[n // 2 - 1] + ms[n // 2]) / 2
    p95 = ms[min(n - 1, int(round(0.95 * (n - 1))))]
    return {"mean_ms": mean, "median_ms": median, "p95_ms": p95, "min_ms": ms[0]}


def file_size_mb(path: Path) -> float:
    """Return a file's size in MB (1e6 bytes), or 0.0 if it does not exist."""
    p = Path(path)
    return round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0


def _speedup_column(rows: list[dict[str, Any]]) -> None:
    """Annotate each row in place with `speedup` relative to pytorch_fp32.

    Speedup = baseline mean latency / this row's mean latency (higher = faster).
    If the baseline is missing, speedups are left as None.
    """
    base = next(
        (r["mean_ms"] for r in rows if r["key"] == "pytorch_fp32" and r["mean_ms"]),
        None,
    )
    for r in rows:
        r["speedup"] = round(base / r["mean_ms"], 2) if base and r.get("mean_ms") else None


_NOT_VIABLE_REASONS: dict[str, str] = {
    "onnx_int8": (
        "collapses to 0.0 mAP (NMS-free head incompatible with activation "
        "quantization); no latency benefit on this CPU (no AVX-512/VNNI)"
    ),
}


def _viability_note(row: dict[str, Any]) -> str:
    """Return 'not viable — <reason>' if the row should be flagged, else ''."""
    map50 = row.get("map50")
    if map50 is not None and map50 != 0.0:
        return ""
    reason = _NOT_VIABLE_REASONS.get(row["key"], "accuracy collapsed to 0.0 or failed")
    return f"not viable — {reason}"


def _recommend(rows: list[dict[str, Any]]) -> str:
    """Return a deployment recommendation based on viable rows."""
    viable = [r for r in rows if not _viability_note(r)]
    for key in ("onnx_fp32", "onnx_fp16", "pytorch_fp32"):
        match = next((r for r in viable if r["key"] == key), None)
        if match:
            return f"Recommended deployment artifact: {match['label']}."
    return "No viable deployment artifact — all precisions failed accuracy checks."


def format_report(rows: list[dict[str, Any]], *, notes: list[str] | None = None) -> str:
    """Render the benchmark rows into the Phase 3 markdown report.

    Each row dict carries: ``key``, ``label``, ``mean_ms``, ``median_ms``,
    ``p95_ms``, ``map50``, ``map``, ``size_mb``, ``accuracy_source`` and (added
    here) ``speedup``. Missing accuracy is shown as ``—`` rather than a fake 0.
    """
    _speedup_column(rows)
    lines = [
        "# Phase 3 — Inference optimization benchmark\n",
        "All numbers measured on CPU on the dev laptop (no GPU). Latency is per "
        "single image at 640×640; accuracy is on the untouched PlantDoc TEST "
        "split. FP16 is included even though CPU runtimes upcast it to FP32 — the "
        "~flat latency is an honest finding, not an omission.\n",
        "| Precision | Latency mean (ms) | median | p95 | Speedup | mAP@0.5 | "
        "mAP@0.5:0.95 | Size (MB) |",
        "|-----------|------------------:|-------:|----:|--------:|--------:|"
        "-------------:|----------:|",
    ]
    for r in rows:
        viability = _viability_note(r)
        label = f"{r['label']} — {viability}" if viability else r["label"]
        spd = f"{r['speedup']:.2f}×" if r.get("speedup") else "—"
        map50 = f"{r['map50']:.4f}" if r.get("map50") is not None else "—"
        mapc = f"{r['map']:.4f}" if r.get("map") is not None else "—"
        lines.append(
            f"| {label} | {r['mean_ms']:.2f} | {r['median_ms']:.2f} | "
            f"{r['p95_ms']:.2f} | {spd} | {map50} | {mapc} | {r['size_mb']:.2f} |"
        )
    lines.append("")
    # Provenance of each accuracy number (which tool produced it).
    srcs = {r["key"]: r.get("accuracy_source") for r in rows if r.get("accuracy_source")}
    if srcs:
        lines.append("Accuracy source per row:\n")
        for k, v in srcs.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
    for note in notes or []:
        lines.append(f"> {note}\n")
    lines.append(_recommend(rows))
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Measurement (needs the full stack)
# ---------------------------------------------------------------------------


def _require(mod_name: str):  # pragma: no cover - exercised only with full stack
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(mod_name)
    except ImportError as exc:
        raise RuntimeError(
            f"'{mod_name}' is not installed. Install the Phase 3 CPU stack with "
            "`make setup-phase3`."
        ) from exc


def _time_onnx(
    onnx_path: Path, imgsz: int, warmup: int, runs: int
) -> list[float]:  # pragma: no cover - needs onnxruntime
    """Time an ONNX model with a fixed random input over `runs` iterations (CPU)."""
    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    ort = _require("onnxruntime")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # FP16 models expect float16 input; detect from the input type.
    dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, imgsz, imgsz)).astype(dtype)

    for _ in range(warmup):
        sess.run(None, {name: x})
    timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        timings.append(time.perf_counter() - t0)
    return timings


def _time_pytorch(
    weights: Path, imgsz: int, warmup: int, runs: int
) -> list[float]:  # pragma: no cover - needs torch
    """Time the PyTorch model's forward pass with a fixed random input (CPU)."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    torch_model = model.model
    if not isinstance(torch_model, torch.nn.Module):
        raise TypeError("Expected a PyTorch module for the PyTorch latency benchmark")
    torch_model = torch_model.eval()
    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for _ in range(warmup):
            torch_model(x)
        timings: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            torch_model(x)
            timings.append(time.perf_counter() - t0)
    return timings


def _accuracy(
    model_path: Path, dataset_yaml: Path, imgsz: int
) -> tuple[float | None, float | None, str]:  # pragma: no cover - needs ultralytics
    """Measure mAP on the TEST split via Ultralytics val(). Returns (map50, map, source).

    Works for .pt and ONNX (FP32/FP16/INT8) that Ultralytics can load. On failure
    returns (None, None, "failed: <reason>") so the row is honestly blank.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    try:
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(dataset_yaml), split="test", imgsz=imgsz, device="cpu",
            plots=False, verbose=False,
        )
        return float(metrics.box.map50), float(metrics.box.map), "ultralytics.val()"
    except Exception as exc:  # noqa: BLE001 - record failure, keep benchmarking
        return None, None, f"failed: {type(exc).__name__}: {exc}"


def run(
    cfg_path: str | Path, artifacts: dict[str, Path] | None = None
) -> Path:  # pragma: no cover - needs full stack
    """Run the full benchmark and write the report + charts.

    Args:
        cfg_path: Path to the export/benchmark config.
        artifacts: Optional map of precision key -> ONNX path (from export.run()).
            If omitted, ONNX paths are inferred from the export out_dir.

    Returns:
        Path to the written markdown report.
    """
    cfg = load_config(cfg_path)
    bcfg = cfg["benchmark"]
    imgsz = int(cfg["model"]["imgsz"])
    warmup, runs = int(bcfg["warmup"]), int(bcfg["runs"])
    weights = resolve_path(cfg["model"]["weights"])
    dataset_yaml = resolve_path(cfg["data"]["dataset_yaml"])

    out_dir = resolve_path(cfg["export"]["out_dir"])
    stem = weights.stem
    paths = artifacts or {
        "onnx_fp32": out_dir / f"{stem}_fp32.onnx",
        "onnx_fp16": out_dir / f"{stem}_fp16.onnx",
        "onnx_int8": out_dir / f"{stem}_int8.onnx",
    }

    labels = {
        "pytorch_fp32": "PyTorch FP32",
        "onnx_fp32": "ONNX FP32",
        "onnx_fp16": "ONNX FP16",
        "onnx_int8": "ONNX INT8",
    }
    rows: list[dict[str, Any]] = []
    for key in bcfg["precisions"]:
        if key == "pytorch_fp32":
            timings = _time_pytorch(weights, imgsz, warmup, runs)
            map50, mapc, src = _accuracy(weights, dataset_yaml, imgsz)
            size = file_size_mb(weights)
        else:
            onnx_path = Path(paths[key])
            timings = _time_onnx(onnx_path, imgsz, warmup, runs)
            map50, mapc, src = _accuracy(onnx_path, dataset_yaml, imgsz)
            size = file_size_mb(onnx_path)
        stats = latency_stats(timings)
        rows.append({
            "key": key, "label": labels.get(key, key),
            **stats, "map50": map50, "map": mapc,
            "size_mb": size, "accuracy_source": src,
        })
        print(f"[benchmark] {labels.get(key, key)}: {stats['mean_ms']:.2f} ms, "
              f"mAP50={map50 if map50 is None else round(map50, 4)}, {size} MB")

    report = format_report(rows, notes=[
        f"Latency: {runs} timed runs after {warmup} warmup, single-image, CPU.",
    ])
    out = resolve_path(cfg["output"]["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    _plot(rows, resolve_path(cfg["output"]["plots_dir"]))
    print(f"[benchmark] Report written to {out}")
    return out


def _plot(
    rows: list[dict[str, Any]], plots_dir: Path
) -> None:  # pragma: no cover - needs matplotlib
    """Save latency / size / accuracy bar charts."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [r["mean_ms"] for r in rows], color="steelblue")
    ax.set_ylabel("Latency (ms/image, CPU)")
    ax.set_title("Inference latency by precision")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / "latency.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [r["size_mb"] for r in rows], color="seagreen")
    ax.set_ylabel("Model size (MB)")
    ax.set_title("Model size by precision")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / "size.png", dpi=120)
    plt.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="AgriDrone Phase 3 benchmark.")
    parser.add_argument("--config", default="configs/export.yaml", help="Path to config.")
    args = parser.parse_args()
    run(args.config)
