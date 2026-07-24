"""Phase 3 — ONNX export and INT8 quantization for the fine-tuned detector.

Pipeline:

1. **Export** the PyTorch YOLO26 detector to ONNX at FP32 and FP16 via Ultralytics.
2. **Quantize** the FP32 ONNX to INT8 with onnxruntime *static* quantization,
   calibrating on a fixed random subset of TRAIN images (never test/val, so the
   accuracy number stays honest).

Everything runs on CPU — the deployment story is "provably fast on commodity
CPU", so exports target the CPU execution provider and the benchmark measures
real latency on this machine.

Heavy imports (torch/ultralytics/onnxruntime) are lazy, mirroring
:mod:`agridrone.train`, so the pure helpers (image listing, preprocessing) stay
importable and unit-testable in the lightweight environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import load_config, resolve_path

if TYPE_CHECKING:
    from collections.abc import Sequence

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Pure helpers (no heavy deps — unit-testable in the lightweight env)
# ---------------------------------------------------------------------------


def list_split_images(dataset_yaml: Path, split: str) -> list[Path]:
    """Return sorted image paths for a split of the processed YOLO dataset.

    Reads the ``path`` field of ``dataset_yaml`` and globs ``images/<split>``.

    Raises:
        FileNotFoundError: If the images directory does not exist.
    """
    cfg = load_config(dataset_yaml)
    root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else resolve_path(cfg["path"])
    images_dir = root / "images" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No images dir for split '{split}': {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)


def select_calibration_images(
    images: Sequence[Path], n_images: int, seed: int
) -> list[Path]:
    """Pick a deterministic random subset of images for INT8 calibration.

    Sampling without replacement; if ``n_images`` exceeds the pool, returns the
    whole pool (sorted). Deterministic given ``seed`` so quantization is
    reproducible.
    """
    pool = list(images)
    if n_images >= len(pool):
        return sorted(pool)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n_images, replace=False)
    return sorted(pool[int(i)] for i in idx)


def preprocess_image(path: Path, imgsz: int) -> np.ndarray:
    """Load and preprocess one image into a YOLO input tensor.

    Letterbox-free simple resize to ``(imgsz, imgsz)``, RGB, CHW, float32 in
    ``[0, 1]`` with a leading batch dim — matching Ultralytics' ONNX input
    contract for a fixed square input. Returns shape ``(1, 3, imgsz, imgsz)``.
    """
    from PIL import Image  # noqa: PLC0415 - keep Pillow out of import path

    with Image.open(path) as im:
        im = im.convert("RGB").resize((imgsz, imgsz))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr[np.newaxis, ...]  # add batch dim


# Ops that only appear in the YOLO26 NMS-free detection head (box decode + top-k
# selection). Walking upstream from these finds the head compute to keep in FP32.
_HEAD_MARKER_OPS = frozenset({"TopK", "GatherElements", "Softmax", "Mod", "ReduceMax"})
# Quantizable op types: excluding these from INT8 keeps them in FP32.
_QUANTIZABLE_OPS = frozenset({"Conv", "MatMul", "Gemm"})


def find_head_nodes_to_exclude(
    nodes: Sequence[Any], marker_ops: frozenset[str] = _HEAD_MARKER_OPS, max_depth: int = 40
) -> list[str]:
    """Return names of quantizable nodes in the detection head, to keep in FP32.

    Naive INT8 static quantization of a YOLO detector collapses its outputs to
    zero because the sensitive box/score-decoding layers in the head lose too much
    precision. The robust, model-agnostic fix is to quantize the backbone but keep
    the head in FP32. This locates the head by walking the graph *upstream* from
    the NMS-free marker ops (``TopK``/``Softmax``/... — present only in the head)
    and collecting the ``Conv``/``MatMul``/``Gemm`` nodes that feed them, bounded
    by ``max_depth`` hops so the whole backbone is not swept in.

    Args:
        nodes: ONNX graph nodes (each with ``.name``, ``.op_type``, ``.input``,
            ``.output``). Kept generic so it is unit-testable without onnx.
        marker_ops: Op types that mark the head boundary.
        max_depth: Max upstream hops from a marker before stopping.

    Returns:
        Sorted, de-duplicated node names to pass as ``nodes_to_exclude``.
    """
    producer: dict[str, Any] = {}
    for n in nodes:
        for out in n.output:
            producer[out] = n

    excluded: set[str] = set()
    seen: set[int] = set()
    # Seed the walk with each marker node's inputs.
    frontier: list[tuple[Any, int]] = [
        (n, 0) for n in nodes if n.op_type in marker_ops
    ]
    while frontier:
        node, depth = frontier.pop()
        if id(node) in seen or depth > max_depth:
            continue
        seen.add(id(node))
        if node.op_type in _QUANTIZABLE_OPS and node.name:
            excluded.add(node.name)
        for inp in node.input:
            parent = producer.get(inp)
            if parent is not None:
                frontier.append((parent, depth + 1))
    return sorted(excluded)


# ---------------------------------------------------------------------------
# Export (needs torch + ultralytics)
# ---------------------------------------------------------------------------


def _require(mod_name: str):  # pragma: no cover - exercised only with full stack
    """Import a heavy module lazily with a friendly error if it is missing."""
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(mod_name)
    except ImportError as exc:
        raise RuntimeError(
            f"'{mod_name}' is not installed. Install the Phase 3 CPU stack with "
            "`make setup-phase3` (torch-cpu, ultralytics, onnx, onnxruntime)."
        ) from exc


def export_onnx(
    cfg: dict[str, Any], *, half: bool = False
) -> Path:  # pragma: no cover - needs ultralytics
    """Export the PyTorch model to ONNX (FP32 or FP16).

    Args:
        cfg: Parsed export config.
        half: If True, export FP16 weights; else FP32.

    Returns:
        Path to the exported ``.onnx`` file (renamed with an fp32/fp16 suffix).
    """
    _require("torch")
    from ultralytics import YOLO  # noqa: PLC0415

    ecfg = cfg["export"]
    weights = resolve_path(cfg["model"]["weights"])
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=int(cfg["model"]["imgsz"]),
        opset=int(ecfg["opset"]),
        dynamic=bool(ecfg["dynamic"]),
        simplify=bool(ecfg["simplify"]),
        half=half,
        device="cpu",
    )
    out_dir = resolve_path(ecfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if half else "fp32"
    dst = out_dir / f"{weights.stem}_{tag}.onnx"
    Path(exported).replace(dst)
    print(f"[export] Wrote {tag.upper()} ONNX -> {dst}")
    return dst


def quantize_int8(
    cfg: dict[str, Any], fp32_onnx: Path
) -> Path:  # pragma: no cover - needs onnxruntime
    """Statically quantize an FP32 ONNX model to INT8.

    Calibrates on a deterministic TRAIN subset (never test/val). Returns the path
    to the quantized model.
    """
    _require("onnxruntime")
    from onnxruntime.quantization import (  # noqa: PLC0415
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: PLC0415

    qcfg = cfg["quantize"]
    cal = qcfg["calibration"]
    imgsz = int(cfg["model"]["imgsz"])
    images = select_calibration_images(
        list_split_images(resolve_path(cfg["data"]["dataset_yaml"]), cal["split"]),
        int(cal["n_images"]),
        int(cal["seed"]),
    )

    onnx_mod = _require("onnx")

    # Pre-process (shape inference + graph cleanup). Quantizing the pre-processed
    # graph is what onnxruntime recommends and silences the calibration warning.
    prepped = fp32_onnx.with_name(fp32_onnx.stem + "_prep.onnx")
    quant_pre_process(str(fp32_onnx), str(prepped), skip_symbolic_shape=True)

    model = onnx_mod.load(str(prepped))
    input_name = model.graph.input[0].name

    # Keep the sensitive detection-head layers in FP32 — quantizing them collapses
    # YOLO outputs to zero mAP. Model-agnostic: detected from the graph, not hardcoded.
    exclude: list[str] = []
    if qcfg.get("exclude_head", True):
        exclude = find_head_nodes_to_exclude(list(model.graph.node))
        print(
            f"[export] Keeping {len(exclude)} detection-head node(s) in FP32 "
            "(INT8 backbone only)."
        )

    class _Reader(CalibrationDataReader):
        """Feeds preprocessed calibration images one batch at a time."""

        def __init__(self) -> None:
            self._it = iter(images)

        def get_next(self) -> dict[str, np.ndarray] | None:
            path = next(self._it, None)
            if path is None:
                return None
            return {input_name: preprocess_image(path, imgsz)}

    out = fp32_onnx.with_name(fp32_onnx.stem.replace("_fp32", "") + "_int8.onnx")
    quantize_static(
        model_input=str(prepped),
        model_output=str(out),
        calibration_data_reader=_Reader(),
        quant_format=QuantFormat.QDQ,  # QDQ is the portable, per-channel-friendly format
        per_channel=bool(qcfg["per_channel"]),
        reduce_range=bool(qcfg["reduce_range"]),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
    )
    prepped.unlink(missing_ok=True)  # tidy the intermediate
    print(f"[export] Wrote INT8 ONNX -> {out} (calibrated on {len(images)} train images)")
    return out


def run(cfg_path: str | Path) -> dict[str, Path]:  # pragma: no cover - needs full stack
    """Export FP32 + FP16 ONNX and quantize to INT8. Returns the artifact paths."""
    cfg = load_config(cfg_path)
    fp32 = export_onnx(cfg, half=False)
    fp16 = export_onnx(cfg, half=True)
    int8 = quantize_int8(cfg, fp32)
    return {"onnx_fp32": fp32, "onnx_fp16": fp16, "onnx_int8": int8}


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="AgriDrone Phase 3 export + quantize.")
    parser.add_argument("--config", default="configs/export.yaml", help="Path to export config.")
    args = parser.parse_args()
    run(args.config)
