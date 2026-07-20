"""Phase 2 training orchestration: PlantVillage pretrain -> PlantDoc fine-tune.

Pipeline:

1. **Pretrain** a YOLO26 classifier backbone on PlantVillage (large, related
   leaf/disease imagery). This is feature pretraining only — PlantVillage never
   supplies detection labels.
2. **Fine-tune** the detector on PlantDoc's 27 kept classes, optionally with a
   weighted-oversampling dataloader (see :mod:`agridrone.weighting`) so rare
   classes are not drowned out by common ones.
3. **Evaluate** on the untouched TEST split and write a per-class AP + aggregate
   mAP table.

Training needs a GPU and the full ML stack (torch + ultralytics). The lightweight
dev environment has neither, so every function that touches ultralytics imports it
lazily through :func:`_require_ultralytics`, and this module imports cleanly (and
stays unit-testable) without it. Actual training is expected to run on Colab via
``notebooks/phase2_colab.ipynb``.

All hyperparameters and paths come from ``configs/train.yaml`` — nothing is
hardcoded here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .weighting import class_instance_counts, class_weights

_ULTRALYTICS_HINT = (
    "The training stack (torch + ultralytics) is not installed in this "
    "environment. Install it with `make setup` (needs a GPU box) or run "
    "training on Colab via notebooks/phase2_colab.ipynb."
)


def _require_ultralytics():  # pragma: no cover - exercised only with the full stack
    """Import ultralytics lazily, with a friendly message if it is absent.

    Returns the ``ultralytics`` module. Raising here (rather than at import time)
    keeps the module importable and testable in the lightweight environment.
    """
    try:
        import ultralytics  # noqa: PLC0415  (intentional lazy import)
    except ImportError as exc:
        raise RuntimeError(_ULTRALYTICS_HINT) from exc
    return ultralytics


# ---------------------------------------------------------------------------
# Weighted-oversampling dataloader
# ---------------------------------------------------------------------------


def make_weighted_dataset_class(agg: str = "mean"):  # pragma: no cover - needs ultralytics
    """Build a ``YOLODataset`` subclass that oversamples rare-class images.

    The subclass computes per-image sampling probabilities once at init (using the
    pure-numpy math in :mod:`agridrone.weighting`) and, in training mode, redirects
    each ``__getitem__`` to an index drawn from that distribution. Validation and
    test loading are untouched, so evaluation stays honest.

    Args:
        agg: Aggregation used to combine an image's per-class weights into one
            per-image weight (``mean|max|median|sum``).

    Returns:
        A ``YOLODataset`` subclass ready to be patched into ``ultralytics.data.build``.
    """
    _require_ultralytics()
    from ultralytics.data.dataset import YOLODataset  # noqa: PLC0415

    from .weighting import image_weights, sampling_probabilities

    class YOLOWeightedDataset(YOLODataset):
        """YOLODataset that samples images by inverse class frequency in training."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.train_mode = "train" in getattr(self, "prefix", "")
            self._build_sampling_distribution()

        def _build_sampling_distribution(self) -> None:
            # self.labels is a list of dicts; each has a "cls" array of class ids.
            per_image = [
                np.asarray(lbl["cls"], dtype=np.int64).reshape(-1).tolist()
                for lbl in self.labels
            ]
            n_classes = len(getattr(self, "data", {}).get("names", {})) or (
                int(max((max(p) for p in per_image if p), default=-1)) + 1
            )
            counts = class_instance_counts(per_image, n_classes=max(n_classes, 1))
            cls_w = class_weights(counts)
            img_w = image_weights(per_image, cls_w, agg=agg)
            self.probabilities = sampling_probabilities(img_w)

        def __getitem__(self, index: int) -> Any:
            if not self.train_mode:
                return super().__getitem__(index)
            drawn = int(np.random.choice(len(self.labels), p=self.probabilities))
            return super().__getitem__(drawn)

    return YOLOWeightedDataset


def enable_weighted_sampling(agg: str = "mean") -> None:  # pragma: no cover - needs ultralytics
    """Monkey-patch ultralytics so training uses the weighted dataset.

    Ultralytics builds datasets via ``ultralytics.data.build.YOLODataset``; swapping
    that name in is the documented, source-free way to inject a custom dataset for
    ``Detect`` training. Call this before ``model.train(...)``.
    """
    _require_ultralytics()
    import ultralytics.data.build as build  # noqa: PLC0415

    build.YOLODataset = make_weighted_dataset_class(agg=agg)
    print(f"[train] Weighted oversampling enabled (agg={agg}).")


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------


def resolve_dataset_yaml(cfg: dict[str, Any]) -> Path:
    """Return the dataset.yaml to train on, preferring the Drive-cached copy.

    On Colab the processed dataset is mounted from Google Drive
    (``data.drive_dataset_dir``) to avoid re-downloading + re-converting PlantDoc
    each session. If that is unset or missing, fall back to the locally converted
    ``data.dataset_yaml``.

    Raises:
        FileNotFoundError: If neither location has a dataset.yaml.
    """
    data_cfg = cfg["data"]
    drive_dir = (data_cfg.get("drive_dataset_dir") or "").strip()
    if drive_dir:
        candidate = Path(drive_dir) / "dataset.yaml"
        if candidate.exists():
            return candidate
        print(
            f"[train] drive_dataset_dir set but {candidate} not found; "
            "falling back to local dataset_yaml."
        )
    local = resolve_path(data_cfg["dataset_yaml"])
    if not local.exists():
        raise FileNotFoundError(
            f"No dataset.yaml found. Looked in Drive ('{drive_dir or 'unset'}') "
            f"and local '{local}'. Run `make data` (and sync to Drive for Colab)."
        )
    return local


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def pretrain_backbone(cfg: dict[str, Any]) -> Path | None:  # pragma: no cover - needs ultralytics
    """Pretrain the backbone as a classifier on PlantVillage.

    Returns the path to the saved classifier checkpoint, or ``None`` if pretraining
    is disabled or no PlantVillage dataset was provided (fine-tuning then starts
    from the stock COCO-pretrained detector instead).
    """
    ul = cfg.get("pretrain", {})
    if not ul.get("enabled", False):
        print("[train] Pretraining disabled; skipping PlantVillage step.")
        return None
    dataset_dir = (ul.get("dataset_dir") or "").strip()
    if not dataset_dir:
        print(
            "[train] pretrain.dataset_dir is empty; skipping PlantVillage pretrain. "
            "Set it to an ImageFolder dir (local or Drive) to enable."
        )
        return None

    _require_ultralytics()
    from ultralytics import YOLO  # noqa: PLC0415

    tcfg = cfg["train"]
    model = YOLO(cfg["models"]["classifier"])
    model.train(
        data=dataset_dir,
        epochs=int(ul["epochs"]),
        imgsz=int(ul["imgsz"]),
        batch=int(tcfg["batch"]),
        device=tcfg["device"],
        seed=int(tcfg["seed"]),
        project=cfg["output"]["runs_dir"],
        name="plantvillage_pretrain",
    )
    out = resolve_path(ul["backbone_weights"])
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"[train] Pretrained backbone saved to {out}")
    return out


def finetune_detector(
    cfg: dict[str, Any], *, backbone: Path | None = None, baseline: bool = False
):  # pragma: no cover - needs ultralytics
    """Fine-tune the detector on PlantDoc.

    Args:
        cfg: Parsed train config.
        backbone: Optional pretrained classifier checkpoint to warm-start from.
        baseline: If True, train the YOLOv8 baseline instead of the YOLO26 detector
            and skip weighted sampling (baseline is deliberately vanilla).

    Returns:
        The Ultralytics training results object.
    """
    _require_ultralytics()
    from ultralytics import YOLO  # noqa: PLC0415

    tcfg = cfg["train"]
    wcfg = cfg.get("weighting", {})
    if wcfg.get("enabled", False) and not baseline:
        enable_weighted_sampling(agg=wcfg.get("agg_func", "mean"))

    weights = cfg["models"]["baseline"] if baseline else cfg["models"]["detector"]
    start_from = str(backbone) if (backbone and not baseline) else weights
    model = YOLO(start_from)

    return model.train(
        data=str(resolve_dataset_yaml(cfg)),
        epochs=int(tcfg["epochs"]),
        imgsz=int(tcfg["imgsz"]),
        batch=int(tcfg["batch"]),
        device=tcfg["device"],
        seed=int(tcfg["seed"]),
        patience=int(tcfg["patience"]),
        workers=int(tcfg["workers"]),
        project=cfg["output"]["runs_dir"],
        name="plantdoc_baseline" if baseline else "plantdoc_finetune",
    )


def _format_results_table(names: dict[int, str], metrics: Any) -> str:
    """Render a per-class AP + aggregate mAP markdown table from val metrics."""
    lines = [
        "# Phase 2 — Detection results\n",
        f"**mAP@0.5:** {float(metrics.box.map50):.4f}  ",
        f"**mAP@0.5:0.95:** {float(metrics.box.map):.4f}\n",
        "Per-class AP@0.5 (tail classes reported explicitly, not hidden in the mean):\n",
        "| ID | Class | AP@0.5 |",
        "|---:|-------|-------:|",
    ]
    # metrics.box.ap50 is per-class, ordered by metrics.box.ap_class_index.
    ap50 = np.asarray(metrics.box.ap50, dtype=float)
    ap_idx = list(getattr(metrics.box, "ap_class_index", range(len(ap50))))
    ap_by_class = dict(zip(ap_idx, ap50, strict=False))
    for cid in sorted(names):
        ap = ap_by_class.get(cid)
        cell = f"{ap:.4f}" if ap is not None else "—"
        lines.append(f"| {cid} | {names[cid]} | {cell} |")
    return "\n".join(lines) + "\n"


def evaluate(cfg: dict[str, Any], weights: str | Path):  # pragma: no cover - needs ultralytics
    """Evaluate a trained detector on the TEST split; write the results table."""
    _require_ultralytics()
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    metrics = model.val(data=str(resolve_dataset_yaml(cfg)), split="test")
    names = dict(metrics.names) if hasattr(metrics, "names") else dict(model.names)
    out = resolve_path(cfg["output"]["results_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_format_results_table(names, metrics), encoding="utf-8")
    print(f"[train] Results table written to {out}")
    return metrics


def run(cfg_path: str | Path, stage: str) -> None:  # pragma: no cover - needs ultralytics
    """Run the requested stage(s): ``pretrain|finetune|baseline|eval|all``."""
    cfg = load_config(cfg_path)
    backbone: Path | None = None
    if stage in ("pretrain", "all"):
        backbone = pretrain_backbone(cfg)
    if stage in ("finetune", "all"):
        finetune_detector(cfg, backbone=backbone)
    if stage == "baseline":
        finetune_detector(cfg, baseline=True)
    if stage in ("eval", "all"):
        best = Path(cfg["output"]["runs_dir"]) / "plantdoc_finetune/weights/best.pt"
        evaluate(cfg, best)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="AgriDrone Phase 2 training.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["pretrain", "finetune", "baseline", "eval", "all"],
        help="Which stage(s) to run.",
    )
    args = parser.parse_args()
    run(args.config, args.stage)
