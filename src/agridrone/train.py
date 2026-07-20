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
from .weighting import (
    class_instance_counts,
    class_weights,
    expected_class_exposure,
)

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


def make_weighted_dataset_class(
    agg: str = "mean", *, power: float = 0.5, max_ratio: float | None = 5.0
):  # pragma: no cover - needs ultralytics
    """Build a ``YOLODataset`` subclass that oversamples rare-class images.

    The subclass computes per-image sampling probabilities once at init (using the
    pure-numpy math in :mod:`agridrone.weighting`) and, in training mode, redirects
    each ``__getitem__`` to an index drawn from that distribution. Validation and
    test loading are untouched, so evaluation stays honest.

    Args:
        agg: Aggregation used to combine an image's per-class weights into one
            per-image weight (``mean|max|median|sum``).
        power: Inverse-frequency exponent for class weights (``0.5`` = sqrt
            dampening, ``1.0`` = pure inverse frequency). See
            :func:`agridrone.weighting.class_weights`.
        max_ratio: Cap on the rarest/commonest class-weight ratio (``None`` =
            uncapped). Bounds the realized oversampling amplification.

    Returns:
        A ``YOLODataset`` subclass ready to be patched into ``ultralytics.data.build``.
    """
    _require_ultralytics()
    from ultralytics.data.dataset import YOLODataset  # noqa: PLC0415

    from .weighting import image_weights, sampling_probabilities

    class YOLOWeightedDataset(YOLODataset):
        """YOLODataset that samples images by inverse class frequency in training."""

        # Set by the train-mode instance so finetune_detector can report the
        # oversampling effect (raw vs effective per-class exposure) after training.
        last_train_stats: dict[str, Any] | None = None

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
            n_classes = max(n_classes, 1)
            counts = class_instance_counts(per_image, n_classes=n_classes)
            cls_w = class_weights(counts, power=power, max_ratio=max_ratio)
            img_w = image_weights(per_image, cls_w, agg=agg)
            self.probabilities = sampling_probabilities(img_w)
            # Record the exact distribution the sampler will use so the effect is
            # reportable as a number, not just trusted. Only the train split
            # oversamples, so only it is worth capturing.
            if self.train_mode:
                type(self).last_train_stats = {
                    "names": dict(getattr(self, "data", {}).get("names", {})),
                    "n_classes": n_classes,
                    "counts": counts,
                    "probabilities": self.probabilities,
                    "per_image": per_image,
                    "agg": agg,
                    "power": power,
                    "max_ratio": max_ratio,
                }

        def __getitem__(self, index: int) -> Any:
            if not self.train_mode:
                return super().__getitem__(index)
            drawn = int(np.random.choice(len(self.labels), p=self.probabilities))
            return super().__getitem__(drawn)

    return YOLOWeightedDataset


def enable_weighted_sampling(
    agg: str = "mean", *, power: float = 0.5, max_ratio: float | None = 5.0
):  # pragma: no cover - needs ultralytics
    """Monkey-patch ultralytics so training uses the weighted dataset.

    Ultralytics builds datasets via ``ultralytics.data.build.YOLODataset``; swapping
    that name in is the documented, source-free way to inject a custom dataset for
    ``Detect`` training. Call this before ``model.train(...)``.

    Args:
        agg: Per-image weight aggregation (``mean|max|median|sum``).
        power: Inverse-frequency exponent (``0.5`` = sqrt dampening, ``1.0`` = pure).
        max_ratio: Cap on the rarest/commonest weight ratio (``None`` = uncapped).

    Returns the patched-in dataset class so the caller can read
    ``last_train_stats`` after training to report the oversampling effect.
    """
    _require_ultralytics()
    import ultralytics.data.build as build  # noqa: PLC0415

    weighted_cls = make_weighted_dataset_class(agg=agg, power=power, max_ratio=max_ratio)
    build.YOLODataset = weighted_cls
    print(
        f"[train] Weighted oversampling enabled "
        f"(agg={agg}, power={power}, max_ratio={max_ratio})."
    )
    return weighted_cls


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
    weighted_cls = None
    if wcfg.get("enabled", False) and not baseline:
        weighted_cls = enable_weighted_sampling(
            agg=wcfg.get("agg_func", "mean"),
            power=float(wcfg.get("power", 0.5)),
            max_ratio=wcfg.get("max_ratio", 5.0),
        )

    weights = cfg["models"]["baseline"] if baseline else cfg["models"]["detector"]
    start_from = str(backbone) if (backbone and not baseline) else weights
    model = YOLO(start_from)

    results = model.train(
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

    # Record the actual oversampling effect (raw vs effective per-class exposure)
    # so it is a number to point at, not code taken on trust.
    if weighted_cls is not None and weighted_cls.last_train_stats is not None:
        write_sampling_report(cfg, weighted_cls.last_train_stats)

    return results


_RESULTS_H1 = "# Phase 2 — Detection results"
_RESULTS_HEADING = "Detection accuracy — per-class AP + mAP"
_SAMPLING_HEADING = "Oversampling effect — raw vs effective per-class exposure"


def _upsert_section(existing: str, heading: str, body: str) -> str:
    """Insert or replace a ``## heading`` section within a results markdown doc.

    Keeps the shared ``# Phase 2`` H1 and any sibling sections intact, so the
    accuracy table and the oversampling table coexist in one file no matter which
    stage writes first or whether a stage is re-run. ``body`` is the section text
    starting at its ``## heading`` line.
    """
    body = body.rstrip("\n")
    marker = f"## {heading}"
    if not existing.strip():
        return f"{_RESULTS_H1}\n\n{body}\n"

    lines = existing.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == marker), None)
    if start is None:
        return existing.rstrip("\n") + "\n\n" + body + "\n"
    # Replace from the heading up to (but not including) the next H2, or EOF.
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    new_lines = lines[:start] + body.splitlines() + [""] + lines[end:]
    return "\n".join(new_lines).rstrip("\n") + "\n"


def format_sampling_report(stats: dict[str, Any]) -> str:
    """Render the raw-vs-effective per-class exposure table from sampling stats.

    ``stats`` is the ``last_train_stats`` dict captured by the weighted dataset:
    ``names``, ``n_classes``, ``counts`` (raw boxes per class), ``probabilities``
    (the exact per-image distribution the sampler used), and ``per_image`` (each
    image's class-ID list). The effective column is the *expected* number of boxes
    of each class seen in one epoch of oversampling — comparable to the raw counts
    because a uniform sampler would reproduce them exactly. See
    :func:`agridrone.weighting.expected_class_exposure`.
    """
    names: dict[int, str] = stats["names"]
    n_classes: int = stats["n_classes"]
    counts = np.asarray(stats["counts"], dtype=float)
    effective = expected_class_exposure(
        stats["per_image"], stats["probabilities"], n_classes=n_classes
    )
    total_boxes = float(counts.sum())
    power = stats.get("power")
    max_ratio = stats.get("max_ratio")

    # Describe the active weighting knobs so the report is self-explaining: a
    # reader can see whether dampening/capping was on without reading the config.
    if power is None:
        knob_line = ""
    else:
        cap_txt = f"capped at {max_ratio:g}x" if max_ratio is not None else "uncapped"
        power_txt = {1.0: "pure inverse frequency", 0.5: "sqrt-dampened"}.get(
            float(power), f"inverse frequency ** {power:g}"
        )
        knob_line = (
            f"Weighting: **{power_txt}** (`power={power:g}`), class-weight ratio "
            f"**{cap_txt}** (`max_ratio={max_ratio if max_ratio is not None else 'none'}`). "
            "Dampening + the cap deliberately hold back the rarest classes so training "
            "does not just memorize a handful of repeated tail images.\n"
        )

    lines = [
        f"## {_SAMPLING_HEADING}\n",
        "How many bounding-box instances of each class the model *actually* sees "
        "per epoch once weighted oversampling is active, versus the raw dataset "
        "counts. `Effective` is the expected boxes/epoch under the sampler's own "
        "probability vector (one epoch = one draw per image, with replacement); a "
        "uniform sampler would reproduce `Raw` exactly, so `x` is the amplification "
        "factor. Rare classes should show `x > 1`, common classes `x < 1`.\n",
    ]
    if knob_line:
        lines.append(knob_line)
    lines += [
        "| ID | Class | Raw boxes | Raw % | Effective/epoch | Effective % | x |",
        "|---:|-------|----------:|------:|----------------:|------------:|----:|",
    ]
    eff_total = float(effective.sum())
    # Order by rarity (rarest first) so the tail classes the oversampling targets
    # are read first, not buried at the bottom.
    order = sorted(range(n_classes), key=lambda c: counts[c])
    for c in order:
        name = names.get(c, str(c))
        raw = counts[c]
        eff = effective[c]
        raw_pct = (raw / total_boxes * 100.0) if total_boxes else 0.0
        eff_pct = (eff / eff_total * 100.0) if eff_total else 0.0
        factor = (eff / raw) if raw > 0 else float("nan")
        factor_cell = f"{factor:.2f}" if raw > 0 else "—"
        lines.append(
            f"| {c} | {name} | {int(raw)} | {raw_pct:.1f}% | "
            f"{eff:.1f} | {eff_pct:.1f}% | {factor_cell} |"
        )
    return "\n".join(lines) + "\n"


def write_sampling_report(cfg: dict[str, Any], stats: dict[str, Any]) -> Path:
    """Upsert the oversampling-effect table into the Phase 2 results markdown.

    Writes to ``output.sampling_report`` if set, else shares the results-table
    file (``output.results_table``). The section is inserted or replaced in place
    so it coexists with the accuracy table regardless of stage order or re-runs.
    """
    target = cfg["output"].get("sampling_report") or cfg["output"]["results_table"]
    out = resolve_path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    merged = _upsert_section(existing, _SAMPLING_HEADING, format_sampling_report(stats))
    out.write_text(merged, encoding="utf-8")
    print(f"[train] Oversampling report written to {out}")
    return out


def _format_results_table(names: dict[int, str], metrics: Any) -> str:
    """Render a per-class AP + aggregate mAP markdown section from val metrics."""
    lines = [
        f"## {_RESULTS_HEADING}\n",
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
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    merged = _upsert_section(
        existing, _RESULTS_HEADING, _format_results_table(names, metrics)
    )
    out.write_text(merged, encoding="utf-8")
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
