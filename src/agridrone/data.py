"""PlantDoc object-detection data pipeline.

Responsibilities:
    1. Download and extract the PlantDoc object-detection dataset.
    2. Parse its Pascal-VOC-style CSV annotations.
    3. Convert boxes to YOLO format (normalized cx, cy, w, h) and lay the data
       out as the ``images/`` + ``labels/`` structure Ultralytics expects.
    4. Carve a validation split out of TRAIN (TEST is left untouched).
    5. Emit a ``dataset.yaml`` that Ultralytics can train against.

PlantDoc is a genuine object-detection dataset (bounding boxes), unlike
PlantVillage which is classification-only. See CLAUDE.md for why that matters.

The CSV schema is Pascal-VOC style, one row per bounding box:
    filename, width, height, class, xmin, ymin, xmax, ymax
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agridrone.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BoxRecord:
    """A single bounding box in absolute pixel coordinates (Pascal VOC style)."""

    filename: str
    img_w: int
    img_h: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def _raw_norm(self) -> tuple[float, float, float, float]:
        """Return un-clamped normalized (cx, cy, w, h)."""
        cx = ((self.xmin + self.xmax) / 2.0) / self.img_w
        cy = ((self.ymin + self.ymax) / 2.0) / self.img_h
        w = (self.xmax - self.xmin) / self.img_w
        h = (self.ymax - self.ymin) / self.img_h
        return cx, cy, w, h

    def needs_clamp(self, *, eps: float = 1e-9) -> bool:
        """True if any normalized coordinate falls outside [0, 1] before clamping.

        Used to count (and report) how many PlantDoc boxes spill past image
        edges and therefore get clamped during conversion.
        """
        return any(v < -eps or v > 1.0 + eps for v in self._raw_norm())

    def to_yolo(self, class_id: int) -> str:
        """Return a YOLO-format label line: ``class cx cy w h`` (normalized).

        Coordinates are clamped to [0, 1] to guard against annotation noise
        (PlantDoc has some boxes that spill slightly past image edges). Use
        :meth:`needs_clamp` to detect when clamping actually changes a box.
        """
        cx, cy, w, h = self._raw_norm()
        cx, cy = _clamp01(cx), _clamp01(cy)
        w, h = _clamp01(w), _clamp01(h)
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class SplitStats:
    """Summary counts for one dataset split."""

    name: str
    n_images: int = 0
    n_boxes: int = 0
    n_images_without_boxes: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    # Data-quality counters, surfaced in the EDA report.
    n_boxes_dropped_degenerate: int = 0  # zero/negative-area boxes skipped
    n_boxes_dropped_bad_dims: int = 0  # rows with width<=0 or height<=0 in the CSV
    n_boxes_dropped_excluded_class: int = 0  # boxes of classes excluded via config
    n_boxes_clamped: int = 0  # boxes kept but with coords clamped to [0, 1]
    n_rows_missing_image: int = 0  # CSV rows whose image file was absent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _load_labels_csv(csv_path: Path, cols: dict[str, str]) -> pd.DataFrame:
    """Load a PlantDoc labels CSV, normalizing column names to a fixed schema."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    rename = {
        cols["filename"]: "filename",
        cols["width"]: "width",
        cols["height"]: "height",
        cols["class_name"]: "class_name",
        cols["xmin"]: "xmin",
        cols["ymin"]: "ymin",
        cols["xmax"]: "xmax",
        cols["ymax"]: "ymax",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df.rename(columns=rename)[list(rename.values())]


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------


def download_dataset(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Download and extract the PlantDoc object-detection dataset.

    Args:
        cfg: Parsed data config (see ``configs/data.yaml``).
        force: Re-download even if the raw directory already looks populated.

    Returns:
        Path to the extracted dataset root (the folder containing TRAIN/TEST
        and the two ``*_labels.csv`` files).
    """
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_dataset_root(raw_dir)
    if existing is not None and not force:
        print(f"[data] Dataset already present at {existing} (use force=True to redownload).")
        return existing

    url = cfg["dataset"]["download_url"]
    print(f"[data] Downloading PlantDoc detection dataset from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "agridrone/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted GitHub URL from config
        blob = resp.read()
    print(f"[data] Downloaded {len(blob) / 1e6:.1f} MB; extracting ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(raw_dir)

    root = _find_dataset_root(raw_dir)
    if root is None:
        raise RuntimeError(
            f"Extraction finished but no TRAIN/ + *_labels.csv found under {raw_dir}."
        )
    print(f"[data] Extracted to {root}")
    return root


def _find_dataset_root(search_dir: Path) -> Path | None:
    """Find the folder that actually contains the dataset (handles zip nesting).

    A GitHub archive extracts to ``<repo>-<branch>/`` so we search one level deep
    for a directory containing ``train_labels.csv``.
    """
    if not search_dir.exists():
        return None
    candidates = [search_dir, *[p for p in search_dir.iterdir() if p.is_dir()]]
    for c in candidates:
        if (c / "train_labels.csv").exists() and (c / "test_labels.csv").exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Conversion CSV -> YOLO
# ---------------------------------------------------------------------------


def build_class_list(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    excluded: Iterable[str] | None = None,
) -> list[str]:
    """Return the sorted, de-duplicated class name list across both splits.

    Sorting makes class IDs deterministic and reproducible across runs.
    Any names in ``excluded`` are removed (matched after stripping whitespace)
    so their boxes never receive a class ID.
    """
    excluded_set = {str(e).strip() for e in (excluded or [])}
    names = pd.concat([train_df["class_name"], test_df["class_name"]], ignore_index=True)
    uniq = names.astype(str).str.strip().unique().tolist()
    return sorted(n for n in uniq if n not in excluded_set)


def convert_split(
    df: pd.DataFrame,
    src_images_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    class_to_id: dict[str, int],
    *,
    filenames: set[str] | None = None,
) -> SplitStats:
    """Convert one split's rows into YOLO images/ + labels/ layout.

    Args:
        df: Annotation rows (normalized schema) for the source split.
        src_images_dir: Directory holding the source images.
        dst_images_dir: Destination images directory (created if missing).
        dst_labels_dir: Destination labels directory (created if missing).
        class_to_id: Mapping from class name to integer id.
        filenames: If given, only process rows whose filename is in this set
            (used to separate train vs. val out of the original TRAIN split).

    Returns:
        SplitStats with per-class and per-image counts.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(name=dst_images_dir.parent.name)
    grouped = df.groupby("filename")

    for filename, rows in grouped:
        fname = str(filename)
        if filenames is not None and fname not in filenames:
            continue
        src_img = src_images_dir / fname
        if not src_img.exists():
            # Some PlantDoc CSV rows reference images missing from the repo.
            stats.n_rows_missing_image += len(rows)
            continue

        label_lines: list[str] = []
        for _, r in rows.iterrows():
            cname = str(r["class_name"]).strip()
            # Classes excluded via config carry no class ID; drop their boxes.
            # (An image left with no remaining boxes becomes a background image,
            # matching how degenerate/bad boxes are already handled.)
            if cname not in class_to_id:
                stats.n_boxes_dropped_excluded_class += 1
                continue
            # A handful of PlantDoc rows record width/height as 0 (annotation
            # errors). Their normalized coords are undefined, so drop them.
            if int(r["width"]) <= 0 or int(r["height"]) <= 0:
                stats.n_boxes_dropped_bad_dims += 1
                continue
            # Skip degenerate boxes (zero/negative area) that exist in PlantDoc.
            if float(r["xmax"]) <= float(r["xmin"]) or float(r["ymax"]) <= float(r["ymin"]):
                stats.n_boxes_dropped_degenerate += 1
                continue
            box = BoxRecord(
                filename=fname,
                img_w=int(r["width"]),
                img_h=int(r["height"]),
                class_name=cname,
                xmin=float(r["xmin"]),
                ymin=float(r["ymin"]),
                xmax=float(r["xmax"]),
                ymax=float(r["ymax"]),
            )
            if box.needs_clamp():
                stats.n_boxes_clamped += 1
            label_lines.append(box.to_yolo(class_to_id[cname]))
            stats.n_boxes += 1
            stats.class_counts[cname] = stats.class_counts.get(cname, 0) + 1

        # Copy image and write its label file (empty label = background image).
        shutil.copy2(src_img, dst_images_dir / fname)
        label_path = dst_labels_dir / (Path(fname).stem + ".txt")
        body = "\n".join(label_lines) + ("\n" if label_lines else "")
        label_path.write_text(body, encoding="utf-8")

        stats.n_images += 1
        if not label_lines:
            stats.n_images_without_boxes += 1

    return stats


def _split_train_val(
    train_df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Split TRAIN filenames into train/val sets deterministically.

    Splitting is done at the *image* level (not the box level) so that all boxes
    for an image stay together in one split.
    """
    unique_files = pd.Series(train_df["filename"].astype(str).unique())
    shuffled = unique_files.sample(frac=1.0, random_state=seed).tolist()
    n_val = int(round(len(shuffled) * val_fraction))
    val_files = set(shuffled[:n_val])
    train_files = set(shuffled[n_val:])
    return train_files, val_files


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_dataset(
    config_path: str | Path = "configs/data.yaml", *, force_download: bool = False
) -> dict[str, Any]:
    """Run the full Phase-1 pipeline: download -> convert -> split -> dataset.yaml.

    Returns:
        A summary dict with class names, split stats, and the path to the
        generated ``dataset.yaml``.
    """
    cfg = load_config(config_path)
    root = download_dataset(cfg, force=force_download)

    csv_cols = cfg["csv_columns"]
    train_df = _load_labels_csv(root / "train_labels.csv", csv_cols)
    test_df = _load_labels_csv(root / "test_labels.csv", csv_cols)

    excluded = cfg.get("excluded_classes", [])
    class_names = build_class_list(train_df, test_df, excluded=excluded)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    if excluded:
        print(f"[data] Excluding {len(excluded)} class(es) from training: {excluded}")

    processed = resolve_path(cfg["paths"]["processed_dir"])
    if processed.exists():
        shutil.rmtree(processed)

    train_src = root / cfg["splits"]["train_dir"]
    test_src = root / cfg["splits"]["test_dir"]

    tr_files, val_files = _split_train_val(
        train_df, cfg["splits"]["val_fraction"], cfg["splits"]["seed"]
    )

    stats = {
        "train": convert_split(
            train_df, train_src, processed / "images/train", processed / "labels/train",
            class_to_id, filenames=tr_files,
        ),
        "val": convert_split(
            train_df, train_src, processed / "images/val", processed / "labels/val",
            class_to_id, filenames=val_files,
        ),
        "test": convert_split(
            test_df, test_src, processed / "images/test", processed / "labels/test",
            class_to_id,
        ),
    }

    dataset_yaml = write_dataset_yaml(processed, class_names)

    summary = {
        "dataset_root": str(root),
        "processed_dir": str(processed),
        "dataset_yaml": str(dataset_yaml),
        "n_classes": len(class_names),
        "class_names": class_names,
        "excluded_classes": list(excluded),
        "splits": {k: _stats_to_dict(v) for k, v in stats.items()},
    }

    # Persist the summary so downstream tools (EDA) can report data-quality
    # counters (dropped/clamped boxes) without recomputing them.
    import json as _json

    (processed / "prepare_summary.json").write_text(
        _json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_dataset_yaml(processed_dir: Path, class_names: list[str]) -> Path:
    """Write the Ultralytics ``dataset.yaml`` describing the YOLO layout."""
    lines = [
        "# Auto-generated by agridrone.data — do not edit by hand.",
        f"path: {processed_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(class_names))
    out = processed_dir / "dataset.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def sync_to_drive(
    dest_dir: str | Path, config_path: str | Path = "configs/data.yaml"
) -> Path:
    """Copy the processed YOLO dataset to ``dest_dir`` (e.g. a mounted Drive path).

    Training runs on Colab, where re-downloading + re-converting PlantDoc (~995 MB)
    every session is wasteful. Converting once locally and caching the processed
    dataset to Google Drive lets Colab mount it directly. This copies the
    ``images/``, ``labels/``, and metadata into ``dest_dir`` and rewrites the
    ``dataset.yaml`` ``path:`` line so it points at the destination (the original
    absolute path would be meaningless on Colab).

    Args:
        dest_dir: Target directory (created if absent). On Colab this is under the
            mounted Drive, e.g. ``/content/drive/MyDrive/agridrone/plantdoc``.
        config_path: Data config, used to locate the local processed dir.

    Returns:
        Path to the copied ``dataset.yaml`` in ``dest_dir``.

    Raises:
        FileNotFoundError: If the local processed dataset does not exist yet.
    """
    cfg = load_config(config_path)
    processed = resolve_path(cfg["paths"]["processed_dir"])
    if not (processed / "dataset.yaml").exists():
        raise FileNotFoundError(
            f"No processed dataset at {processed}. Run `make data` first."
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # Copy the dataset tree (images/, labels/, prepare_summary.json). dirs_exist_ok
    # lets a re-sync overwrite an earlier copy in place.
    shutil.copytree(processed, dest, dirs_exist_ok=True)

    # Rewrite the absolute `path:` line to the destination so Ultralytics resolves
    # images/ and labels/ correctly wherever the dir is mounted.
    dst_yaml = dest / "dataset.yaml"
    rewritten = [
        f"path: {dest.resolve()}" if line.startswith("path:") else line
        for line in dst_yaml.read_text(encoding="utf-8").splitlines()
    ]
    dst_yaml.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    print(f"[data] Synced processed dataset to {dest}")
    return dst_yaml


def _stats_to_dict(s: SplitStats) -> dict[str, Any]:
    return {
        "n_images": s.n_images,
        "n_boxes": s.n_boxes,
        "n_images_without_boxes": s.n_images_without_boxes,
        "n_boxes_dropped_degenerate": s.n_boxes_dropped_degenerate,
        "n_boxes_dropped_bad_dims": s.n_boxes_dropped_bad_dims,
        "n_boxes_dropped_excluded_class": s.n_boxes_dropped_excluded_class,
        "n_boxes_clamped": s.n_boxes_clamped,
        "n_rows_missing_image": s.n_rows_missing_image,
        "class_counts": dict(sorted(s.class_counts.items())),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Prepare the PlantDoc detection dataset.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config.")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download even if present."
    )
    parser.add_argument(
        "--drive-cache",
        metavar="DIR",
        help="After preparing, copy the processed dataset to DIR (e.g. a mounted "
        "Google Drive path) for reuse on Colab.",
    )
    args = parser.parse_args()

    result = prepare_dataset(args.config, force_download=args.force_download)
    print(json.dumps(result, indent=2))
    if args.drive_cache:
        sync_to_drive(args.drive_cache, config_path=args.config)
