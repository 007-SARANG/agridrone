"""Exploratory data analysis for the PlantDoc detection dataset.

Produces a standalone markdown report (``reports/eda_report.md``) plus PNG plots:
    - class balance (box counts per class, per split)
    - image size distribution
    - bounding-box count per image
    - a grid of sample images with ground-truth boxes drawn on

This is run after ``agridrone.data.prepare_dataset`` has produced the YOLO layout.
The report is the deliverable — not inline notebook cells (see CLAUDE.md).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display on the laptop
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from agridrone.config import load_config, resolve_path


def _read_yolo_labels(labels_dir: Path) -> pd.DataFrame:
    """Read all YOLO label txt files in a split into a tidy DataFrame."""
    rows: list[dict[str, Any]] = []
    for txt in sorted(labels_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, cx, cy, w, h = parts
            rows.append(
                {
                    "stem": txt.stem,
                    "class_id": int(cid),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                }
            )
    return pd.DataFrame(rows)


def _image_sizes(images_dir: Path) -> pd.DataFrame:
    """Collect (width, height) for every image in a split."""
    rows: list[dict[str, Any]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            with Image.open(img) as im:
                rows.append({"file": img.name, "width": im.width, "height": im.height})
        except Exception:  # noqa: BLE001 - skip corrupt images, they're reported separately
            rows.append({"file": img.name, "width": -1, "height": -1})
    return pd.DataFrame(rows)


def _plot_class_balance(
    label_dfs: dict[str, pd.DataFrame], class_names: list[str], out: Path
) -> None:
    counts = {}
    for split, df in label_dfs.items():
        if df.empty:
            counts[split] = pd.Series(0, index=range(len(class_names)))
        else:
            idx = range(len(class_names))
            counts[split] = df["class_id"].value_counts().reindex(idx, fill_value=0)
    counts_df = pd.DataFrame(counts).fillna(0)
    counts_df.index = [class_names[i] for i in counts_df.index]
    if "train" in counts_df:
        counts_df = counts_df.sort_values(by="train", ascending=True)

    ax = counts_df.plot(kind="barh", stacked=True, figsize=(10, max(6, len(class_names) * 0.35)))
    ax.set_xlabel("bounding boxes")
    ax.set_title("PlantDoc class balance (boxes per class, by split)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def _plot_image_sizes(size_dfs: dict[str, pd.DataFrame], out: Path) -> None:
    plt.figure(figsize=(8, 6))
    for split, df in size_dfs.items():
        valid = df[(df["width"] > 0) & (df["height"] > 0)]
        plt.scatter(valid["width"], valid["height"], s=8, alpha=0.4, label=split)
    plt.xlabel("width (px)")
    plt.ylabel("height (px)")
    plt.title("Image size distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def _plot_boxes_per_image(label_dfs: dict[str, pd.DataFrame], out: Path) -> None:
    plt.figure(figsize=(8, 5))
    for split, df in label_dfs.items():
        if df.empty:
            continue
        per_img = df.groupby("stem").size()
        plt.hist(per_img, bins=range(1, max(per_img.max() + 2, 3)), alpha=0.5, label=split)
    plt.xlabel("boxes per image")
    plt.ylabel("number of images")
    plt.title("Bounding boxes per image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def _plot_sample_grid(
    images_dir: Path, labels_dir: Path, class_names: list[str], n: int, seed: int, out: Path
) -> None:
    from matplotlib.patches import Rectangle

    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    if not imgs:
        return
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:n]

    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, img_path in zip(axes, imgs, strict=False):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            ax.imshow(im)
        label_file = labels_dir / (img_path.stem + ".txt")
        if label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                bw, bh = w * W, h * H
                x0, y0 = cx * W - bw / 2, cy * H - bh / 2
                ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, edgecolor="red", linewidth=2))
                ax.text(x0, max(y0 - 4, 0), class_names[cid], color="white", fontsize=7,
                        bbox={"facecolor": "red", "alpha": 0.7, "pad": 1})
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Sample PlantDoc images with ground-truth boxes", fontsize=14)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def generate_report(config_path: str | Path = "configs/data.yaml") -> Path:
    """Generate the full EDA markdown report + plots. Returns the report path."""
    cfg = load_config(config_path)
    processed = resolve_path(cfg["paths"]["processed_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    plots_dir = reports_dir / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = load_config(processed / "dataset.yaml")
    class_names = [dataset_yaml["names"][i] for i in range(dataset_yaml["nc"])]

    # Data-quality counters written by agridrone.data.prepare_dataset.
    prepare_summary: dict[str, Any] = {}
    summary_path = processed / "prepare_summary.json"
    if summary_path.exists():
        import json

        prepare_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    splits = ["train", "val", "test"]
    label_dfs = {s: _read_yolo_labels(processed / "labels" / s) for s in splits}
    size_dfs = {s: _image_sizes(processed / "images" / s) for s in splits}

    _plot_class_balance(label_dfs, class_names, plots_dir / "class_balance.png")
    _plot_image_sizes(size_dfs, plots_dir / "image_sizes.png")
    _plot_boxes_per_image(label_dfs, plots_dir / "boxes_per_image.png")
    _plot_sample_grid(
        processed / "images/train", processed / "labels/train",
        class_names, cfg["eda"]["n_sample_images"], cfg["eda"]["seed"],
        plots_dir / "sample_grid.png",
    )

    report_path = reports_dir / "eda_report.md"
    report_path.write_text(
        _render_markdown(cfg, class_names, label_dfs, size_dfs, prepare_summary),
        encoding="utf-8",
    )
    print(f"[eda] Report written to {report_path}")
    return report_path


def _render_markdown(
    cfg: dict[str, Any],
    class_names: list[str],
    label_dfs: dict[str, pd.DataFrame],
    size_dfs: dict[str, pd.DataFrame],
    prepare_summary: dict[str, Any] | None = None,
) -> str:
    def split_summary(s: str) -> tuple[int, int]:
        df = label_dfs[s]
        n_imgs = len(size_dfs[s])
        n_boxes = 0 if df.empty else len(df)
        return n_imgs, n_boxes

    lines: list[str] = []
    lines.append("# PlantDoc Detection — EDA Report\n")
    lines.append(f"**Dataset:** {cfg['dataset']['name']}  ")
    lines.append(f"**Source:** {cfg['dataset']['source_repo']}  ")
    lines.append(f"**License:** {cfg['dataset']['license']}\n")
    lines.append(
        "> PlantDoc is a genuine **object-detection** dataset (real bounding boxes on "
        "in-the-wild images), unlike PlantVillage which is classification-only. This is "
        "why the project can honestly claim detection + localization.\n"
    )

    lines.append("## Split sizes\n")
    lines.append("| Split | Images | Boxes | Avg boxes/img |")
    lines.append("|-------|-------:|------:|--------------:|")
    for s in ["train", "val", "test"]:
        n_imgs, n_boxes = split_summary(s)
        avg = f"{n_boxes / n_imgs:.2f}" if n_imgs else "0"
        lines.append(f"| {s} | {n_imgs} | {n_boxes} | {avg} |")
    lines.append("")

    # Data quality — box drops during CSV -> YOLO conversion (from data.py).
    lines.append("## Data quality (conversion drops)\n")
    if prepare_summary and prepare_summary.get("splits"):
        lines.append(
            "Counts of annotations dropped or adjusted while converting PlantDoc's "
            "CSV boxes to YOLO labels. Kept boxes exclude the dropped ones; clamped "
            "boxes are kept but had coordinates pulled back into `[0, 1]`.\n"
        )
        lines.append(
            "| Split | Kept boxes | Dropped (degenerate) | Dropped (bad dims) | "
            "Clamped | Rows w/ missing image |"
        )
        lines.append(
            "|-------|-----------:|---------------------:|-------------------:|"
            "--------:|----------------------:|"
        )
        tot = {"kept": 0, "deg": 0, "dims": 0, "clamp": 0, "miss": 0}
        for s in ["train", "val", "test"]:
            st = prepare_summary["splits"].get(s, {})
            kept = int(st.get("n_boxes", 0))
            deg = int(st.get("n_boxes_dropped_degenerate", 0))
            dims = int(st.get("n_boxes_dropped_bad_dims", 0))
            clamp = int(st.get("n_boxes_clamped", 0))
            miss = int(st.get("n_rows_missing_image", 0))
            tot["kept"] += kept
            tot["deg"] += deg
            tot["dims"] += dims
            tot["clamp"] += clamp
            tot["miss"] += miss
            lines.append(f"| {s} | {kept} | {deg} | {dims} | {clamp} | {miss} |")
        lines.append(
            f"| **total** | **{tot['kept']}** | **{tot['deg']}** | **{tot['dims']}** | "
            f"**{tot['clamp']}** | **{tot['miss']}** |"
        )
        lines.append("")
        lines.append(
            f"- **Degenerate boxes dropped:** {tot['deg']} — zero/negative-area boxes "
            "(`xmax <= xmin` or `ymax <= ymin`) that carry no usable localization signal.\n"
            f"- **Bad-dimension boxes dropped:** {tot['dims']} — CSV rows recording image "
            "`width` or `height` as 0 (annotation errors); normalized coords are undefined.\n"
            f"- **Boxes clamped:** {tot['clamp']} — boxes spilling past the image edge, "
            "kept with coordinates clamped to `[0, 1]`.\n"
            f"- **Rows referencing missing images:** {tot['miss']} — CSV rows whose image "
            "file was absent from the download; the whole image (and its boxes) is skipped.\n"
        )
    else:
        lines.append(
            "> No `prepare_summary.json` found in the processed dir. Run "
            "`make data` (which now records conversion drop counts) before `make eda` "
            "to populate this section.\n"
        )

    lines.append(f"## Classes ({len(class_names)})\n")
    lines.append("Per-class box counts (train / val / test):\n")
    lines.append("| ID | Class | Train | Val | Test |")
    lines.append("|---:|-------|------:|----:|-----:|")
    for i, name in enumerate(class_names):
        c = []
        for s in ["train", "val", "test"]:
            df = label_dfs[s]
            c.append(0 if df.empty else int((df["class_id"] == i).sum()))
        lines.append(f"| {i} | {name} | {c[0]} | {c[1]} | {c[2]} |")
    lines.append("")

    # Image size summary
    lines.append("## Image sizes\n")
    all_sizes = pd.concat([d[(d.width > 0)] for d in size_dfs.values()], ignore_index=True)
    if not all_sizes.empty:
        lines.append(f"- Width:  min {int(all_sizes.width.min())}, "
                     f"median {int(all_sizes.width.median())}, max {int(all_sizes.width.max())}")
        lines.append(f"- Height: min {int(all_sizes.height.min())}, "
                     f"median {int(all_sizes.height.median())}, max {int(all_sizes.height.max())}")
    corrupt = sum(int((d.width <= 0).sum()) for d in size_dfs.values())
    lines.append(f"- Unreadable/corrupt images: {corrupt}\n")

    lines.append("## Plots\n")
    for title, fname in [
        ("Class balance", "class_balance.png"),
        ("Image size distribution", "image_sizes.png"),
        ("Boxes per image", "boxes_per_image.png"),
        ("Sample images with boxes", "sample_grid.png"),
    ]:
        lines.append(f"### {title}\n")
        lines.append(f"![{title}](eda_plots/{fname})\n")

    lines.append("## Honest notes\n")
    lines.append(
        "- PlantDoc is small (~2.6k images) and scraped from the web, so images are "
        "noisy and class balance is skewed. Expect modest mAP — this is reported honestly "
        "rather than cherry-picked.\n"
        "- Some CSV rows reference missing images or contain degenerate (zero-area) boxes; "
        "these are skipped during conversion (see `agridrone.data`).\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the PlantDoc EDA report.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    generate_report(args.config)
