"""Smoke test for the EDA report generator.

Builds a tiny synthetic YOLO-format dataset on disk, points a minimal config at
it, and checks that the report + all four plots are produced. This runs offline
(no dataset download) and is fast.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from agridrone.data import (
    _split_train_val,
    build_class_list,
    convert_split,
    write_dataset_yaml,
)


@pytest.fixture
def synthetic_processed(tmp_path: Path) -> Path:
    """Create a small YOLO-layout dataset and return its processed dir."""
    src = tmp_path / "TRAIN"
    src.mkdir()
    rows = []
    for i in range(8):
        w = 120 + i * 5
        Image.new("RGB", (w, 100), (0, 120, 0)).save(src / f"img{i}.jpg")
        rows.append(
            {
                "filename": f"img{i}.jpg", "width": w, "height": 100,
                "class_name": ["rust", "blight"][i % 2],
                "xmin": 10, "ymin": 10, "xmax": 60, "ymax": 60,
            }
        )
    df = pd.DataFrame(rows)
    classes = build_class_list(df, df)
    c2i = {n: i for i, n in enumerate(classes)}
    proc = tmp_path / "processed"
    tr, val = _split_train_val(df, 0.25, 42)
    convert_split(df, src, proc / "images/train", proc / "labels/train", c2i, filenames=tr)
    convert_split(df, src, proc / "images/val", proc / "labels/val", c2i, filenames=val)
    convert_split(df, src, proc / "images/test", proc / "labels/test", c2i)
    write_dataset_yaml(proc, classes)
    return proc


def test_generate_report_produces_markdown_and_plots(
    tmp_path: Path, synthetic_processed: Path
):
    from agridrone.eda import generate_report

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset: {name: Synthetic, source_repo: x, license: CC-BY-4.0}\n"
        f"paths: {{raw_dir: {tmp_path}/raw, processed_dir: {synthetic_processed}, "
        f"reports_dir: {tmp_path}/reports}}\n"
        "eda: {n_sample_images: 6, seed: 42}\n",
        encoding="utf-8",
    )

    report = generate_report(cfg)

    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "EDA Report" in text
    assert "## Classes" in text

    plots_dir = tmp_path / "reports" / "eda_plots"
    for name in ["class_balance.png", "image_sizes.png", "boxes_per_image.png", "sample_grid.png"]:
        assert (plots_dir / name).exists(), f"missing plot {name}"
