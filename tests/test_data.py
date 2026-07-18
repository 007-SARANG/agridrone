"""Tests for the PlantDoc data pipeline.

These run without any network access: we build a tiny synthetic dataset on disk
(a couple of images + CSVs in PlantDoc's schema) and check the CSV -> YOLO
conversion, the deterministic train/val split, and dataset.yaml generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from agridrone.data import (
    BoxRecord,
    _split_train_val,
    build_class_list,
    convert_split,
    write_dataset_yaml,
)

# ---------------------------------------------------------------------------
# BoxRecord.to_yolo
# ---------------------------------------------------------------------------


def test_to_yolo_center_and_size():
    # A 100x200 box in the top-left quadrant of a 200x400 image.
    box = BoxRecord("a.jpg", img_w=200, img_h=400, class_name="leaf",
                    xmin=0, ymin=0, xmax=100, ymax=200)
    line = box.to_yolo(class_id=3)
    cid, cx, cy, w, h = line.split()
    assert cid == "3"
    assert float(cx) == pytest.approx(0.25)   # center x = 50/200
    assert float(cy) == pytest.approx(0.25)   # center y = 100/400
    assert float(w) == pytest.approx(0.5)     # 100/200
    assert float(h) == pytest.approx(0.5)     # 200/400


def test_to_yolo_clamps_overflowing_box():
    # Box extends past image bounds — normalized values must clamp to [0, 1].
    box = BoxRecord("a.jpg", img_w=100, img_h=100, class_name="leaf",
                    xmin=-10, ymin=-10, xmax=200, ymax=200)
    _, cx, cy, w, h = box.to_yolo(0).split()
    for v in (cx, cy, w, h):
        assert 0.0 <= float(v) <= 1.0


# ---------------------------------------------------------------------------
# class list
# ---------------------------------------------------------------------------


def test_build_class_list_sorted_and_unique():
    train = pd.DataFrame({"class_name": ["tomato", "apple", "tomato"]})
    test = pd.DataFrame({"class_name": ["apple", "corn"]})
    assert build_class_list(train, test) == ["apple", "corn", "tomato"]


def test_build_class_list_strips_whitespace():
    train = pd.DataFrame({"class_name": [" leaf ", "leaf"]})
    test = pd.DataFrame({"class_name": ["leaf"]})
    assert build_class_list(train, test) == ["leaf"]


# ---------------------------------------------------------------------------
# train/val split determinism
# ---------------------------------------------------------------------------


def test_split_train_val_is_deterministic_and_disjoint():
    df = pd.DataFrame({"filename": [f"img_{i}.jpg" for i in range(20)]})
    tr1, val1 = _split_train_val(df, val_fraction=0.25, seed=42)
    tr2, val2 = _split_train_val(df, val_fraction=0.25, seed=42)
    assert (tr1, val1) == (tr2, val2)          # deterministic
    assert tr1.isdisjoint(val1)                 # no leakage
    assert len(val1) == 5                        # 25% of 20
    assert len(tr1) + len(val1) == 20            # all accounted for


def test_split_train_val_groups_boxes_by_image():
    # Two rows for the same image must never land in different splits.
    df = pd.DataFrame({"filename": ["a.jpg", "a.jpg", "b.jpg", "b.jpg"]})
    tr, val = _split_train_val(df, val_fraction=0.5, seed=1)
    assert ("a.jpg" in tr) != ("a.jpg" in val)  # in exactly one
    assert ("b.jpg" in tr) != ("b.jpg" in val)


# ---------------------------------------------------------------------------
# convert_split — full round trip on a synthetic dataset
# ---------------------------------------------------------------------------


def _make_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(0, 128, 0)).save(path)


def test_convert_split_writes_images_and_labels(tmp_path: Path):
    src = tmp_path / "TRAIN"
    _make_image(src / "leaf1.jpg", 100, 100)
    _make_image(src / "leaf2.jpg", 200, 100)

    df = pd.DataFrame(
        [
            {"filename": "leaf1.jpg", "width": 100, "height": 100,
             "class_name": "rust", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
            {"filename": "leaf2.jpg", "width": 200, "height": 100,
             "class_name": "blight", "xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
        ]
    )
    class_to_id = {"blight": 0, "rust": 1}
    stats = convert_split(
        df, src, tmp_path / "out/images/train", tmp_path / "out/labels/train", class_to_id
    )

    assert stats.n_images == 2
    assert stats.n_boxes == 2
    assert (tmp_path / "out/images/train/leaf1.jpg").exists()
    label = (tmp_path / "out/labels/train/leaf1.txt").read_text().strip()
    assert label.startswith("1 ")  # rust -> class id 1
    assert stats.class_counts == {"rust": 1, "blight": 1}


def test_convert_split_skips_missing_images_and_bad_boxes(tmp_path: Path):
    src = tmp_path / "TRAIN"
    _make_image(src / "present.jpg", 100, 100)

    df = pd.DataFrame(
        [
            # references an image that doesn't exist -> skipped entirely
            {"filename": "ghost.jpg", "width": 100, "height": 100,
             "class_name": "rust", "xmin": 1, "ymin": 1, "xmax": 9, "ymax": 9},
            # degenerate zero-area box -> box skipped, image kept with empty label
            {"filename": "present.jpg", "width": 100, "height": 100,
             "class_name": "rust", "xmin": 50, "ymin": 50, "xmax": 50, "ymax": 50},
        ]
    )
    stats = convert_split(
        df, src, tmp_path / "out/images/train", tmp_path / "out/labels/train", {"rust": 0}
    )
    assert stats.n_images == 1                 # ghost.jpg skipped
    assert stats.n_boxes == 0                  # degenerate box skipped
    assert stats.n_images_without_boxes == 1
    assert stats.n_boxes_dropped_degenerate == 1
    assert stats.n_rows_missing_image == 1     # the single ghost.jpg row
    assert (tmp_path / "out/labels/train/present.txt").read_text() == ""


def test_convert_split_counts_clamped_boxes(tmp_path: Path):
    src = tmp_path / "TRAIN"
    _make_image(src / "spill.jpg", 100, 100)

    df = pd.DataFrame(
        [
            # box spills past the right/bottom edge -> kept but clamped
            {"filename": "spill.jpg", "width": 100, "height": 100,
             "class_name": "rust", "xmin": 10, "ymin": 10, "xmax": 140, "ymax": 130},
        ]
    )
    stats = convert_split(
        df, src, tmp_path / "out/images/train", tmp_path / "out/labels/train", {"rust": 0}
    )
    assert stats.n_boxes == 1                  # kept
    assert stats.n_boxes_clamped == 1          # but flagged as clamped
    assert stats.n_boxes_dropped_degenerate == 0


def test_needs_clamp_detects_in_and_out_of_bounds():
    inside = BoxRecord("a.jpg", img_w=100, img_h=100, class_name="x",
                       xmin=10, ymin=10, xmax=90, ymax=90)
    outside = BoxRecord("a.jpg", img_w=100, img_h=100, class_name="x",
                        xmin=10, ymin=10, xmax=140, ymax=90)
    assert not inside.needs_clamp()
    assert outside.needs_clamp()


def test_write_dataset_yaml_roundtrip(tmp_path: Path):
    import yaml

    out = write_dataset_yaml(tmp_path, ["apple", "corn", "tomato"])
    parsed = yaml.safe_load(out.read_text())
    assert parsed["nc"] == 3
    assert parsed["names"] == {0: "apple", 1: "corn", 2: "tomato"}
    assert parsed["train"] == "images/train"
