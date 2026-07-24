"""Tests for agridrone.export.

Cover the pure helpers (image listing, deterministic calibration selection,
preprocessing) that run without torch/ultralytics/onnxruntime. The heavy
export/quantize paths are guarded and only exercised with the full CPU stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agridrone import export


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _make_image(path: Path, w: int = 64, h: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(0, 120, 0)).save(path)


# ---------------------------------------------------------------------------
# list_split_images
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path: Path, split: str, n: int) -> Path:
    root = tmp_path / "proc"
    for i in range(n):
        _make_image(root / "images" / split / f"img_{i}.jpg")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(f"path: {root}\n", encoding="utf-8")
    return yaml_path


def test_list_split_images_returns_sorted(tmp_path: Path):
    ds = _write_dataset(tmp_path, "train", 3)
    imgs = export.list_split_images(ds, "train")
    assert len(imgs) == 3
    assert imgs == sorted(imgs)
    assert all(p.suffix == ".jpg" for p in imgs)


def test_list_split_images_missing_split_raises(tmp_path: Path):
    ds = _write_dataset(tmp_path, "train", 1)
    with pytest.raises(FileNotFoundError):
        export.list_split_images(ds, "test")


# ---------------------------------------------------------------------------
# select_calibration_images — determinism + bounds
# ---------------------------------------------------------------------------


def test_calibration_subset_is_deterministic():
    pool = [Path(f"{i}.jpg") for i in range(100)]
    a = export.select_calibration_images(pool, 10, seed=42)
    b = export.select_calibration_images(pool, 10, seed=42)
    assert a == b
    assert len(a) == 10
    assert len(set(a)) == 10  # no duplicates (sampling without replacement)


def test_calibration_subset_returns_all_when_pool_small():
    pool = [Path(f"{i}.jpg") for i in range(5)]
    out = export.select_calibration_images(pool, 10, seed=1)
    assert sorted(out) == sorted(pool)


def test_calibration_different_seeds_differ():
    pool = [Path(f"{i}.jpg") for i in range(100)]
    a = export.select_calibration_images(pool, 10, seed=1)
    b = export.select_calibration_images(pool, 10, seed=2)
    assert a != b  # overwhelmingly likely for 10-of-100


# ---------------------------------------------------------------------------
# preprocess_image — shape / dtype / range
# ---------------------------------------------------------------------------


def test_preprocess_shape_and_range(tmp_path: Path):
    p = tmp_path / "x.jpg"
    _make_image(p, 200, 100)
    arr = export.preprocess_image(p, imgsz=640)
    assert arr.shape == (1, 3, 640, 640)  # batch, CHW, square
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


# ---------------------------------------------------------------------------
# find_head_nodes_to_exclude (pure graph walk — no onnx needed)
# ---------------------------------------------------------------------------


class _FakeNode:
    """Minimal stand-in for an ONNX NodeProto (name, op_type, input, output)."""

    def __init__(self, name, op_type, inputs, outputs):
        self.name = name
        self.op_type = op_type
        self.input = inputs
        self.output = outputs


def _toy_detector_graph():
    """Backbone Conv -> head Conv/MatMul -> TopK marker. Only the head feeds TopK."""
    return [
        _FakeNode("backbone_conv0", "Conv", ["x"], ["b0"]),
        _FakeNode("backbone_conv1", "Conv", ["b0"], ["b1"]),
        _FakeNode("head_conv", "Conv", ["b1"], ["h0"]),
        _FakeNode("head_matmul", "MatMul", ["h0"], ["h1"]),
        _FakeNode("select", "TopK", ["h1"], ["out"]),  # marker op
    ]


def test_find_head_nodes_excludes_head_layers():
    excluded = export.find_head_nodes_to_exclude(_toy_detector_graph())
    assert "head_conv" in excluded
    assert "head_matmul" in excluded


def test_find_head_nodes_respects_depth_limit():
    # With max_depth=1, only the layer directly feeding TopK is reached.
    excluded = export.find_head_nodes_to_exclude(_toy_detector_graph(), max_depth=1)
    assert excluded == ["head_matmul"]


def test_find_head_nodes_empty_when_no_markers():
    nodes = [_FakeNode("c0", "Conv", ["x"], ["y"]), _FakeNode("c1", "Conv", ["y"], ["z"])]
    assert export.find_head_nodes_to_exclude(nodes) == []


# ---------------------------------------------------------------------------
# Guarded heavy path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_has("torch"), reason="torch present; guard not raised")
def test_export_requires_stack_when_absent():
    with pytest.raises(RuntimeError, match="not installed"):
        export._require("torch")
