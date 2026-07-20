"""Tests for agridrone.train.

The module must import and expose its API in the lightweight env (no torch /
ultralytics). Functions that actually train are import-guarded; here we cover the
parts reachable without the ML stack, plus a skipif-guarded check of the weighted
dataset wiring when ultralytics happens to be installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from agridrone import train


def _has_ultralytics() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def test_module_imports_without_ml_stack():
    # Public entry points exist regardless of whether ultralytics is installed.
    for name in (
        "pretrain_backbone",
        "finetune_detector",
        "evaluate",
        "enable_weighted_sampling",
        "resolve_dataset_yaml",
        "run",
    ):
        assert hasattr(train, name)


@pytest.mark.skipif(_has_ultralytics(), reason="ultralytics present; guard not raised")
def test_require_ultralytics_raises_helpful_message():
    with pytest.raises(RuntimeError, match="ultralytics"):
        train._require_ultralytics()


@pytest.mark.skipif(_has_ultralytics(), reason="ultralytics present; guard not raised")
def test_enable_weighted_sampling_guarded_without_stack():
    with pytest.raises(RuntimeError):
        train.enable_weighted_sampling()


def test_resolve_dataset_yaml_prefers_drive(tmp_path):
    # Drive copy present -> it wins over the local fallback.
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "dataset.yaml").write_text("path: x\n", encoding="utf-8")
    cfg = {
        "data": {
            "drive_dataset_dir": str(drive),
            "dataset_yaml": str(tmp_path / "local/dataset.yaml"),
        }
    }
    assert train.resolve_dataset_yaml(cfg) == drive / "dataset.yaml"


def test_resolve_dataset_yaml_falls_back_to_local(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "dataset.yaml").write_text("path: x\n", encoding="utf-8")
    cfg = {
        "data": {
            "drive_dataset_dir": "",
            "dataset_yaml": str(local / "dataset.yaml"),
        }
    }
    assert train.resolve_dataset_yaml(cfg) == local / "dataset.yaml"


def test_resolve_dataset_yaml_missing_raises(tmp_path):
    cfg = {
        "data": {
            "drive_dataset_dir": "",
            "dataset_yaml": str(tmp_path / "nope/dataset.yaml"),
        }
    }
    with pytest.raises(FileNotFoundError):
        train.resolve_dataset_yaml(cfg)


@pytest.mark.skipif(not _has_ultralytics(), reason="needs ultralytics")
def test_weighted_dataset_class_builds():  # pragma: no cover - only with full stack
    cls = train.make_weighted_dataset_class(agg="mean")
    from ultralytics.data.dataset import YOLODataset

    assert issubclass(cls, YOLODataset)
