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


# --- Oversampling report (pure, no ML stack) -------------------------------


def _sampling_stats():
    """Stats as captured by the weighted dataset: class1 rare, class0 common."""
    from agridrone.weighting import (
        class_instance_counts,
        class_weights,
        image_weights,
        sampling_probabilities,
    )

    per_image = [[0, 0], [0, 0], [0, 0], [0, 1]]
    counts = class_instance_counts(per_image, n_classes=2)
    cw = class_weights(counts)
    probs = sampling_probabilities(image_weights(per_image, cw, agg="mean"))
    return {
        "names": {0: "Common", 1: "Rare"},
        "n_classes": 2,
        "counts": counts,
        "probabilities": probs,
        "per_image": per_image,
    }


def test_format_sampling_report_shows_amplification():
    md = train.format_sampling_report(_sampling_stats())
    assert train._SAMPLING_HEADING in md
    assert "Common" in md and "Rare" in md
    # Rarest class is listed first (rarity-ordered), before the common one.
    assert md.index("| 1 | Rare") < md.index("| 0 | Common")
    # Columns present.
    assert "Effective/epoch" in md and "| x |" in md


def test_write_sampling_report_upserts_alongside_results(tmp_path):
    out = tmp_path / "phase2_results.md"
    cfg = {"output": {"results_table": str(out), "sampling_report": ""}}

    # A pre-existing accuracy section (as the eval stage would write).
    out.write_text(
        f"{train._RESULTS_H1}\n\n## {train._RESULTS_HEADING}\n\n"
        "**mAP@0.5:** 0.5000\n",
        encoding="utf-8",
    )
    train.write_sampling_report(cfg, _sampling_stats())
    text = out.read_text(encoding="utf-8")

    # Both sections coexist under a single H1.
    assert text.count(train._RESULTS_H1) == 1
    assert train._RESULTS_HEADING in text
    assert train._SAMPLING_HEADING in text
    assert "**mAP@0.5:** 0.5000" in text


def test_write_sampling_report_replaces_stale_section(tmp_path):
    out = tmp_path / "phase2_results.md"
    cfg = {"output": {"results_table": str(out), "sampling_report": ""}}

    train.write_sampling_report(cfg, _sampling_stats())
    train.write_sampling_report(cfg, _sampling_stats())  # re-run
    text = out.read_text(encoding="utf-8")

    # Re-running does not duplicate the section.
    assert text.count(f"## {train._SAMPLING_HEADING}") == 1


def test_write_sampling_report_honors_separate_path(tmp_path):
    results = tmp_path / "results.md"
    sampling = tmp_path / "sampling.md"
    cfg = {"output": {"results_table": str(results), "sampling_report": str(sampling)}}

    train.write_sampling_report(cfg, _sampling_stats())
    assert sampling.exists()
    assert not results.exists()  # went to the dedicated file, not the results one


def test_upsert_section_preserves_siblings():
    doc = (
        f"{train._RESULTS_H1}\n\n## {train._RESULTS_HEADING}\n\nACCURACY BODY\n"
    )
    merged = train._upsert_section(doc, train._SAMPLING_HEADING, "## X\n\nNEW\n")
    assert "ACCURACY BODY" in merged
    assert "NEW" in merged
    assert merged.count(train._RESULTS_H1) == 1
