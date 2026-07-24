"""Tests for agridrone.benchmark pure helpers.

Latency statistics, size reporting, speedup annotation, and report formatting
all run without the ML stack. The timing/accuracy/plot paths are guarded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agridrone import benchmark as bm


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# ---------------------------------------------------------------------------
# latency_stats
# ---------------------------------------------------------------------------


def test_latency_stats_basic():
    # 10ms .. 50ms in seconds
    timings = [0.01, 0.02, 0.03, 0.04, 0.05]
    s = bm.latency_stats(timings)
    assert s["mean_ms"] == pytest.approx(30.0)
    assert s["median_ms"] == pytest.approx(30.0)
    assert s["min_ms"] == pytest.approx(10.0)
    assert s["p95_ms"] == pytest.approx(50.0)  # top of a 5-sample set


def test_latency_stats_empty_is_zeros():
    s = bm.latency_stats([])
    assert s == {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0}


def test_latency_stats_median_even_count():
    s = bm.latency_stats([0.01, 0.03])  # mean of the two middle -> 20ms
    assert s["median_ms"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# file_size_mb
# ---------------------------------------------------------------------------


def test_file_size_mb(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\0" * 2_000_000)  # 2 MB
    assert bm.file_size_mb(p) == pytest.approx(2.0)


def test_file_size_mb_missing_is_zero(tmp_path: Path):
    assert bm.file_size_mb(tmp_path / "nope.bin") == 0.0


# ---------------------------------------------------------------------------
# _speedup_column — relative to pytorch_fp32 baseline
# ---------------------------------------------------------------------------


def test_speedup_relative_to_baseline():
    rows = [
        {"key": "pytorch_fp32", "mean_ms": 100.0},
        {"key": "onnx_fp32", "mean_ms": 50.0},
        {"key": "onnx_int8", "mean_ms": 25.0},
    ]
    bm._speedup_column(rows)
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[1]["speedup"] == pytest.approx(2.0)
    assert rows[2]["speedup"] == pytest.approx(4.0)


def test_speedup_none_without_baseline():
    rows = [{"key": "onnx_fp32", "mean_ms": 50.0}]
    bm._speedup_column(rows)
    assert rows[0]["speedup"] is None


# ---------------------------------------------------------------------------
# format_report — honest blanks, real numbers
# ---------------------------------------------------------------------------


def _row(key, label, mean, map50=None, mapc=None, size=5.0, src="ultralytics.val()"):
    return {
        "key": key, "label": label, "mean_ms": mean, "median_ms": mean,
        "p95_ms": mean, "map50": map50, "map": mapc, "size_mb": size,
        "accuracy_source": src,
    }


def test_format_report_has_all_rows_and_headers():
    rows = [
        _row("pytorch_fp32", "PyTorch FP32", 100.0, 0.60, 0.48, 5.4),
        _row("onnx_int8", "ONNX INT8", 40.0, 0.58, 0.46, 1.5),
    ]
    md = bm.format_report(rows)
    assert "# Phase 3" in md
    assert "PyTorch FP32" in md and "ONNX INT8" in md
    assert "mAP@0.5" in md
    assert "2.50×" in md  # 100/40 speedup on the INT8 row


def test_format_report_missing_accuracy_shows_dash():
    rows = [_row("onnx_int8", "ONNX INT8", 40.0, map50=None, mapc=None, src="failed: X")]
    md = bm.format_report(rows)
    # No fabricated 0.0000 — an em dash stands in for the missing number.
    assert "—" in md
    assert "failed: X" in md  # provenance recorded honestly


# ---------------------------------------------------------------------------
# _viability_note — flag rows whose accuracy collapsed or is missing
# ---------------------------------------------------------------------------


def test_viability_note_flags_zero_map():
    note = bm._viability_note(_row("onnx_int8", "ONNX INT8", 40.0, map50=0.0))
    assert note.startswith("not viable — ")
    assert "collapses to 0.0 mAP" in note  # uses the INT8-specific reason


def test_viability_note_flags_missing_map():
    note = bm._viability_note(_row("onnx_int8", "ONNX INT8", 40.0, map50=None))
    assert note.startswith("not viable — ")


def test_viability_note_empty_for_viable_row():
    note = bm._viability_note(_row("onnx_fp32", "ONNX FP32", 50.0, map50=0.60))
    assert note == ""


# ---------------------------------------------------------------------------
# _recommend — preference order onnx_fp32 -> onnx_fp16 -> pytorch_fp32
# ---------------------------------------------------------------------------


def test_recommend_prefers_onnx_fp32():
    rows = [
        _row("pytorch_fp32", "PyTorch FP32", 100.0, map50=0.60),
        _row("onnx_fp32", "ONNX FP32", 50.0, map50=0.60),
        _row("onnx_fp16", "ONNX FP16", 50.0, map50=0.60),
        _row("onnx_int8", "ONNX INT8", 40.0, map50=0.0),
    ]
    assert bm._recommend(rows) == "Recommended deployment artifact: ONNX FP32."


def test_recommend_falls_back_to_fp16():
    rows = [
        _row("pytorch_fp32", "PyTorch FP32", 100.0, map50=0.60),
        _row("onnx_fp32", "ONNX FP32", 50.0, map50=0.0),  # not viable
        _row("onnx_fp16", "ONNX FP16", 50.0, map50=0.60),
    ]
    assert bm._recommend(rows) == "Recommended deployment artifact: ONNX FP16."


def test_recommend_falls_back_to_pytorch():
    rows = [
        _row("pytorch_fp32", "PyTorch FP32", 100.0, map50=0.60),
        _row("onnx_fp32", "ONNX FP32", 50.0, map50=None),  # not viable
        _row("onnx_fp16", "ONNX FP16", 50.0, map50=0.0),  # not viable
    ]
    assert bm._recommend(rows) == "Recommended deployment artifact: PyTorch FP32."


# ---------------------------------------------------------------------------
# Guarded heavy path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_has("onnxruntime"), reason="onnxruntime present; guard not raised")
def test_require_raises_without_stack():
    with pytest.raises(RuntimeError, match="not installed"):
        bm._require("onnxruntime")
