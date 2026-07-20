"""Tests for the pure-numpy class-balancing math (agridrone.weighting).

These run in the lightweight env (no torch/ultralytics needed) and cover the
logic that decides how strongly rare-class images get oversampled.
"""

from __future__ import annotations

import numpy as np
import pytest

from agridrone import weighting as w


def test_class_instance_counts_basic():
    labels = [[0, 0, 1], [1], [2, 2], []]  # img4 is background
    counts = w.class_instance_counts(labels, n_classes=3)
    assert counts.tolist() == [2, 2, 2]


def test_class_instance_counts_rejects_bad_n_classes():
    with pytest.raises(ValueError):
        w.class_instance_counts([[0]], n_classes=0)


def test_class_instance_counts_rejects_out_of_range_id():
    with pytest.raises(ValueError):
        w.class_instance_counts([[3]], n_classes=3)


def test_class_weights_inverse_frequency():
    # 8 rare vs 2 common -> rare class gets the larger weight.
    counts = np.array([8, 2])
    cw = w.class_weights(counts)
    # total=10: class0 -> 10/8=1.25, class1 -> 10/2=5.0
    assert cw[0] == pytest.approx(1.25)
    assert cw[1] == pytest.approx(5.0)
    assert cw[1] > cw[0]  # the rarer class is up-weighted


def test_class_weights_zero_count_guard():
    # An absent class must not produce inf/nan.
    counts = np.array([5, 0, 5])
    cw = w.class_weights(counts)
    assert np.all(np.isfinite(cw))
    assert cw[1] == pytest.approx(10.0)  # total/1


def test_class_weights_all_zero_returns_ones():
    cw = w.class_weights(np.zeros(4))
    assert cw.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_image_weights_mean_default():
    cw = np.array([1.0, 5.0])
    # image with both classes -> mean(1,5)=3; image with only rare -> 5
    iw = w.image_weights([[0, 1], [1]], cw, agg="mean")
    assert iw[0] == pytest.approx(3.0)
    assert iw[1] == pytest.approx(5.0)


def test_image_weights_background_gets_default():
    cw = np.array([1.0, 5.0])
    iw = w.image_weights([[]], cw, background_weight=1.0)
    assert iw[0] == pytest.approx(1.0)


def test_image_weights_agg_variants():
    cw = np.array([1.0, 5.0])
    ids = [[0, 1]]
    assert w.image_weights(ids, cw, agg="max")[0] == pytest.approx(5.0)
    assert w.image_weights(ids, cw, agg="sum")[0] == pytest.approx(6.0)
    assert w.image_weights(ids, cw, agg="median")[0] == pytest.approx(3.0)


def test_image_weights_rejects_unknown_agg():
    with pytest.raises(ValueError):
        w.image_weights([[0]], np.array([1.0]), agg="nope")


def test_image_weights_rejects_out_of_range_id():
    with pytest.raises(ValueError):
        w.image_weights([[2]], np.array([1.0, 1.0]), agg="mean")


def test_sampling_probabilities_sum_to_one():
    probs = w.sampling_probabilities(np.array([1.0, 3.0, 6.0]))
    assert probs.sum() == pytest.approx(1.0)
    # monotonic: larger weight -> larger probability
    assert probs[0] < probs[1] < probs[2]


def test_sampling_probabilities_uniform_when_all_zero():
    probs = w.sampling_probabilities(np.zeros(4))
    assert probs.sum() == pytest.approx(1.0)
    assert np.allclose(probs, 0.25)


def test_rare_class_image_oversampled_end_to_end():
    # Two classes: class0 common (8 boxes across 4 imgs), class1 rare (1 box).
    labels = [[0, 0], [0, 0], [0, 0], [0, 1]]
    counts = w.class_instance_counts(labels, n_classes=2)
    cw = w.class_weights(counts)
    iw = w.image_weights(labels, cw, agg="mean")
    probs = w.sampling_probabilities(iw)
    # The image containing the rare class must be the most likely to be drawn.
    assert int(np.argmax(probs)) == 3
