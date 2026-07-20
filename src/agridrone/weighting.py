"""Class-balancing math for imbalanced detection training.

PlantDoc's 27 kept classes are heavily skewed (e.g. ``Blueberry leaf`` ~718 train
boxes vs ``Corn Gray leaf spot`` ~66). To stop head classes from drowning out the
tail, we oversample images that contain rare classes rather than reweighting the
loss term. Oversampling is the version-stable choice for Ultralytics *detection*
(8.4.x exposes no built-in class-weighted-loss flag for detect), and it avoids the
sharp gradient spikes that hard loss-weighting can cause when a rare class finally
appears in a batch.

This module is deliberately **pure numpy** — it imports no ultralytics/torch — so
the weighting logic is unit-testable in the lightweight environment. ``train.py``
wires these functions into a ``YOLODataset`` subclass; the actual sampling happens
there.

The pipeline mirrors the community weighted-dataloader approach:

    counts   = class_instance_counts(labels, n_classes)   # boxes per class
    cls_w    = class_weights(counts)                       # total / count
    img_w    = image_weights(per_image_class_ids, cls_w)   # aggregate per image
    probs    = sampling_probabilities(img_w)               # normalize to sum 1

``probs`` then feeds ``np.random.choice`` at __getitem__ time so rare-class images
are drawn more often.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

# Aggregation functions allowed for turning an image's per-class weights into a
# single per-image weight. Exposed by name so the config stays declarative.
AGG_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(np.mean(a)),
    "max": lambda a: float(np.max(a)),
    "median": lambda a: float(np.median(a)),
    "sum": lambda a: float(np.sum(a)),
}

# Weight assigned to a background image (no boxes); matches the reference
# implementation, which keeps such images in the pool at neutral weight.
DEFAULT_BACKGROUND_WEIGHT = 1.0


def class_instance_counts(
    labels: Iterable[Sequence[int]], n_classes: int
) -> np.ndarray:
    """Tally bounding-box instances per class across every image.

    Args:
        labels: One entry per image; each entry is that image's list of class
            IDs (one ID per box). Empty lists (background images) contribute
            nothing.
        n_classes: Total number of classes. IDs must be in ``[0, n_classes)``.

    Returns:
        Integer array of shape ``(n_classes,)`` with per-class box counts.

    Raises:
        ValueError: If ``n_classes`` is not positive or a class ID is out of range.
    """
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}.")
    counts = np.zeros(n_classes, dtype=np.int64)
    for img_classes in labels:
        for cid in img_classes:
            c = int(cid)
            if c < 0 or c >= n_classes:
                raise ValueError(
                    f"class id {c} out of range for n_classes={n_classes}."
                )
            counts[c] += 1
    return counts


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights: ``total_instances / class_count``.

    Rare classes get large weights, common classes small ones. Zero counts are
    replaced with 1 before dividing so absent classes do not blow up to infinity
    (they simply receive weight ``total``, which is harmless — no image references
    them, so the weight is never actually aggregated).

    Args:
        counts: Per-class instance counts (see :func:`class_instance_counts`).

    Returns:
        Float array of shape ``(n_classes,)`` with per-class weights. If there
        are no instances at all, returns all-ones (nothing to balance).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.ones_like(counts)
    safe = np.where(counts <= 0, 1.0, counts)
    return total / safe


def image_weights(
    per_image_class_ids: Iterable[Sequence[int]],
    cls_weights: np.ndarray,
    agg: str = "mean",
    *,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
) -> np.ndarray:
    """Aggregate each image's per-class weights into one sampling weight.

    An image containing several boxes is reduced to a single scalar via ``agg``.
    ``mean`` (the default) gives a smooth, well-behaved distribution; ``max``
    emphasizes an image as much as its rarest class; ``sum`` biases toward images
    with many boxes.

    Args:
        per_image_class_ids: One class-ID list per image (same order/length as
            the images to be sampled).
        cls_weights: Per-class weights from :func:`class_weights`.
        agg: Aggregation name, one of :data:`AGG_FUNCS`.
        background_weight: Weight for images with no boxes.

    Returns:
        Float array of shape ``(n_images,)`` of per-image weights.

    Raises:
        ValueError: If ``agg`` is unknown or a class ID is out of range.
    """
    if agg not in AGG_FUNCS:
        raise ValueError(
            f"Unknown agg '{agg}'. Choose one of {sorted(AGG_FUNCS)}."
        )
    agg_fn = AGG_FUNCS[agg]
    cls_weights = np.asarray(cls_weights, dtype=np.float64)
    n_classes = cls_weights.shape[0]

    weights: list[float] = []
    for img_classes in per_image_class_ids:
        ids = [int(c) for c in img_classes]
        if not ids:
            weights.append(float(background_weight))
            continue
        for c in ids:
            if c < 0 or c >= n_classes:
                raise ValueError(
                    f"class id {c} out of range for {n_classes} class weights."
                )
        weights.append(agg_fn(cls_weights[ids]))
    return np.asarray(weights, dtype=np.float64)


def sampling_probabilities(weights: np.ndarray) -> np.ndarray:
    """Normalize per-image weights into a probability distribution (sums to 1).

    Args:
        weights: Non-negative per-image weights from :func:`image_weights`.

    Returns:
        Float array of shape ``(n_images,)`` summing to 1. If all weights are
        zero (or the input is empty in total mass), falls back to a uniform
        distribution so sampling still works.

    Raises:
        ValueError: If any weight is negative.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return weights
    if np.any(weights < 0):
        raise ValueError("Sampling weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0:
        # Degenerate: no signal to balance on, sample uniformly.
        return np.full_like(weights, 1.0 / weights.size)
    return weights / total
