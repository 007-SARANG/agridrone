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


def class_weights(
    counts: np.ndarray, *, power: float = 1.0, max_ratio: float | None = None
) -> np.ndarray:
    """Inverse-frequency class weights, with optional dampening and a cap.

    The base weight is ``(total_instances / class_count) ** power``:

    - ``power=1.0`` — **pure** inverse frequency (full equalization). The rarest
      classes get enormous weights; on a long-tailed set like PlantDoc this can
      amplify a ~15-box class 10x+, so training oversamples the same few images
      over and over and risks memorizing them instead of generalizing.
    - ``power=0.5`` — **sqrt dampening**. Still favors rare classes but far more
      gently (a 42x raw weight ratio becomes ~6.5x), the standard mitigation for
      the overfitting risk above.
    - ``power=0.0`` — uniform (no balancing).

    ``max_ratio`` then hard-caps how strongly the rarest class may be weighted
    relative to the most common *present* class: weights are clipped so
    ``max(weight) <= min_present_weight * max_ratio``. This bounds the realized
    oversampling amplification regardless of how extreme the raw imbalance is.
    ``None`` disables the cap.

    Zero counts are replaced with 1 before dividing so absent classes do not blow
    up to infinity (they are excluded from the cap's reference min and are never
    aggregated anyway, since no image references them).

    Args:
        counts: Per-class instance counts (see :func:`class_instance_counts`).
        power: Inverse-frequency exponent (>= 0). See above.
        max_ratio: Optional cap (>= 1) on the rarest/commonest weight ratio.

    Returns:
        Float array of shape ``(n_classes,)`` with per-class weights. If there
        are no instances at all, returns all-ones (nothing to balance).

    Raises:
        ValueError: If ``power`` is negative or ``max_ratio`` is < 1.
    """
    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}.")
    if max_ratio is not None and max_ratio < 1:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}.")
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.ones_like(counts)
    safe = np.where(counts <= 0, 1.0, counts)
    weights = (total / safe) ** power
    if max_ratio is not None:
        present = counts > 0
        if present.any():
            floor = float(weights[present].min())
            weights = np.minimum(weights, floor * float(max_ratio))
    return weights


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


def expected_class_exposure(
    per_image_class_ids: Iterable[Sequence[int]],
    probabilities: np.ndarray,
    n_classes: int,
    n_draws: int | None = None,
) -> np.ndarray:
    """Expected per-class box instances seen in one oversampled epoch.

    Each of ``n_draws`` samples in an epoch is an image drawn (with replacement)
    according to ``probabilities``. The expected number of class-``c`` boxes seen
    across those draws is::

        n_draws * sum_i  p_i * (boxes of class c in image i)

    With ``n_draws`` defaulting to the image count (one Ultralytics epoch draws
    exactly ``len(dataset)`` samples), this is directly comparable to the raw
    per-class box counts from :func:`class_instance_counts`: under *uniform*
    sampling (``p_i = 1/N``) the formula collapses back to the raw counts, so the
    ratio ``effective / raw`` is a clean amplification factor showing how much more
    (or less) often each class is seen once oversampling is active.

    Args:
        per_image_class_ids: One class-ID list per image (same order/length as
            ``probabilities``).
        probabilities: Per-image sampling distribution from
            :func:`sampling_probabilities`.
        n_classes: Total number of classes. IDs must be in ``[0, n_classes)``.
        n_draws: Samples drawn per epoch; defaults to the number of images.

    Returns:
        Float array of shape ``(n_classes,)`` of expected box instances per epoch.

    Raises:
        ValueError: If ``n_classes`` is not positive, the lengths disagree, or a
            class ID is out of range.
    """
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    per_image = list(per_image_class_ids)
    if len(per_image) != probabilities.shape[0]:
        raise ValueError(
            f"per_image_class_ids ({len(per_image)}) and probabilities "
            f"({probabilities.shape[0]}) must have the same length."
        )
    if n_draws is None:
        n_draws = len(per_image)

    exposure = np.zeros(n_classes, dtype=np.float64)
    for p, img_classes in zip(probabilities, per_image, strict=True):
        for cid in img_classes:
            c = int(cid)
            if c < 0 or c >= n_classes:
                raise ValueError(
                    f"class id {c} out of range for n_classes={n_classes}."
                )
            exposure[c] += float(p)
    return exposure * float(n_draws)
