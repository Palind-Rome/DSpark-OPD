from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .np_utils import logit, sigmoid


@dataclass(frozen=True)
class CalibrationResult:
    temperatures: np.ndarray
    before_ece: np.ndarray
    after_ece: np.ndarray


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Order-preserving temperature scaling for probabilities."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return sigmoid(logit(probabilities) / float(temperature))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """ECE for binary soft/hard acceptance labels."""

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labs = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.shape != labs.shape:
        raise ValueError("probabilities and labels must have the same shape.")
    probs = np.clip(probs, 0.0, 1.0)
    labs = np.clip(labs, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    total = max(len(probs), 1)
    for i in range(int(n_bins)):
        left = edges[i]
        right = edges[i + 1]
        if i == int(n_bins) - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labs[mask]))
        ece += float(np.sum(mask)) / total * abs(conf - acc)
    return float(ece)


def sequential_temperature_scaling(
    confidence: np.ndarray,
    prefix_labels: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    n_bins: int = 15,
) -> CalibrationResult:
    """Sequential Temperature Scaling for DSpark confidence heads.

    ``confidence[:, k]`` is the conditional score c_k. ``prefix_labels[:, k]``
    should be the empirical prefix-survival label for position k. The procedure
    calibrates cumulative products from left to right while keeping previous
    positions fixed, matching the DSpark paper's STS recipe.
    """

    conf = np.asarray(confidence, dtype=np.float64)
    labels = np.asarray(prefix_labels, dtype=np.float64)
    if conf.shape != labels.shape or conf.ndim != 2:
        raise ValueError("confidence and prefix_labels must share shape [N, block_size].")
    if grid is None:
        grid = np.linspace(0.3, 3.0, 55)
    temps = np.ones(conf.shape[1], dtype=np.float64)
    calibrated_conditional = conf.copy()
    before = np.zeros(conf.shape[1], dtype=np.float64)
    after = np.zeros(conf.shape[1], dtype=np.float64)

    raw_prefix = np.cumprod(conf, axis=1)
    for k in range(conf.shape[1]):
        before[k] = expected_calibration_error(raw_prefix[:, k], labels[:, k], n_bins=n_bins)

    for k in range(conf.shape[1]):
        best_temp = 1.0
        best_ece = float("inf")
        for temp in grid:
            candidate = calibrated_conditional.copy()
            candidate[:, k] = apply_temperature(conf[:, k], float(temp))
            prefix = np.cumprod(candidate[:, : k + 1], axis=1)[:, -1]
            ece = expected_calibration_error(prefix, labels[:, k], n_bins=n_bins)
            if ece < best_ece:
                best_ece = ece
                best_temp = float(temp)
        temps[k] = best_temp
        calibrated_conditional[:, k] = apply_temperature(conf[:, k], best_temp)
        after[k] = best_ece

    return CalibrationResult(temperatures=temps, before_ece=before, after_ece=after)
