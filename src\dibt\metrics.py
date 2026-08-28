from __future__ import annotations

import numpy as np


def _off_diagonal_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("edge arrays must be square matrices")
    return values[~np.eye(values.shape[0], dtype=bool)]


def edge_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError("truth and prediction arrays must have the same shape")

    truth = _off_diagonal_values(np.asarray(y_true, dtype=bool))
    prediction = _off_diagonal_values(np.asarray(y_pred, dtype=bool))
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Match the common MCC convention for a degenerate confusion matrix.
    if denominator == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def predicted_edge_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(_off_diagonal_values(np.asarray(y_pred, dtype=bool))))


def viability_fraction(
    states: np.ndarray, partition: np.ndarray, radius: float
) -> float:
    internal = np.asarray(states, dtype=float)[:, np.asarray(partition) == 2]
    if internal.shape[1] == 0:
        raise ValueError("partition contains no internal variables")
    return float(np.mean(np.linalg.norm(internal, axis=1) <= radius))


def internal_variance(states: np.ndarray, partition: np.ndarray) -> float:
    internal = np.asarray(states, dtype=float)[:, np.asarray(partition) == 2]
    if internal.shape[1] == 0:
        raise ValueError("partition contains no internal variables")
    return float(np.mean(np.var(internal, axis=0)))


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, seed: int = 2026
) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must all be finite")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        bootstrap_sample = rng.choice(samples, size=len(samples), replace=True)
        bootstrap_means[repeat] = np.mean(bootstrap_sample)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(np.mean(samples)), float(lower), float(upper)
