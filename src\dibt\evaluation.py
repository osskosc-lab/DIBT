from __future__ import annotations

import numpy as np


def boundary_candidate_mask(partition: np.ndarray) -> np.ndarray:
    partition = np.asarray(partition, dtype=int)
    target_group = partition[:, None]
    source_group = partition[None, :]
    boundary = 1
    external_or_internal = (source_group == 0) | (source_group == 2)
    target_external_or_internal = (target_group == 0) | (target_group == 2)
    mask = ((target_group == boundary) & external_or_internal) | (
        (source_group == boundary) & target_external_or_internal
    )
    np.fill_diagonal(mask, False)
    return mask


def off_diagonal_mask(n_variables: int) -> np.ndarray:
    return ~np.eye(n_variables, dtype=bool)


def _confusion(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[int, int, int, int]:
    truth_values = np.asarray(truth, dtype=bool)[mask]
    prediction_values = np.asarray(prediction, dtype=bool)[mask]
    tp = int(np.sum(truth_values & prediction_values))
    tn = int(np.sum(~truth_values & ~prediction_values))
    fp = int(np.sum(~truth_values & prediction_values))
    fn = int(np.sum(truth_values & ~prediction_values))
    return tp, tn, fp, fn


def masked_mcc(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    tp, tn, fp, fn = _confusion(truth, prediction, mask)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else float((tp * tn - fp * fn) / denominator)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    tp, tn, fp, fn = _confusion(truth, prediction, mask)
    return {
        "mcc": masked_mcc(truth, prediction, mask),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "edge_rate": float(np.mean(np.asarray(prediction, dtype=bool)[mask])),
    }


def boundary_specific_false_positive_rate(
    prediction: np.ndarray, partition: np.ndarray
) -> float:
    mask = boundary_candidate_mask(partition)
    return float(np.mean(np.asarray(prediction, dtype=bool)[mask]))


def within_system_true_edge_recall(
    causal_truth: np.ndarray,
    prediction: np.ndarray,
    partition: np.ndarray,
) -> float:
    within_mask = off_diagonal_mask(len(partition)) & ~boundary_candidate_mask(partition)
    true_within = np.asarray(causal_truth, dtype=bool) & within_mask
    denominator = int(np.sum(true_within))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.asarray(prediction, dtype=bool) & true_within) / denominator)
