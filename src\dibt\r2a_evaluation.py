from __future__ import annotations

from itertools import product
import math

import numpy as np


def _choose2(value: int) -> float:
    return value * (value - 1) / 2.0


def adjusted_rand_index(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    truth_labels = np.unique(truth)
    predicted_labels = np.unique(prediction)
    contingency = np.asarray(
        [
            [np.sum((truth == left) & (prediction == right)) for right in predicted_labels]
            for left in truth_labels
        ],
        dtype=int,
    )
    sum_cells = float(sum(_choose2(int(value)) for value in contingency.ravel()))
    sum_rows = float(sum(_choose2(int(value)) for value in contingency.sum(axis=1)))
    sum_columns = float(
        sum(_choose2(int(value)) for value in contingency.sum(axis=0))
    )
    total_pairs = _choose2(len(truth))
    if total_pairs == 0:
        return 1.0
    expected = sum_rows * sum_columns / total_pairs
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0 if np.array_equal(truth, prediction) else 0.0
    return float((sum_cells - expected) / denominator)


def _macro_f1(truth: np.ndarray, mapped: np.ndarray) -> float:
    values = []
    for role in (0, 1, 2):
        tp = int(np.sum((truth == role) & (mapped == role)))
        fp = int(np.sum((truth != role) & (mapped == role)))
        fn = int(np.sum((truth == role) & (mapped != role)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def matched_macro_f1(truth_roles: np.ndarray, cluster_labels: np.ndarray) -> float:
    truth_roles = np.asarray(truth_roles, dtype=int)
    clusters = np.unique(cluster_labels)
    best_correct = -1
    best_f1 = 0.0
    # Each inferred cluster maps to at most one true role; -1 is unmatched.
    for assignment in product((-1, 0, 1, 2), repeat=len(clusters)):
        used = [role for role in assignment if role >= 0]
        if len(set(used)) != len(used):
            continue
        mapping = dict(zip(clusters, assignment))
        mapped = np.asarray([mapping[label] for label in cluster_labels])
        correct = int(np.sum(mapped == truth_roles))
        f1 = _macro_f1(truth_roles, mapped)
        if correct > best_correct or (correct == best_correct and f1 > best_f1):
            best_correct = correct
            best_f1 = f1
    return best_f1


def binary_mcc(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    truth_values = np.asarray(truth, dtype=bool)[mask]
    prediction_values = np.asarray(prediction, dtype=bool)[mask]
    tp = int(np.sum(truth_values & prediction_values))
    tn = int(np.sum(~truth_values & ~prediction_values))
    fp = int(np.sum(~truth_values & prediction_values))
    fn = int(np.sum(truth_values & ~prediction_values))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else float((tp * tn - fp * fn) / denominator)


def true_boundary_relations(role_labels: np.ndarray) -> np.ndarray:
    boundary = np.asarray(role_labels) == 1
    relations = boundary[:, None] ^ boundary[None, :]
    np.fill_diagonal(relations, False)
    return relations


def coassignment(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


def largest_cluster_fraction(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / len(labels))
