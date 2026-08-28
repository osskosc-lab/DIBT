import inspect

import numpy as np

import dibt.estimators as estimators
from dibt.evaluation import (
    boundary_candidate_mask,
    boundary_specific_false_positive_rate,
    within_system_true_edge_recall,
)


def test_boundary_mask_is_evaluation_only():
    source = inspect.getsource(estimators)
    assert "true_edges" not in source
    assert "partition" not in source
    partition = np.array([0, 0, 1, 2])
    mask = boundary_candidate_mask(partition)
    assert mask[2, 0] and mask[0, 2]
    assert mask[3, 2] and mask[2, 3]
    assert not mask[1, 0]


def test_common_driver_boundary_specific_fpr():
    partition = np.array([0, 0, 1, 2, 2])
    prediction = np.zeros((5, 5), dtype=bool)
    causal_truth = np.zeros((5, 5), dtype=bool)
    # A correctly recovered within-E edge must not count as a boundary FP.
    prediction[1, 0] = True
    causal_truth[1, 0] = True
    assert boundary_specific_false_positive_rate(prediction, partition) == 0.0
    assert within_system_true_edge_recall(
        causal_truth, prediction, partition
    ) == 1.0


def test_truth_blinding():
    fields = set(estimators.EstimatorInput.__dataclass_fields__)
    assert fields == {"observational", "interventional", "split_name"}
