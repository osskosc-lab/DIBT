import numpy as np

from dibt.r2a_evaluation import (
    adjusted_rand_index,
    binary_mcc,
    matched_macro_f1,
    true_boundary_relations,
)


def test_adjusted_rand_is_label_permutation_invariant():
    truth = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_index(truth, prediction) == 1.0


def test_macro_f1_uses_evaluation_only_label_matching():
    truth = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    prediction = np.array([4, 4, 4, 8, 8, 3, 3, 3])
    assert matched_macro_f1(truth, prediction) == 1.0


def test_boundary_relation_mcc_is_all_pair_end_to_end():
    roles = np.array([0, 0, 1, 2, 2])
    truth = true_boundary_relations(roles)
    mask = ~np.eye(len(roles), dtype=bool)
    assert binary_mcc(truth, truth, mask) == 1.0
    all_negative = np.zeros_like(truth)
    assert binary_mcc(truth, all_negative, mask) == 0.0
