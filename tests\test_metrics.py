import numpy as np
import pytest

from dibt.metrics import bootstrap_mean_ci, edge_mcc


def test_mcc_perfect():
    truth = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    assert edge_mcc(truth, truth) == 1.0


def test_mcc_inverted_is_negative():
    truth = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    prediction = ~truth
    np.fill_diagonal(prediction, False)
    assert edge_mcc(truth, prediction) < 0.0


def test_mcc_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        edge_mcc(np.zeros((2, 2)), np.zeros((3, 3)))


def test_bootstrap_is_reproducible():
    values = np.array([-0.1, 0.2, 0.4])
    assert bootstrap_mean_ci(values, 100) == bootstrap_mean_ci(values, 100)
