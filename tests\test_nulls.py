import numpy as np
import pytest

from dibt.dynamics import build_adjacency
from conftest import simulation_config


def test_common_driver_has_no_true_boundary_edges():
    _, true_edges, _ = build_adjacency(simulation_config(), "common_driver")
    assert not np.any(true_edges)


def test_boundary_removal_zeroes_true_boundary_edges():
    intact, true_edges, _ = build_adjacency(simulation_config(), "intact")
    removed, _, _ = build_adjacency(simulation_config(), "boundary_removal")
    assert np.any(true_edges)
    assert np.allclose(removed[true_edges], 0.0)
    assert np.any(np.abs(intact[true_edges]) > 0.0)


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown condition"):
        build_adjacency(simulation_config(), "unknown")
