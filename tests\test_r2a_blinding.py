from dataclasses import replace
import inspect

import numpy as np

import dibt.blind_partition as blind_partition
from dibt.blind_partition import BlindInput, infer_integrated_partition
from dibt.phase0_r2a import _permute_blind_data, _permuted_trial
from dibt.r2a_config import load_phase0_r2a_config
from dibt.r2a_evaluation import coassignment


def _small_config():
    cfg = load_phase0_r2a_config("configs/phase0_r2a.yaml")
    return replace(
        cfg,
        simulation=replace(cfg.simulation, steps=400, burn_in=50),
        estimation=replace(
            cfg.estimation,
            permutation_repeats=12,
            permutation_quantile=0.95,
            min_intervention_samples=5,
        ),
    )


def test_r2a_blind_input_excludes_truth_and_counts():
    assert set(BlindInput.__dataclass_fields__) == {
        "transition_inputs",
        "next_states",
        "intervention_index",
        "intervention_value",
    }
    source = inspect.getsource(blind_partition)
    assert "true_edges" not in source
    assert "n_external" not in source
    assert "n_boundary" not in source
    assert "n_internal" not in source


def test_r2a_node_permutation_equivariance():
    cfg = _small_config()
    data, truth_roles, _, _ = _permuted_trial(cfg, 0)
    first = infer_integrated_partition(data, cfg.estimation, cfg.clustering)
    order = np.array([3, 6, 1, 7, 0, 5, 2, 4])
    permuted, inverse = _permute_blind_data(data, order)
    second = infer_integrated_partition(
        permuted, cfg.estimation, cfg.clustering
    )
    restored = second.cluster_labels[inverse]
    assert np.array_equal(
        coassignment(first.cluster_labels), coassignment(restored)
    )
    assert len(truth_roles) == len(first.cluster_labels)


def test_r2a_seed_reproducibility():
    cfg = _small_config()
    first = _permuted_trial(cfg, 3)
    second = _permuted_trial(cfg, 3)
    assert np.array_equal(first[0].transition_inputs, second[0].transition_inputs)
    assert np.array_equal(first[0].intervention_index, second[0].intervention_index)
    assert np.array_equal(first[1], second[1])
