import numpy as np

from dibt.dynamics import simulate
from dibt.estimators import (
    estimate_baseline,
    estimate_dibt_reference,
    make_estimator_input,
)
from conftest import estimation_config, simulation_config


def test_paired_estimators_share_observational_scores_and_threshold():
    sim = simulate(simulation_config(steps=220), seed=3)
    cfg = estimation_config()
    data = make_estimator_input(
        sim.transition_inputs,
        sim.next_states,
        sim.intervention_index,
        sim.intervention_value,
        split_name="TRAIN",
    )
    baseline = estimate_baseline(data, cfg, seed=9)
    dibt = estimate_dibt_reference(data, cfg, seed=9)
    assert np.array_equal(baseline.cmi_scores, dibt.cmi_scores)
    assert baseline.cmi_threshold == dibt.cmi_threshold


def test_estimator_outputs_are_finite_and_have_no_self_edges():
    sim = simulate(simulation_config(steps=220), seed=4)
    data = make_estimator_input(
        sim.transition_inputs,
        sim.next_states,
        sim.intervention_index,
        sim.intervention_value,
        split_name="TRAIN",
    )
    result = estimate_dibt_reference(data, estimation_config(), seed=4)
    assert np.all(np.isfinite(result.cmi_scores))
    assert result.do_scores is not None
    assert np.all(np.isfinite(result.do_scores))
    assert not np.any(np.diag(result.edges))
