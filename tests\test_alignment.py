import numpy as np

from dibt.dynamics import simulate
from conftest import simulation_config


def test_transition_arrays_share_exact_alignment():
    cfg = simulation_config(steps=120, intervention_probability=1.0)
    result = simulate(cfg, seed=11)
    assert result.states.shape == result.transition_inputs.shape == (120, 8)
    assert result.next_states.shape == (120, 8)
    assert result.intervention_index.shape == (120,)
    assert result.intervention_value.shape == (120,)


def test_intervention_is_applied_only_to_labeled_input_coordinate():
    result = simulate(
        simulation_config(steps=80, intervention_probability=1.0), seed=2
    )
    changed = ~np.isclose(result.states, result.transition_inputs)
    assert np.all(np.sum(changed, axis=1) == 1)
    assert np.all(changed[np.arange(len(changed)), result.intervention_index])
    assert np.allclose(
        result.transition_inputs[
            np.arange(len(result.transition_inputs)), result.intervention_index
        ],
        result.intervention_value,
    )
