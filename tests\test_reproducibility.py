import numpy as np

from dibt.dynamics import simulate
from conftest import simulation_config


def test_seed_reproducibility():
    first = simulate(simulation_config(), seed=7, condition="intact")
    second = simulate(simulation_config(), seed=7, condition="intact")
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.transition_inputs, second.transition_inputs)
    assert np.array_equal(first.next_states, second.next_states)
    assert np.array_equal(first.intervention_index, second.intervention_index)
    assert np.array_equal(first.intervention_value, second.intervention_value)
