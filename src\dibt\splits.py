from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import SimulationConfig
from .dynamics import SimResult, simulate
from .estimators import EstimatorInput, make_estimator_input
from .r1_config import SplitConfig


def generate_split(
    simulation_cfg: SimulationConfig,
    split: SplitConfig,
    seed: int,
    *,
    condition: str,
) -> SimResult:
    split_cfg = replace(
        simulation_cfg,
        train_intervention_amplitudes=split.intervention_amplitudes,
    )
    return simulate(
        split_cfg,
        seed=seed + split.seed_offset,
        condition=condition,
        use_ood_interventions=False,
    )


def blind_for_estimation(sim: SimResult, split_name: str) -> EstimatorInput:
    return make_estimator_input(
        sim.transition_inputs,
        sim.next_states,
        sim.intervention_index,
        sim.intervention_value,
        split_name=split_name,
    )


def split_amplitudes_are_disjoint(*splits: SplitConfig) -> bool:
    amplitude_sets = [set(split.intervention_amplitudes) for split in splits]
    return not any(
        amplitude_sets[left] & amplitude_sets[right]
        for left in range(len(amplitude_sets))
        for right in range(left + 1, len(amplitude_sets))
    )


def split_rng_streams_are_distinct(*splits: SplitConfig) -> bool:
    return len({split.seed_offset for split in splits}) == len(splits)


def transition_alignment_valid(sim: SimResult, expected_steps: int) -> bool:
    lengths = {
        len(sim.states),
        len(sim.transition_inputs),
        len(sim.next_states),
        len(sim.intervention_index),
        len(sim.intervention_value),
    }
    return lengths == {expected_steps}


def simulations_equal(first: SimResult, second: SimResult) -> bool:
    return all(
        np.array_equal(left, right)
        for left, right in (
            (first.states, second.states),
            (first.transition_inputs, second.transition_inputs),
            (first.next_states, second.next_states),
            (first.intervention_index, second.intervention_index),
            (first.intervention_value, second.intervention_value),
        )
    )
