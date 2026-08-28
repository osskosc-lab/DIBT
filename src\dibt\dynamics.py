from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


VALID_CONDITIONS = frozenset({"intact", "common_driver", "boundary_removal"})


@dataclass(frozen=True)
class SimResult:
    states: np.ndarray
    transition_inputs: np.ndarray
    next_states: np.ndarray
    intervention_index: np.ndarray
    intervention_value: np.ndarray
    true_edges: np.ndarray
    causal_edges: np.ndarray
    partition: np.ndarray
    adjacency: np.ndarray
    condition: str


def _stable_scale(a: np.ndarray, target_radius: float = 0.92) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if radius > target_radius:
        return a * (target_radius / radius)
    return a


def build_adjacency(
    cfg: SimulationConfig, condition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of "
            f"{sorted(VALID_CONDITIONS)}"
        )

    ne, nb, ni = cfg.n_external, cfg.n_boundary, cfg.n_internal
    n = ne + nb + ni
    external = np.arange(0, ne)
    boundary = np.arange(ne, ne + nb)
    internal = np.arange(ne + nb, n)
    adjacency = np.zeros((n, n), dtype=float)

    for index in external:
        adjacency[index, index] = 0.55
    for index in boundary:
        adjacency[index, index] = 0.45
    for index in internal:
        adjacency[index, index] = 0.72

    if ne >= 2:
        adjacency[external[1], external[0]] = 0.18
    if ni >= 2:
        adjacency[internal[1], internal[0]] = 0.14

    true_edges = np.zeros((n, n), dtype=bool)

    # Matrix orientation is [target, source].
    for index in range(min(nb, ne)):
        adjacency[boundary[index], external[index]] = 0.55
        true_edges[boundary[index], external[index]] = True

    for index in range(min(nb, ni)):
        adjacency[internal[index], boundary[index]] = 0.62
        true_edges[internal[index], boundary[index]] = True
        adjacency[boundary[index], internal[index]] = -0.28
        true_edges[boundary[index], internal[index]] = True

    if condition in {"common_driver", "boundary_removal"}:
        adjacency[true_edges] = 0.0

    adjacency = _stable_scale(adjacency)
    partition = np.concatenate(
        [
            np.zeros(ne, dtype=int),
            np.ones(nb, dtype=int),
            np.full(ni, 2, dtype=int),
        ]
    )

    if condition == "common_driver":
        true_edges = np.zeros_like(true_edges)

    return adjacency, true_edges, partition


def simulate(
    cfg: SimulationConfig,
    seed: int,
    condition: str = "intact",
    use_ood_interventions: bool = False,
) -> SimResult:
    rng = np.random.default_rng(seed)
    adjacency, true_edges, partition = build_adjacency(cfg, condition)
    n_variables = adjacency.shape[0]
    total_steps = cfg.steps + cfg.burn_in
    state = rng.normal(0.0, 0.15, size=n_variables)

    states: list[np.ndarray] = []
    transition_inputs: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    intervention_indices: list[int] = []
    intervention_values: list[float] = []

    amplitudes = (
        cfg.ood_intervention_amplitudes
        if use_ood_interventions
        else cfg.train_intervention_amplitudes
    )
    common_driver = 0.0

    for step in range(total_steps):
        natural_state = state.copy()
        transition_input = natural_state.copy()
        intervention_index = -1
        intervention_value = 0.0

        if rng.random() < cfg.intervention_probability:
            intervention_index = int(rng.integers(0, n_variables))
            intervention_value = float(rng.choice(amplitudes))
            transition_input[intervention_index] = intervention_value

        linear_next = adjacency @ transition_input
        linear_next += rng.normal(0.0, cfg.process_noise, size=n_variables)

        if condition == "common_driver":
            common_driver = 0.75 * common_driver + rng.normal(0.0, 0.55)
            linear_next[partition == 0] += (
                cfg.common_driver_strength * common_driver
            )
            linear_next[partition == 2] += (
                cfg.common_driver_strength * common_driver
            )

        resulting_state = np.tanh(linear_next)

        if step >= cfg.burn_in:
            states.append(natural_state)
            transition_inputs.append(transition_input)
            next_states.append(resulting_state)
            intervention_indices.append(intervention_index)
            intervention_values.append(intervention_value)

        state = resulting_state

    return SimResult(
        states=np.asarray(states, dtype=float),
        transition_inputs=np.asarray(transition_inputs, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
        intervention_index=np.asarray(intervention_indices, dtype=int),
        intervention_value=np.asarray(intervention_values, dtype=float),
        true_edges=true_edges,
        causal_edges=(np.abs(adjacency) > 0.0)
        & ~np.eye(n_variables, dtype=bool),
        partition=partition,
        adjacency=adjacency,
        condition=condition,
    )
