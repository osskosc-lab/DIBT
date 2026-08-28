from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EstimationConfig
from .dynamics import SimResult


def _ridge_residual(y: np.ndarray, controls: np.ndarray, ridge: float) -> np.ndarray:
    outcome = np.asarray(y, dtype=float)
    design = np.asarray(controls, dtype=float)
    if design.size == 0:
        return outcome - np.mean(outcome)

    augmented = np.column_stack([design, np.ones(len(design))])
    penalty = ridge * np.eye(augmented.shape[1])
    penalty[-1, -1] = 1e-10
    beta = np.linalg.solve(
        augmented.T @ augmented + penalty, augmented.T @ outcome
    )
    return outcome - augmented @ beta


def gaussian_cmi_score(
    transition_inputs: np.ndarray,
    next_states: np.ndarray,
    source: int,
    target: int,
    ridge: float,
) -> float:
    predictors = np.asarray(transition_inputs, dtype=float)
    outcomes = np.asarray(next_states, dtype=float)
    source_values = predictors[:, source]
    target_values = outcomes[:, target]
    controls = predictors[:, [i for i in range(predictors.shape[1]) if i != source]]
    source_residual = _ridge_residual(source_values, controls, ridge)
    target_residual = _ridge_residual(target_values, controls, ridge)

    if np.std(source_residual) < 1e-12 or np.std(target_residual) < 1e-12:
        return 0.0

    rho = float(np.corrcoef(source_residual, target_residual)[0, 1])
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(-0.5 * np.log(1.0 - rho * rho))


def _standardized_slope(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([x, np.ones(len(x))])
    slope = np.linalg.lstsq(design, y, rcond=None)[0][0]
    return float(abs(slope) / (np.std(y) + 1e-8))


def do_effect_score(
    sim: SimResult, source: int, target: int, min_samples: int
) -> float:
    mask = sim.intervention_index == source
    if int(np.sum(mask)) < min_samples:
        return 0.0
    return _standardized_slope(
        sim.intervention_value[mask], sim.next_states[mask, target]
    )


def _cmi_permutation_threshold(
    sim: SimResult, cfg: EstimationConfig, rng: np.random.Generator
) -> float:
    n_variables = sim.transition_inputs.shape[1]
    null_scores = []
    for _ in range(cfg.permutation_repeats):
        source = int(rng.integers(0, n_variables))
        target = int(rng.integers(0, n_variables - 1))
        if target >= source:
            target += 1
        permuted_inputs = sim.transition_inputs.copy()
        permuted_inputs[:, source] = rng.permutation(permuted_inputs[:, source])
        null_scores.append(
            gaussian_cmi_score(
                permuted_inputs,
                sim.next_states,
                source,
                target,
                cfg.ridge,
            )
        )
    return float(np.quantile(null_scores, cfg.permutation_quantile))


def _do_permutation_threshold(
    sim: SimResult, cfg: EstimationConfig, rng: np.random.Generator
) -> float:
    n_variables = sim.transition_inputs.shape[1]
    null_scores = []
    attempts = 0
    max_attempts = cfg.permutation_repeats * 20

    while len(null_scores) < cfg.permutation_repeats and attempts < max_attempts:
        attempts += 1
        source = int(rng.integers(0, n_variables))
        target = int(rng.integers(0, n_variables - 1))
        if target >= source:
            target += 1
        mask = sim.intervention_index == source
        if int(np.sum(mask)) < cfg.min_intervention_samples:
            continue
        exposure = sim.intervention_value[mask]
        outcome = sim.next_states[mask, target]
        null_scores.append(_standardized_slope(exposure, rng.permutation(outcome)))

    if not null_scores:
        return float("inf")
    return float(np.quantile(null_scores, cfg.permutation_quantile))


@dataclass(frozen=True)
class EstimationResult:
    edges: np.ndarray
    cmi_scores: np.ndarray
    do_scores: np.ndarray | None
    cmi_threshold: float
    do_threshold: float | None


def _all_cmi_scores(sim: SimResult, cfg: EstimationConfig) -> np.ndarray:
    n_variables = sim.transition_inputs.shape[1]
    scores = np.zeros((n_variables, n_variables), dtype=float)
    for source in range(n_variables):
        for target in range(n_variables):
            if source != target:
                scores[target, source] = gaussian_cmi_score(
                    sim.transition_inputs,
                    sim.next_states,
                    source,
                    target,
                    cfg.ridge,
                )
    return scores


def estimate_baseline(
    sim: SimResult, cfg: EstimationConfig, seed: int
) -> EstimationResult:
    cmi_scores = _all_cmi_scores(sim, cfg)
    cmi_rng = np.random.default_rng(seed + 100_000)
    threshold = _cmi_permutation_threshold(sim, cfg, cmi_rng)
    edges = cmi_scores > threshold
    np.fill_diagonal(edges, False)
    return EstimationResult(edges, cmi_scores, None, threshold, None)


def estimate_dibt_reference(
    sim: SimResult, cfg: EstimationConfig, seed: int
) -> EstimationResult:
    cmi_scores = _all_cmi_scores(sim, cfg)
    cmi_rng = np.random.default_rng(seed + 100_000)
    do_rng = np.random.default_rng(seed + 200_000)
    cmi_threshold = _cmi_permutation_threshold(sim, cfg, cmi_rng)
    do_threshold = _do_permutation_threshold(sim, cfg, do_rng)
    n_variables = sim.transition_inputs.shape[1]
    do_scores = np.zeros((n_variables, n_variables), dtype=float)

    for source in range(n_variables):
        for target in range(n_variables):
            if source != target:
                do_scores[target, source] = do_effect_score(
                    sim, source, target, cfg.min_intervention_samples
                )

    edges = (cmi_scores > cmi_threshold) & (do_scores > do_threshold)
    np.fill_diagonal(edges, False)
    return EstimationResult(
        edges, cmi_scores, do_scores, cmi_threshold, do_threshold
    )
