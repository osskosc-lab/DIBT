from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EstimationConfig


@dataclass(frozen=True)
class ObservationalInput:
    """Truth-blinded inputs for CMI scoring."""

    transition_inputs: np.ndarray
    next_states: np.ndarray


@dataclass(frozen=True)
class InterventionalInput:
    """Truth-blinded inputs for direct-intervention scoring."""

    intervention_index: np.ndarray
    intervention_value: np.ndarray
    next_states: np.ndarray


@dataclass(frozen=True)
class EstimatorInput:
    """Complete estimator-facing dataset with no generator truth fields."""

    observational: ObservationalInput
    interventional: InterventionalInput
    split_name: str

    @property
    def n_variables(self) -> int:
        return int(self.observational.transition_inputs.shape[1])

    @property
    def n_transitions(self) -> int:
        return int(self.observational.transition_inputs.shape[0])


@dataclass(frozen=True)
class FrozenThresholds:
    cmi: float
    do: float
    source_split: str
    fit_seed: int
    n_train_transitions: int


@dataclass(frozen=True)
class ScoreBundle:
    cmi: np.ndarray
    do: np.ndarray


@dataclass(frozen=True)
class EstimationResult:
    edges: np.ndarray
    cmi_scores: np.ndarray | None
    do_scores: np.ndarray | None
    cmi_threshold: float | None
    do_threshold: float | None


def make_estimator_input(
    transition_inputs: np.ndarray,
    next_states: np.ndarray,
    intervention_index: np.ndarray,
    intervention_value: np.ndarray,
    *,
    split_name: str,
) -> EstimatorInput:
    transition_inputs = np.asarray(transition_inputs, dtype=float)
    next_states = np.asarray(next_states, dtype=float)
    intervention_index = np.asarray(intervention_index, dtype=int)
    intervention_value = np.asarray(intervention_value, dtype=float)
    lengths = {
        len(transition_inputs),
        len(next_states),
        len(intervention_index),
        len(intervention_value),
    }
    if len(lengths) != 1:
        raise ValueError("estimator inputs must be transition-aligned")
    if transition_inputs.shape != next_states.shape:
        raise ValueError("transition inputs and next states must share shape")
    return EstimatorInput(
        observational=ObservationalInput(transition_inputs, next_states),
        interventional=InterventionalInput(
            intervention_index, intervention_value, next_states
        ),
        split_name=split_name,
    )


def subset_by_intervention_amplitude(
    data: EstimatorInput, amplitude: float
) -> EstimatorInput:
    mask = np.isclose(data.interventional.intervention_value, amplitude)
    return make_estimator_input(
        data.observational.transition_inputs[mask],
        data.observational.next_states[mask],
        data.interventional.intervention_index[mask],
        data.interventional.intervention_value[mask],
        split_name=f"{data.split_name}:amplitude={amplitude:g}",
    )


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
    data: ObservationalInput,
    source: int,
    target: int,
    ridge: float,
) -> float:
    predictors = data.transition_inputs
    outcomes = data.next_states
    if len(predictors) < 3:
        return 0.0
    source_values = predictors[:, source]
    target_values = outcomes[:, target]
    controls = predictors[
        :, [index for index in range(predictors.shape[1]) if index != source]
    ]
    source_residual = _ridge_residual(source_values, controls, ridge)
    target_residual = _ridge_residual(target_values, controls, ridge)
    if np.std(source_residual) < 1e-12 or np.std(target_residual) < 1e-12:
        return 0.0
    rho = float(np.corrcoef(source_residual, target_residual)[0, 1])
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(-0.5 * np.log(1.0 - rho * rho))


def _standardized_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    design = np.column_stack([x, np.ones(len(x))])
    slope = np.linalg.lstsq(design, y, rcond=None)[0][0]
    return float(abs(slope) / (np.std(y) + 1e-8))


def do_effect_score(
    data: InterventionalInput,
    source: int,
    target: int,
    min_samples: int,
) -> float:
    mask = data.intervention_index == source
    if int(np.sum(mask)) < min_samples:
        return 0.0
    return _standardized_slope(
        data.intervention_value[mask], data.next_states[mask, target]
    )


def score_cmi(data: ObservationalInput, cfg: EstimationConfig) -> np.ndarray:
    n_variables = data.transition_inputs.shape[1]
    scores = np.zeros((n_variables, n_variables), dtype=float)
    for source in range(n_variables):
        for target in range(n_variables):
            if source != target:
                scores[target, source] = gaussian_cmi_score(
                    data, source, target, cfg.ridge
                )
    return scores


def score_do(data: InterventionalInput, cfg: EstimationConfig) -> np.ndarray:
    n_variables = data.next_states.shape[1]
    scores = np.zeros((n_variables, n_variables), dtype=float)
    for source in range(n_variables):
        for target in range(n_variables):
            if source != target:
                scores[target, source] = do_effect_score(
                    data, source, target, cfg.min_intervention_samples
                )
    return scores


def score_dataset(data: EstimatorInput, cfg: EstimationConfig) -> ScoreBundle:
    return ScoreBundle(
        cmi=score_cmi(data.observational, cfg),
        do=score_do(data.interventional, cfg),
    )


def _cmi_permutation_threshold(
    data: ObservationalInput,
    cfg: EstimationConfig,
    rng: np.random.Generator,
) -> float:
    n_variables = data.transition_inputs.shape[1]
    null_scores = []
    for _ in range(cfg.permutation_repeats):
        source = int(rng.integers(0, n_variables))
        target = int(rng.integers(0, n_variables - 1))
        if target >= source:
            target += 1
        permuted = data.transition_inputs.copy()
        permuted[:, source] = rng.permutation(permuted[:, source])
        null_scores.append(
            gaussian_cmi_score(
                ObservationalInput(permuted, data.next_states),
                source,
                target,
                cfg.ridge,
            )
        )
    return float(np.quantile(null_scores, cfg.permutation_quantile))


def _do_permutation_threshold(
    data: InterventionalInput,
    cfg: EstimationConfig,
    rng: np.random.Generator,
) -> float:
    n_variables = data.next_states.shape[1]
    null_scores = []
    attempts = 0
    max_attempts = cfg.permutation_repeats * 20
    while len(null_scores) < cfg.permutation_repeats and attempts < max_attempts:
        attempts += 1
        source = int(rng.integers(0, n_variables))
        target = int(rng.integers(0, n_variables - 1))
        if target >= source:
            target += 1
        mask = data.intervention_index == source
        if int(np.sum(mask)) < cfg.min_intervention_samples:
            continue
        null_scores.append(
            _standardized_slope(
                data.intervention_value[mask],
                rng.permutation(data.next_states[mask, target]),
            )
        )
    if not null_scores:
        return float("inf")
    return float(np.quantile(null_scores, cfg.permutation_quantile))


def fit_thresholds(
    train_data: EstimatorInput,
    cfg: EstimationConfig,
    seed: int,
) -> FrozenThresholds:
    if train_data.split_name.upper() != "TRAIN":
        raise ValueError("thresholds may only be fit on the TRAIN split")
    cmi_rng = np.random.default_rng(seed + 100_000)
    do_rng = np.random.default_rng(seed + 200_000)
    return FrozenThresholds(
        cmi=_cmi_permutation_threshold(train_data.observational, cfg, cmi_rng),
        do=_do_permutation_threshold(train_data.interventional, cfg, do_rng),
        source_split="TRAIN",
        fit_seed=seed,
        n_train_transitions=train_data.n_transitions,
    )


def _edge_matrix(scores: np.ndarray, threshold: float) -> np.ndarray:
    edges = scores > threshold
    np.fill_diagonal(edges, False)
    return edges


def apply_cmi_only(
    scores: ScoreBundle, thresholds: FrozenThresholds
) -> EstimationResult:
    return EstimationResult(
        edges=_edge_matrix(scores.cmi, thresholds.cmi),
        cmi_scores=scores.cmi,
        do_scores=None,
        cmi_threshold=thresholds.cmi,
        do_threshold=None,
    )


def apply_do_only(
    scores: ScoreBundle, thresholds: FrozenThresholds
) -> EstimationResult:
    return EstimationResult(
        edges=_edge_matrix(scores.do, thresholds.do),
        cmi_scores=None,
        do_scores=scores.do,
        cmi_threshold=None,
        do_threshold=thresholds.do,
    )


def apply_dibt_reference(
    scores: ScoreBundle, thresholds: FrozenThresholds
) -> EstimationResult:
    edges = (scores.cmi > thresholds.cmi) & (scores.do > thresholds.do)
    np.fill_diagonal(edges, False)
    return EstimationResult(
        edges=edges,
        cmi_scores=scores.cmi,
        do_scores=scores.do,
        cmi_threshold=thresholds.cmi,
        do_threshold=thresholds.do,
    )


def _fit_on_same_data(
    data: EstimatorInput, cfg: EstimationConfig, seed: int
) -> tuple[ScoreBundle, FrozenThresholds]:
    train_alias = EstimatorInput(data.observational, data.interventional, "TRAIN")
    return score_dataset(data, cfg), fit_thresholds(train_alias, cfg, seed)


def estimate_baseline(
    data: EstimatorInput, cfg: EstimationConfig, seed: int
) -> EstimationResult:
    scores, thresholds = _fit_on_same_data(data, cfg, seed)
    return apply_cmi_only(scores, thresholds)


def estimate_do_only(
    data: EstimatorInput, cfg: EstimationConfig, seed: int
) -> EstimationResult:
    scores, thresholds = _fit_on_same_data(data, cfg, seed)
    return apply_do_only(scores, thresholds)


def estimate_dibt_reference(
    data: EstimatorInput, cfg: EstimationConfig, seed: int
) -> EstimationResult:
    scores, thresholds = _fit_on_same_data(data, cfg, seed)
    return apply_dibt_reference(scores, thresholds)
