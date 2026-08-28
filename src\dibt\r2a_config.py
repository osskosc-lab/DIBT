from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import EstimationConfig, SimulationConfig


@dataclass(frozen=True)
class ClusteringConfig:
    min_clusters: int
    max_clusters: int
    exact_tie_tolerance: float


@dataclass(frozen=True)
class R2AExperimentConfig:
    seeds: int
    bootstrap_repeats: int
    output_dir: str


@dataclass(frozen=True)
class R2ADecisionConfig:
    min_mean_delta_ari: float
    min_bootstrap_ci_lower: float
    min_positive_seed_fraction: float
    min_candidate_mean_ari: float
    require_boundary_mcc_above_best_baseline: bool


@dataclass(frozen=True)
class Phase0R2AConfig:
    simulation: SimulationConfig
    estimation: EstimationConfig
    clustering: ClusteringConfig
    experiment: R2AExperimentConfig
    decision: R2ADecisionConfig
    trajectory_seed_offset: int
    node_permutation_seed_offset: int
    random_baseline_seed_offset: int


def load_phase0_r2a_config(path: str | Path) -> Phase0R2AConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sim = data["simulation"]
    est = data["estimation"]
    clustering = data["clustering"]
    experiment = data["experiment"]
    decision = data["decision"]
    rng = data["rng"]
    amplitudes = tuple(float(value) for value in sim["intervention_amplitudes"])
    cfg = Phase0R2AConfig(
        simulation=SimulationConfig(
            steps=int(sim["steps"]),
            burn_in=int(sim["burn_in"]),
            n_external=int(sim["n_external"]),
            n_boundary=int(sim["n_boundary"]),
            n_internal=int(sim["n_internal"]),
            process_noise=float(sim["process_noise"]),
            common_driver_strength=float(sim["common_driver_strength"]),
            intervention_probability=float(sim["intervention_probability"]),
            train_intervention_amplitudes=amplitudes,
            ood_intervention_amplitudes=amplitudes,
            viability_radius=1.0,
        ),
        estimation=EstimationConfig(
            ridge=float(est["ridge"]),
            permutation_repeats=int(est["permutation_repeats"]),
            permutation_quantile=float(est["permutation_quantile"]),
            min_intervention_samples=int(est["min_intervention_samples"]),
        ),
        clustering=ClusteringConfig(
            min_clusters=int(clustering["min_clusters"]),
            max_clusters=int(clustering["max_clusters"]),
            exact_tie_tolerance=float(clustering["exact_tie_tolerance"]),
        ),
        experiment=R2AExperimentConfig(
            seeds=int(experiment["seeds"]),
            bootstrap_repeats=int(experiment["bootstrap_repeats"]),
            output_dir=str(experiment["output_dir"]),
        ),
        decision=R2ADecisionConfig(
            min_mean_delta_ari=float(decision["min_mean_delta_ari"]),
            min_bootstrap_ci_lower=float(
                decision["min_bootstrap_ci_lower"]
            ),
            min_positive_seed_fraction=float(
                decision["min_positive_seed_fraction"]
            ),
            min_candidate_mean_ari=float(
                decision["min_candidate_mean_ari"]
            ),
            require_boundary_mcc_above_best_baseline=bool(
                decision["require_boundary_mcc_above_best_baseline"]
            ),
        ),
        trajectory_seed_offset=int(rng["trajectory_seed_offset"]),
        node_permutation_seed_offset=int(rng["node_permutation_seed_offset"]),
        random_baseline_seed_offset=int(rng["random_baseline_seed_offset"]),
    )
    if not 2 <= cfg.clustering.min_clusters <= cfg.clustering.max_clusters:
        raise ValueError("invalid clustering range")
    offsets = {
        cfg.trajectory_seed_offset,
        cfg.node_permutation_seed_offset,
        cfg.random_baseline_seed_offset,
    }
    if len(offsets) != 3:
        raise ValueError("r2A RNG offsets must be distinct")
    if cfg.experiment.seeds != 10:
        raise ValueError("r2A is preregistered for exactly 10 seeds")
    return cfg
