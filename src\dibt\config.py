from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    steps: int
    burn_in: int
    n_external: int
    n_boundary: int
    n_internal: int
    process_noise: float
    common_driver_strength: float
    intervention_probability: float
    train_intervention_amplitudes: tuple[float, ...]
    ood_intervention_amplitudes: tuple[float, ...]
    viability_radius: float


@dataclass(frozen=True)
class EstimationConfig:
    ridge: float
    permutation_repeats: int
    permutation_quantile: float
    min_intervention_samples: int


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: int
    bootstrap_repeats: int
    output_dir: str


@dataclass(frozen=True)
class DecisionConfig:
    min_mean_delta_mcc: float
    min_bootstrap_ci_lower: float
    max_common_driver_edge_rate: float


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig
    estimation: EstimationConfig
    experiment: ExperimentConfig
    decision: DecisionConfig


def _tuple_float(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _validate_config(cfg: Config) -> None:
    sim = cfg.simulation
    est = cfg.estimation
    exp = cfg.experiment
    decision = cfg.decision

    if sim.steps < 2 or sim.burn_in < 0:
        raise ValueError("steps must be >= 2 and burn_in must be >= 0")
    if min(sim.n_external, sim.n_boundary, sim.n_internal) < 1:
        raise ValueError("all state partitions must contain at least one variable")
    if sim.process_noise < 0 or sim.common_driver_strength < 0:
        raise ValueError("noise and common-driver strength must be non-negative")
    if not 0 <= sim.intervention_probability <= 1:
        raise ValueError("intervention_probability must be in [0, 1]")
    if not sim.train_intervention_amplitudes:
        raise ValueError("train_intervention_amplitudes cannot be empty")
    if not sim.ood_intervention_amplitudes:
        raise ValueError("ood_intervention_amplitudes cannot be empty")
    if sim.viability_radius <= 0:
        raise ValueError("viability_radius must be positive")
    if est.ridge < 0 or est.permutation_repeats < 1:
        raise ValueError("ridge must be non-negative and permutations positive")
    if not 0 <= est.permutation_quantile <= 1:
        raise ValueError("permutation_quantile must be in [0, 1]")
    if est.min_intervention_samples < 2:
        raise ValueError("min_intervention_samples must be >= 2")
    if exp.seeds < 1 or exp.bootstrap_repeats < 1:
        raise ValueError("seeds and bootstrap_repeats must be positive")
    if decision.max_common_driver_edge_rate < 0:
        raise ValueError("max_common_driver_edge_rate must be non-negative")


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    simulation = data["simulation"]
    estimation = data["estimation"]
    experiment = data["experiment"]
    decision = data["decision"]

    cfg = Config(
        simulation=SimulationConfig(
            steps=int(simulation["steps"]),
            burn_in=int(simulation["burn_in"]),
            n_external=int(simulation["n_external"]),
            n_boundary=int(simulation["n_boundary"]),
            n_internal=int(simulation["n_internal"]),
            process_noise=float(simulation["process_noise"]),
            common_driver_strength=float(simulation["common_driver_strength"]),
            intervention_probability=float(
                simulation["intervention_probability"]
            ),
            train_intervention_amplitudes=_tuple_float(
                simulation["train_intervention_amplitudes"]
            ),
            ood_intervention_amplitudes=_tuple_float(
                simulation["ood_intervention_amplitudes"]
            ),
            viability_radius=float(simulation["viability_radius"]),
        ),
        estimation=EstimationConfig(
            ridge=float(estimation["ridge"]),
            permutation_repeats=int(estimation["permutation_repeats"]),
            permutation_quantile=float(estimation["permutation_quantile"]),
            min_intervention_samples=int(
                estimation["min_intervention_samples"]
            ),
        ),
        experiment=ExperimentConfig(
            seeds=int(experiment["seeds"]),
            bootstrap_repeats=int(experiment["bootstrap_repeats"]),
            output_dir=str(experiment["output_dir"]),
        ),
        decision=DecisionConfig(
            min_mean_delta_mcc=float(decision["min_mean_delta_mcc"]),
            min_bootstrap_ci_lower=float(decision["min_bootstrap_ci_lower"]),
            max_common_driver_edge_rate=float(
                decision["max_common_driver_edge_rate"]
            ),
        ),
    )
    _validate_config(cfg)
    return cfg
