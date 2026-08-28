from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import EstimationConfig, SimulationConfig


@dataclass(frozen=True)
class SplitConfig:
    name: str
    intervention_amplitudes: tuple[float, ...]
    seed_offset: int


@dataclass(frozen=True)
class R1ExperimentConfig:
    seeds: int
    bootstrap_repeats: int
    output_dir: str


@dataclass(frozen=True)
class R1DecisionConfig:
    min_mean_delta_mcc_cmi: float
    min_bootstrap_ci_lower: float
    min_positive_seed_fraction: float
    max_common_driver_boundary_fpr: float


@dataclass(frozen=True)
class ViabilityCalibrationConfig:
    train_quantile: float


@dataclass(frozen=True)
class RegenerationConfig:
    steps: int
    burn_in: int
    damage_fraction: float
    repair_fraction: float
    recovery_fraction: float
    damage_factor: float


@dataclass(frozen=True)
class Phase0R1Config:
    simulation: SimulationConfig
    estimation: EstimationConfig
    train: SplitConfig
    validation: SplitConfig
    ood_test: SplitConfig
    experiment: R1ExperimentConfig
    decision: R1DecisionConfig
    viability: ViabilityCalibrationConfig
    regeneration: RegenerationConfig


def _float_tuple(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _split(name: str, data: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        name=name,
        intervention_amplitudes=_float_tuple(data["intervention_amplitudes"]),
        seed_offset=int(data["seed_offset"]),
    )


def _validate(cfg: Phase0R1Config) -> None:
    split_sets = [
        set(cfg.train.intervention_amplitudes),
        set(cfg.validation.intervention_amplitudes),
        set(cfg.ood_test.intervention_amplitudes),
    ]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("TRAIN, VALIDATION, and OOD_TEST amplitudes must be disjoint")
    offsets = {cfg.train.seed_offset, cfg.validation.seed_offset, cfg.ood_test.seed_offset}
    if len(offsets) != 3:
        raise ValueError("split seed offsets must be distinct")
    if cfg.experiment.seeds < 1 or cfg.experiment.bootstrap_repeats < 1:
        raise ValueError("seeds and bootstrap repeats must be positive")
    if not 0 <= cfg.decision.min_positive_seed_fraction <= 1:
        raise ValueError("positive-seed fraction must be in [0, 1]")
    if not 0 < cfg.viability.train_quantile < 1:
        raise ValueError("viability training quantile must be in (0, 1)")
    regen = cfg.regeneration
    if not 0 < regen.damage_fraction < regen.repair_fraction < 1:
        raise ValueError("damage and repair fractions must satisfy 0 < damage < repair < 1")
    if not 0 < regen.recovery_fraction <= 1:
        raise ValueError("recovery_fraction must be in (0, 1]")


def load_phase0_r1_config(path: str | Path) -> Phase0R1Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sim = data["simulation"]
    est = data["estimation"]
    splits = data["splits"]
    experiment = data["experiment"]
    decision = data["decision"]
    viability = data["viability"]
    regeneration = data["regeneration"]
    cfg = Phase0R1Config(
        simulation=SimulationConfig(
            steps=int(sim["steps"]),
            burn_in=int(sim["burn_in"]),
            n_external=int(sim["n_external"]),
            n_boundary=int(sim["n_boundary"]),
            n_internal=int(sim["n_internal"]),
            process_noise=float(sim["process_noise"]),
            common_driver_strength=float(sim["common_driver_strength"]),
            intervention_probability=float(sim["intervention_probability"]),
            train_intervention_amplitudes=_float_tuple(
                splits["train"]["intervention_amplitudes"]
            ),
            ood_intervention_amplitudes=_float_tuple(
                splits["ood_test"]["intervention_amplitudes"]
            ),
            viability_radius=1.0,
        ),
        estimation=EstimationConfig(
            ridge=float(est["ridge"]),
            permutation_repeats=int(est["permutation_repeats"]),
            permutation_quantile=float(est["permutation_quantile"]),
            min_intervention_samples=int(est["min_intervention_samples"]),
        ),
        train=_split("TRAIN", splits["train"]),
        validation=_split("VALIDATION", splits["validation"]),
        ood_test=_split("OOD_TEST", splits["ood_test"]),
        experiment=R1ExperimentConfig(
            seeds=int(experiment["seeds"]),
            bootstrap_repeats=int(experiment["bootstrap_repeats"]),
            output_dir=str(experiment["output_dir"]),
        ),
        decision=R1DecisionConfig(
            min_mean_delta_mcc_cmi=float(
                decision["min_mean_delta_mcc_cmi"]
            ),
            min_bootstrap_ci_lower=float(
                decision["min_bootstrap_ci_lower"]
            ),
            min_positive_seed_fraction=float(
                decision["min_positive_seed_fraction"]
            ),
            max_common_driver_boundary_fpr=float(
                decision["max_common_driver_boundary_fpr"]
            ),
        ),
        viability=ViabilityCalibrationConfig(
            train_quantile=float(viability["train_quantile"])
        ),
        regeneration=RegenerationConfig(
            steps=int(regeneration["steps"]),
            burn_in=int(regeneration["burn_in"]),
            damage_fraction=float(regeneration["damage_fraction"]),
            repair_fraction=float(regeneration["repair_fraction"]),
            recovery_fraction=float(regeneration["recovery_fraction"]),
            damage_factor=float(regeneration["damage_factor"]),
        ),
    )
    _validate(cfg)
    return cfg
