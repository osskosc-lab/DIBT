from pathlib import Path

from dibt.config import (
    Config,
    DecisionConfig,
    ExperimentConfig,
)
from dibt.phase0 import run_phase0
from conftest import estimation_config, simulation_config


def test_vacuous_viability_radius_is_reported(tmp_path: Path):
    cfg = Config(
        simulation=simulation_config(steps=100, viability_radius=2.5),
        estimation=estimation_config(permutation_repeats=4),
        experiment=ExperimentConfig(
            seeds=1,
            bootstrap_repeats=10,
            output_dir=str(tmp_path / "results"),
        ),
        decision=DecisionConfig(
            min_mean_delta_mcc=0.05,
            min_bootstrap_ci_lower=0.0,
            max_common_driver_edge_rate=0.10,
        ),
    )
    _, summary = run_phase0(cfg)
    diagnostics = summary["diagnostics"]
    assert diagnostics["viability_radius_informative"] is False
    assert diagnostics["warnings"]
