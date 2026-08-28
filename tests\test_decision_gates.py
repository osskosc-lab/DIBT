import numpy as np

from dibt.decision import decide_phase0_r1
from dibt.r1_config import R1DecisionConfig


def _config():
    return R1DecisionConfig(
        min_mean_delta_mcc_cmi=0.05,
        min_bootstrap_ci_lower=0.0,
        min_positive_seed_fraction=0.80,
        max_common_driver_boundary_fpr=0.10,
    )


def test_positive_seed_gate():
    delta = np.array([0.2] * 23 + [-0.01] * 7)
    result = decide_phase0_r1(
        delta,
        np.full(30, 0.1),
        np.zeros(30),
        np.full(30, 0.2),
        {"tests": True},
        _config(),
        bootstrap_repeats=200,
    )
    assert result.statistics["positive_seed_count"] == 23
    assert result.gates["G3_SEED_CONSISTENCY"] is False
    assert result.verdict != "PHASE0_R1_OOD_CONDITIONAL_SUPPORT"


def test_all_primary_gates_can_pass():
    result = decide_phase0_r1(
        np.full(30, 0.2),
        np.full(30, 0.05),
        np.zeros(30),
        np.full(30, 0.2),
        {"tests": True},
        _config(),
        bootstrap_repeats=200,
    )
    assert all(result.gates.values())
    assert result.verdict == "PHASE0_R1_OOD_CONDITIONAL_SUPPORT"
