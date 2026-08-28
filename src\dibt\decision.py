from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import bootstrap_mean_ci
from .r1_config import R1DecisionConfig


@dataclass(frozen=True)
class DecisionResult:
    verdict: str
    first_failed_gate: str | None
    failed_gates: tuple[str, ...]
    gates: dict[str, bool]
    statistics: dict[str, float | list[float] | int]
    warnings: tuple[str, ...]


FAIL_VERDICTS = {
    "G0_IMPLEMENTATION": "STOP_IMPLEMENTATION_FAILURE",
    "G1_OOD_EFFECT_SIZE": "STOP_OOD_EFFECT_TOO_SMALL",
    "G2_OOD_CONFIDENCE": "STOP_OOD_CI_FAIL",
    "G3_SEED_CONSISTENCY": "STOP_SEED_CONSISTENCY_FAIL",
    "G4_COMMON_DRIVER": "STOP_COMMON_DRIVER_BOUNDARY_FALSE_POSITIVE",
}


def decide_phase0_r1(
    delta_mcc_cmi: np.ndarray,
    delta_mcc_do: np.ndarray,
    common_driver_boundary_fpr: np.ndarray,
    dibt_boundary_edge_rates: np.ndarray,
    audits: dict[str, bool],
    cfg: R1DecisionConfig,
    bootstrap_repeats: int,
) -> DecisionResult:
    delta_cmi = np.asarray(delta_mcc_cmi, dtype=float)
    delta_do = np.asarray(delta_mcc_do, dtype=float)
    common_fpr = np.asarray(common_driver_boundary_fpr, dtype=float)
    mean_delta, ci_lower, ci_upper = bootstrap_mean_ci(
        delta_cmi, bootstrap_repeats, seed=2026
    )
    mean_delta_do, do_ci_lower, do_ci_upper = bootstrap_mean_ci(
        delta_do, bootstrap_repeats, seed=2027
    )
    positive_count = int(np.sum(delta_cmi > 0.0))
    positive_fraction = float(positive_count / len(delta_cmi))
    mean_common_fpr = float(np.mean(common_fpr))
    gates = {
        "G0_IMPLEMENTATION": bool(all(audits.values())),
        "G1_OOD_EFFECT_SIZE": mean_delta >= cfg.min_mean_delta_mcc_cmi,
        "G2_OOD_CONFIDENCE": ci_lower > cfg.min_bootstrap_ci_lower,
        "G3_SEED_CONSISTENCY": positive_fraction
        >= cfg.min_positive_seed_fraction,
        "G4_COMMON_DRIVER": mean_common_fpr
        <= cfg.max_common_driver_boundary_fpr,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    warnings = []
    if float(np.mean(dibt_boundary_edge_rates)) == 0.0:
        warnings.append(
            "DIBT predicted no intact OOD boundary edges; STOP_TRIVIAL_REJECTOR"
        )
    if warnings and not failed:
        verdict = "STOP_TRIVIAL_REJECTOR"
        first_failed = "TRIVIAL_REJECTOR"
    elif failed:
        first_failed = failed[0]
        verdict = FAIL_VERDICTS[first_failed]
    else:
        first_failed = None
        verdict = "PHASE0_R1_OOD_CONDITIONAL_SUPPORT"
    return DecisionResult(
        verdict=verdict,
        first_failed_gate=first_failed,
        failed_gates=failed,
        gates=gates,
        statistics={
            "mean_delta_mcc_cmi": mean_delta,
            "delta_mcc_cmi_bootstrap_95_ci": [ci_lower, ci_upper],
            "positive_seed_count": positive_count,
            "positive_seed_fraction": positive_fraction,
            "mean_delta_mcc_do": mean_delta_do,
            "delta_mcc_do_bootstrap_95_ci": [do_ci_lower, do_ci_upper],
            "mean_common_driver_boundary_fpr": mean_common_fpr,
        },
        warnings=tuple(warnings),
    )
