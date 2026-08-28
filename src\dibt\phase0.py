from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dynamics import SimResult, simulate
from .estimators import estimate_baseline, estimate_dibt_reference
from .metrics import (
    bootstrap_mean_ci,
    edge_mcc,
    internal_variance,
    predicted_edge_rate,
    viability_fraction,
)


def _transition_alignment_valid(sim: SimResult, expected_steps: int) -> bool:
    lengths = {
        len(sim.states),
        len(sim.transition_inputs),
        len(sim.next_states),
        len(sim.intervention_index),
        len(sim.intervention_value),
    }
    return lengths == {expected_steps}


def _seed_reproducibility_valid(cfg: Config) -> bool:
    first = simulate(cfg.simulation, seed=0, condition="intact")
    second = simulate(cfg.simulation, seed=0, condition="intact")
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


def run_phase0(cfg: Config) -> tuple[pd.DataFrame, dict]:
    rows = []
    alignment_checks = []

    for seed in range(cfg.experiment.seeds):
        intact = simulate(cfg.simulation, seed=seed, condition="intact")
        common = simulate(cfg.simulation, seed=seed, condition="common_driver")
        removed = simulate(
            cfg.simulation, seed=seed, condition="boundary_removal"
        )
        alignment_checks.extend(
            _transition_alignment_valid(sim, cfg.simulation.steps)
            for sim in (intact, common, removed)
        )

        baseline = estimate_baseline(intact, cfg.estimation, seed)
        dibt = estimate_dibt_reference(intact, cfg.estimation, seed)
        baseline_mcc = edge_mcc(intact.true_edges, baseline.edges)
        dibt_mcc = edge_mcc(intact.true_edges, dibt.edges)

        common_baseline = estimate_baseline(
            common, cfg.estimation, seed + 10_000
        )
        common_dibt = estimate_dibt_reference(
            common, cfg.estimation, seed + 10_000
        )

        rows.append(
            {
                "seed": seed,
                "baseline_mcc": baseline_mcc,
                "dibt_mcc": dibt_mcc,
                "delta_mcc": dibt_mcc - baseline_mcc,
                "baseline_edge_rate_intact": predicted_edge_rate(
                    baseline.edges
                ),
                "dibt_edge_rate_intact": predicted_edge_rate(dibt.edges),
                "baseline_edge_rate_common_driver": predicted_edge_rate(
                    common_baseline.edges
                ),
                "dibt_edge_rate_common_driver": predicted_edge_rate(
                    common_dibt.edges
                ),
                "viability_intact": viability_fraction(
                    intact.next_states,
                    intact.partition,
                    cfg.simulation.viability_radius,
                ),
                "viability_boundary_removed": viability_fraction(
                    removed.next_states,
                    removed.partition,
                    cfg.simulation.viability_radius,
                ),
                "internal_variance_intact": internal_variance(
                    intact.next_states, intact.partition
                ),
                "internal_variance_boundary_removed": internal_variance(
                    removed.next_states, removed.partition
                ),
            }
        )

    frame = pd.DataFrame(rows)
    mean_delta, ci_lower, ci_upper = bootstrap_mean_ci(
        frame["delta_mcc"].to_numpy(),
        repeats=cfg.experiment.bootstrap_repeats,
    )
    common_rate = float(frame["dibt_edge_rate_common_driver"].mean())
    numeric_values = frame.select_dtypes(include=[np.number]).to_numpy()
    audits = {
        "numerical_finiteness": bool(np.all(np.isfinite(numeric_values))),
        "transition_alignment": bool(all(alignment_checks)),
        "seed_reproducibility": _seed_reproducibility_valid(cfg),
    }
    # Every internal coordinate is bounded to [-1, 1] by tanh, so a radius at
    # least sqrt(n_internal) makes the viability fraction identically one.
    viability_radius_informative = (
        cfg.simulation.viability_radius < np.sqrt(cfg.simulation.n_internal)
    )
    diagnostic_warnings = []
    if not viability_radius_informative:
        diagnostic_warnings.append(
            "viability_radius is at least sqrt(n_internal); under tanh-bounded "
            "states, viability_fraction is structurally 1.0 and cannot "
            "discriminate intact from boundary-removal conditions"
        )
    gates = {
        "delta_mcc_effect": mean_delta >= cfg.decision.min_mean_delta_mcc,
        "delta_mcc_ci": ci_lower > cfg.decision.min_bootstrap_ci_lower,
        "common_driver": common_rate
        <= cfg.decision.max_common_driver_edge_rate,
    }

    if all(gates.values()) and all(audits.values()):
        verdict = "PHASE0_CONDITIONAL_SUPPORT"
    elif not gates["common_driver"]:
        verdict = "STOP_COMMON_DRIVER_FALSE_POSITIVE"
    elif not gates["delta_mcc_effect"] or not gates["delta_mcc_ci"]:
        verdict = "STOP_NO_PRIMARY_ADVANTAGE"
    else:
        verdict = "STOP_AUDIT_FAILURE"

    decision = {
        "mean_delta_mcc": mean_delta,
        "bootstrap_95_ci": [ci_lower, ci_upper],
        "mean_dibt_mcc": float(frame["dibt_mcc"].mean()),
        "mean_baseline_mcc": float(frame["baseline_mcc"].mean()),
        "mean_common_driver_dibt_edge_rate": common_rate,
        "mean_viability_intact": float(frame["viability_intact"].mean()),
        "mean_viability_boundary_removed": float(
            frame["viability_boundary_removed"].mean()
        ),
        "mean_internal_variance_intact": float(
            frame["internal_variance_intact"].mean()
        ),
        "mean_internal_variance_boundary_removed": float(
            frame["internal_variance_boundary_removed"].mean()
        ),
        "gates": gates,
        "audits": audits,
        "diagnostics": {
            "viability_radius_informative": bool(
                viability_radius_informative
            ),
            "warnings": diagnostic_warnings,
        },
        "verdict": verdict,
        "claim_firewall": (
            "Passing Phase 0 supports synthetic boundary-edge recovery only. "
            "It does not establish consciousness, selfhood, subjectivity, "
            "soul, or metaphysical identity."
        ),
    }

    output_directory = Path(cfg.experiment.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_directory / "phase0_seed_results.csv", index=False)
    (output_directory / "phase0_summary.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame, decision
