from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd

from . import blind_partition as blind_module
from .blind_partition import (
    BlindEstimate,
    BlindInput,
    estimator_input_for_edges,
    infer_correlation_baseline,
    infer_do_profile_baseline,
    infer_integrated_partition,
    infer_random_baseline,
)
from .dynamics import SimResult, simulate
from .estimators import (
    apply_dibt_reference,
    apply_do_only,
    fit_thresholds,
    score_dataset,
)
from .metrics import bootstrap_mean_ci
from .r2a_config import Phase0R2AConfig
from .r2a_evaluation import (
    adjusted_rand_index,
    binary_mcc,
    coassignment,
    largest_cluster_fraction,
    matched_macro_f1,
    true_boundary_relations,
)


def _permuted_trial(
    cfg: Phase0R2AConfig, seed: int
) -> tuple[BlindInput, np.ndarray, np.ndarray, np.ndarray]:
    sim = simulate(
        cfg.simulation,
        seed=seed + cfg.trajectory_seed_offset,
        condition="intact",
    )
    rng = np.random.default_rng(seed + cfg.node_permutation_seed_offset)
    order = rng.permutation(sim.transition_inputs.shape[1])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    intervention_index = sim.intervention_index.copy()
    intervened = intervention_index >= 0
    intervention_index[intervened] = inverse[intervention_index[intervened]]
    data = BlindInput(
        transition_inputs=sim.transition_inputs[:, order],
        next_states=sim.next_states[:, order],
        intervention_index=intervention_index,
        intervention_value=sim.intervention_value.copy(),
    )
    return (
        data,
        sim.partition[order],
        sim.causal_edges[np.ix_(order, order)],
        order,
    )


def _permute_blind_data(
    data: BlindInput, order: np.ndarray
) -> tuple[BlindInput, np.ndarray]:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    intervention_index = data.intervention_index.copy()
    intervened = intervention_index >= 0
    intervention_index[intervened] = inverse[intervention_index[intervened]]
    return (
        BlindInput(
            data.transition_inputs[:, order],
            data.next_states[:, order],
            intervention_index,
            data.intervention_value.copy(),
        ),
        inverse,
    )


def _evaluate(
    estimate: BlindEstimate,
    truth_roles: np.ndarray,
    true_relations: np.ndarray,
) -> dict[str, float | int]:
    mask = ~np.eye(len(truth_roles), dtype=bool)
    return {
        "ari": adjusted_rand_index(truth_roles, estimate.cluster_labels),
        "macro_f1": matched_macro_f1(truth_roles, estimate.cluster_labels),
        "boundary_mcc": binary_mcc(
            true_relations, estimate.boundary_relations, mask
        ),
        "selected_k": estimate.selected_k,
        "silhouette": estimate.silhouette,
        "largest_cluster_fraction": largest_cluster_fraction(
            estimate.cluster_labels
        ),
        "boundary_relation_rate": float(
            np.mean(estimate.boundary_relations[mask])
        ),
    }


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "PyYAML", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _test_report_passed(path: Path) -> bool:
    return path.exists() and "returncode: 0" in path.read_text(encoding="utf-8")


def _decision_note(summary: dict) -> str:
    gates = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["gates"].items()
    )
    audits = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["audits"].items()
    )
    return f"""# Phase 0-r2A Blind Partition Feasibility Decision

## Verdict

**{summary['verdict']}**

## Fixed development design

Ten F0 development trajectories used independent r2A RNG streams. Node order
was independently permuted before estimator input. The estimator received only
transition arrays and intervention metadata.

## Implementation audits

{audits}

## Partition results

- Mean candidate ARI: `{summary['mean_candidate_ari']:.6f}`
- Mean best-baseline ARI: `{summary['mean_best_baseline_ari']:.6f}`
- Mean paired ΔARI: `{summary['mean_delta_ari']:.6f}`
- Bootstrap 95% CI: `{summary['delta_ari_bootstrap_95_ci']}`
- Positive seeds: `{summary['positive_seed_count']}/{summary['n_seeds']}`

## End-to-end all-pair boundary relation

- Candidate mean MCC: `{summary['mean_candidate_boundary_mcc']:.6f}`
- Best baseline mean MCC: `{summary['best_baseline_mean_boundary_mcc']:.6f}`

## Gates

{gates}

## Continuation decision

`r2b_authorized = {summary['r2b_authorized']}`. If false, the protocol stops
without searching additional integrated estimators.

## Claim firewall

Even a PASS would support only blind recovery of synthetic partitions and
boundary relations in the fixed F0 development setting. It would not identify
a real individual, self, consciousness, subjectivity, or integration value.
"""


def run_phase0_r2a(
    cfg: Phase0R2AConfig,
    *,
    config_path: str | Path,
    preregistration_path: str | Path,
    implementation_commit_sha: str,
    preregistration_commit_sha: str,
) -> tuple[pd.DataFrame, dict]:
    output_dir = Path(cfg.experiment.output_dir)
    freeze_path = output_dir / "R2A_VERDICT_FROZEN.json"
    if freeze_path.exists():
        raise RuntimeError("r2A verdict is already frozen; refusing to rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    permutation_checks = []
    reproducibility_checks = []

    for seed in range(cfg.experiment.seeds):
        data, truth_roles, causal_truth, _ = _permuted_trial(cfg, seed)
        repeated_data, repeated_roles, repeated_causal, _ = _permuted_trial(
            cfg, seed
        )
        reproducibility_checks.append(
            np.array_equal(data.transition_inputs, repeated_data.transition_inputs)
            and np.array_equal(data.next_states, repeated_data.next_states)
            and np.array_equal(truth_roles, repeated_roles)
            and np.array_equal(causal_truth, repeated_causal)
        )
        estimates = {
            "candidate": infer_integrated_partition(
                data, cfg.estimation, cfg.clustering
            ),
            "random": infer_random_baseline(
                data,
                cfg.estimation,
                cfg.clustering,
                seed + cfg.random_baseline_seed_offset,
            ),
            "correlation": infer_correlation_baseline(
                data, cfg.estimation, cfg.clustering
            ),
            "do_profile": infer_do_profile_baseline(
                data, cfg.estimation, cfg.clustering
            ),
        }
        true_relations = true_boundary_relations(truth_roles)
        row: dict[str, float | int] = {"seed": seed}
        for name, estimate in estimates.items():
            for metric, value in _evaluate(
                estimate, truth_roles, true_relations
            ).items():
                row[f"{name}_{metric}"] = value

        baseline_aris = [
            float(row[f"{name}_ari"])
            for name in ("random", "correlation", "do_profile")
        ]
        row["best_baseline_ari"] = max(baseline_aris)
        row["delta_ari_vs_best"] = (
            float(row["candidate_ari"]) - float(row["best_baseline_ari"])
        )

        edge_input = estimator_input_for_edges(data)
        thresholds = fit_thresholds(edge_input, cfg.estimation, seed)
        scores = score_dataset(edge_input, cfg.estimation)
        dibt_edges = apply_dibt_reference(scores, thresholds).edges
        do_edges = apply_do_only(scores, thresholds).edges
        mask = ~np.eye(len(truth_roles), dtype=bool)
        row["candidate_graph_mcc"] = binary_mcc(
            causal_truth, dibt_edges, mask
        )
        row["candidate_predicted_edge_rate"] = float(
            np.mean(dibt_edges[mask])
        )
        row["oracle_partition_do_graph_mcc"] = binary_mcc(
            causal_truth, do_edges, mask
        )

        extra_order = np.random.default_rng(seed + 7_000_000).permutation(
            len(truth_roles)
        )
        permuted_data, inverse = _permute_blind_data(data, extra_order)
        permuted_estimate = infer_integrated_partition(
            permuted_data, cfg.estimation, cfg.clustering
        )
        restored_labels = permuted_estimate.cluster_labels[inverse]
        permutation_checks.append(
            np.array_equal(
                coassignment(estimates["candidate"].cluster_labels),
                coassignment(restored_labels),
            )
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    numeric_finite = bool(
        np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()))
    )
    blind_fields = set(blind_module.BlindInput.__dataclass_fields__)
    blind_source = inspect.getsource(blind_module)
    audits = {
        "pytest": _test_report_passed(output_dir / "test_report.txt"),
        "blind_input_contract": blind_fields
        == {
            "transition_inputs",
            "next_states",
            "intervention_index",
            "intervention_value",
        }
        and "true_edges" not in blind_source,
        "forbidden_count_blinding": all(
            token not in blind_source
            for token in ("n_external", "n_boundary", "n_internal")
        ),
        "node_permutation_equivariance": bool(all(permutation_checks)),
        "seed_reproducibility": bool(all(reproducibility_checks)),
        "rng_isolation": cfg.trajectory_seed_offset >= 4_000_000
        and cfg.node_permutation_seed_offset >= 5_000_000,
        "numerical_finiteness": numeric_finite,
    }

    delta = frame["delta_ari_vs_best"].to_numpy()
    mean_delta, ci_lower, ci_upper = bootstrap_mean_ci(
        delta, cfg.experiment.bootstrap_repeats, seed=2028
    )
    positive_count = int(np.sum(delta > 0))
    positive_fraction = float(positive_count / len(delta))
    mean_candidate_ari = float(frame["candidate_ari"].mean())
    mean_best_ari = float(frame["best_baseline_ari"].mean())
    mean_candidate_boundary = float(frame["candidate_boundary_mcc"].mean())
    baseline_boundary_means = {
        name: float(frame[f"{name}_boundary_mcc"].mean())
        for name in ("random", "correlation", "do_profile")
    }
    best_boundary_mean = max(baseline_boundary_means.values())
    gates = {
        "G0_IMPLEMENTATION": bool(all(audits.values())),
        "G1_MEAN_ARI_ADVANTAGE": mean_delta > cfg.decision.min_mean_delta_ari
        and mean_candidate_ari > cfg.decision.min_candidate_mean_ari,
        "G2_ARI_CONFIDENCE": ci_lower > cfg.decision.min_bootstrap_ci_lower,
        "G3_SEED_CONSISTENCY": positive_fraction
        >= cfg.decision.min_positive_seed_fraction,
        "G4_END_TO_END_BOUNDARY": mean_candidate_boundary
        > best_boundary_mean,
    }
    passed = bool(all(gates.values()))
    verdict = (
        "PHASE0_R2A_BLIND_PARTITION_FEASIBLE"
        if passed
        else "STOP_PARTITION_NOT_IDENTIFIED"
    )
    summary = {
        "phase": "Phase 0-r2A",
        "verdict": verdict,
        "r2b_authorized": passed,
        "n_seeds": cfg.experiment.seeds,
        "mean_candidate_ari": mean_candidate_ari,
        "mean_best_baseline_ari": mean_best_ari,
        "mean_delta_ari": mean_delta,
        "delta_ari_bootstrap_95_ci": [ci_lower, ci_upper],
        "positive_seed_count": positive_count,
        "positive_seed_fraction": positive_fraction,
        "mean_candidate_boundary_mcc": mean_candidate_boundary,
        "baseline_mean_boundary_mcc": baseline_boundary_means,
        "best_baseline_mean_boundary_mcc": best_boundary_mean,
        "mean_candidate_macro_f1": float(frame["candidate_macro_f1"].mean()),
        "mean_candidate_selected_k": float(frame["candidate_selected_k"].mean()),
        "gates": gates,
        "audits": audits,
        "preregistration_commit_sha": preregistration_commit_sha,
        "implementation_commit_sha": implementation_commit_sha,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_firewall": (
            "This development gate concerns blind recovery of synthetic "
            "partitions only; it does not identify a real individual, self, "
            "consciousness, subjectivity, or integration value."
        ),
    }
    frame.to_csv(output_dir / "seed_results.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(config_path, output_dir / "config_snapshot.yaml")
    shutil.copyfile(
        preregistration_path, output_dir / "preregistration_snapshot.yaml"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "commit_sha.txt").write_text(
        f"preregistration={preregistration_commit_sha}\n"
        f"implementation={implementation_commit_sha}\n",
        encoding="utf-8",
    )
    (output_dir / "DECISION_NOTE.md").write_text(
        _decision_note(summary), encoding="utf-8"
    )
    freeze_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "r2b_authorized": passed,
                "summary_file": "summary.json",
                "frozen_at_utc": summary["completed_at_utc"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame, summary
