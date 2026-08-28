from __future__ import annotations

from dataclasses import asdict
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

from . import estimators as estimator_module
from .decision import DecisionResult, decide_phase0_r1
from .evaluation import (
    boundary_candidate_mask,
    boundary_specific_false_positive_rate,
    classification_metrics,
    off_diagonal_mask,
    within_system_true_edge_recall,
)
from .estimators import (
    FrozenThresholds,
    apply_cmi_only,
    apply_dibt_reference,
    apply_do_only,
    fit_thresholds,
    score_dataset,
    subset_by_intervention_amplitude,
)
from .r1_config import Phase0R1Config
from .splits import (
    blind_for_estimation,
    generate_split,
    simulations_equal,
    split_amplitudes_are_disjoint,
    split_rng_streams_are_distinct,
    transition_alignment_valid,
)


CLAIM_FIREWALL = (
    "Under the preregistered synthetic system and held-out intervention "
    "regime, a PASS supports only improved recovery of boundary-candidate "
    "causal edges and survival of the specified common-driver falsification. "
    "It does not establish a real individual, metaphysical self-boundary, "
    "consciousness, first-person privacy, or substrate-independent identity."
)


def _estimator_results(data, thresholds, estimation_cfg):
    scores = score_dataset(data, estimation_cfg)
    return {
        "cmi": apply_cmi_only(scores, thresholds),
        "do": apply_do_only(scores, thresholds),
        "dibt": apply_dibt_reference(scores, thresholds),
    }


def _metric_columns(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _test_report_passed(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "returncode: 0" in text


def _all_numeric_finite(*frames: pd.DataFrame) -> bool:
    for frame in frames:
        values = frame.select_dtypes(include=[np.number]).to_numpy()
        if not np.all(np.isfinite(values)):
            return False
    return True


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


def _decision_note(summary: dict, regeneration: dict | None = None) -> str:
    stats = summary["decision_statistics"]
    gates = summary["gates"]
    audit_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["audits"].items()
    )
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items()
    )
    regeneration_text = (
        "Not yet executed. The extension may run only after the primary verdict "
        "is frozen."
        if regeneration is None
        else json.dumps(regeneration, ensure_ascii=False, indent=2)
    )
    return f"""# DIBT Phase 0-r1 Decision Note

## 1. Executive Verdict

**{summary['verdict']}**

First failed gate: `{summary['first_failed_gate']}`  
All failed gates: `{summary['failed_gates']}`

## 2. Implementation Audit

{audit_lines}

## 3. OOD Experimental Design

Thresholds were fit separately for each seed on TRAIN INTACT trajectories and
then frozen. Independent RNG streams generated VALIDATION and OOD_TEST data.
No OOD_TEST trajectory was used to fit a threshold or tune a gate.

## 4. CMI-only Baseline

Mean boundary MCC: `{summary['mean_mcc_cmi']:.6f}`.

## 5. DO-only Baseline

Mean boundary MCC: `{summary['mean_mcc_do']:.6f}`.

## 6. DIBT CMI-and-DO Result

Mean boundary MCC: `{summary['mean_mcc_dibt']:.6f}`.

## 7. Delta MCC and Bootstrap CI

- Mean ΔMCC vs CMI-only: `{stats['mean_delta_mcc_cmi']:.6f}`
- Seed-bootstrap 95% CI: `{stats['delta_mcc_cmi_bootstrap_95_ci']}`
- Mean ΔMCC vs DO-only: `{stats['mean_delta_mcc_do']:.6f}`
- Secondary 95% CI: `{stats['delta_mcc_do_bootstrap_95_ci']}`

## 8. Positive-Seed Consistency

`{stats['positive_seed_count']} / {summary['n_seeds']}` seeds were positive
(`{stats['positive_seed_fraction']:.3f}`).

## 9. COMMON_DRIVER Boundary-Specific Falsification

Mean DIBT boundary-specific FPR:
`{stats['mean_common_driver_boundary_fpr']:.6f}`.

## 10. Causal-Graph Recovery vs Boundary Recovery

Primary MCC values use only the preregistered boundary-candidate mask. Full
observed causal-graph MCC values are retained separately in `seed_results.csv`.
The truth partition is used only by the evaluation layer.

## 11. Viability Diagnostic

The viability threshold is the fixed preregistered quantile of the TRAIN
INTACT internal-norm distribution for each seed. OOD intact and
boundary-removal values are reported as secondary exploratory diagnostics.

## 12. Regeneration Diagnostic

{regeneration_text}

## 13. Adversarial Interpretation

Gate table:

{gate_lines}

Warnings: `{summary['warnings']}`. Precision, recall, specificity, edge density,
DO-only comparison, and OOD-amplitude breakdowns are retained to expose sparse
or trivial-rejector behavior.

## 14. What Is Ruled Out

The common-driver test asks whether correlated external and internal states
without boundary-mediated causal edges induce boundary predictions above the
fixed tolerance. A PASS rules out that failure only in this generator family.

## 15. What Remains Descriptive

Viability and regeneration are secondary mechanism diagnostics. They are not
evidence of consciousness, subjectivity, selfhood, or identity persistence.

## 16. Narrowest Defensible Continuation

Continue only with additional falsification, blind boundary-partition inference,
and preregistered robustness checks. Do not widen the claim beyond the synthetic
OOD edge-recovery result.

## Claim Firewall

{CLAIM_FIREWALL}
"""


def run_phase0_r1(
    cfg: Phase0R1Config,
    *,
    config_path: str | Path,
    preregistration_path: str | Path,
    commit_sha: str,
    preregistration_sha: str,
    confirmatory: bool,
) -> tuple[pd.DataFrame, dict]:
    output_dir = Path(cfg.experiment.output_dir)
    freeze_path = output_dir / "PRIMARY_VERDICT_FROZEN.json"
    if confirmatory and freeze_path.exists():
        raise RuntimeError(
            "confirmatory result is already frozen; refusing to overwrite it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    seed_rows: list[dict] = []
    threshold_rows: list[dict] = []
    common_rows: list[dict] = []
    amplitude_rows: list[dict] = []
    all_alignment_checks: list[bool] = []
    frozen_threshold_checks: list[bool] = []

    for seed in range(cfg.experiment.seeds):
        train = generate_split(cfg.simulation, cfg.train, seed, condition="intact")
        validation = generate_split(
            cfg.simulation, cfg.validation, seed, condition="intact"
        )
        ood_intact = generate_split(
            cfg.simulation, cfg.ood_test, seed, condition="intact"
        )
        ood_common = generate_split(
            cfg.simulation, cfg.ood_test, seed, condition="common_driver"
        )
        ood_removed = generate_split(
            cfg.simulation, cfg.ood_test, seed, condition="boundary_removal"
        )
        all_alignment_checks.extend(
            transition_alignment_valid(sim, cfg.simulation.steps)
            for sim in (train, validation, ood_intact, ood_common, ood_removed)
        )

        train_data = blind_for_estimation(train, "TRAIN")
        validation_data = blind_for_estimation(validation, "VALIDATION")
        ood_data = blind_for_estimation(ood_intact, "OOD_TEST")
        common_data = blind_for_estimation(ood_common, "OOD_TEST")
        thresholds = fit_thresholds(train_data, cfg.estimation, seed)
        frozen_copy = FrozenThresholds(**asdict(thresholds))

        # VALIDATION is scored only to exercise the pre-freeze implementation;
        # no result from it changes a threshold, hyperparameter, or gate.
        _estimator_results(validation_data, thresholds, cfg.estimation)
        intact_results = _estimator_results(ood_data, thresholds, cfg.estimation)
        common_results = _estimator_results(
            common_data, thresholds, cfg.estimation
        )
        frozen_threshold_checks.append(thresholds == frozen_copy)

        boundary_mask = boundary_candidate_mask(ood_intact.partition)
        full_mask = off_diagonal_mask(len(ood_intact.partition))
        row = {"seed": seed}
        for estimator_name, result in intact_results.items():
            boundary_metrics = classification_metrics(
                ood_intact.true_edges, result.edges, boundary_mask
            )
            graph_metrics = classification_metrics(
                ood_intact.causal_edges, result.edges, full_mask
            )
            row.update(
                _metric_columns(
                    f"{estimator_name}_boundary", boundary_metrics
                )
            )
            row.update(
                _metric_columns(f"{estimator_name}_graph", graph_metrics)
            )
        row["delta_mcc_cmi"] = (
            row["dibt_boundary_mcc"] - row["cmi_boundary_mcc"]
        )
        row["delta_mcc_do"] = (
            row["dibt_boundary_mcc"] - row["do_boundary_mcc"]
        )

        internal_train_norm = np.linalg.norm(
            train.next_states[:, train.partition == 2], axis=1
        )
        viability_threshold = float(
            np.quantile(internal_train_norm, cfg.viability.train_quantile)
        )
        intact_norm = np.linalg.norm(
            ood_intact.next_states[:, ood_intact.partition == 2], axis=1
        )
        removed_norm = np.linalg.norm(
            ood_removed.next_states[:, ood_removed.partition == 2], axis=1
        )
        row["viability_threshold"] = viability_threshold
        row["viability_intact_ood"] = float(
            np.mean(intact_norm <= viability_threshold)
        )
        row["viability_boundary_removed_ood"] = float(
            np.mean(removed_norm <= viability_threshold)
        )
        seed_rows.append(row)
        threshold_rows.append(
            {
                "seed": seed,
                "cmi_threshold": thresholds.cmi,
                "do_threshold": thresholds.do,
                "threshold_source": thresholds.source_split,
                "threshold_fit_seed": thresholds.fit_seed,
                "n_train_transitions": thresholds.n_train_transitions,
                "viability_threshold": viability_threshold,
                "viability_threshold_source": "TRAIN_INTACT",
                "viability_train_quantile": cfg.viability.train_quantile,
            }
        )

        common_row = {"seed": seed}
        for estimator_name, result in common_results.items():
            common_row[f"{estimator_name}_boundary_fpr"] = (
                boundary_specific_false_positive_rate(
                    result.edges, ood_common.partition
                )
            )
            common_row[f"{estimator_name}_all_edge_rate"] = float(
                np.mean(result.edges[full_mask])
            )
            common_row[f"{estimator_name}_within_true_edge_recall"] = (
                within_system_true_edge_recall(
                    ood_common.causal_edges,
                    result.edges,
                    ood_common.partition,
                )
            )
        common_rows.append(common_row)

        for amplitude in cfg.ood_test.intervention_amplitudes:
            amplitude_data = subset_by_intervention_amplitude(ood_data, amplitude)
            amplitude_results = _estimator_results(
                amplitude_data, thresholds, cfg.estimation
            )
            amplitude_row = {
                "seed": seed,
                "amplitude": amplitude,
                "n_transitions": amplitude_data.n_transitions,
            }
            for estimator_name, result in amplitude_results.items():
                metrics = classification_metrics(
                    ood_intact.true_edges, result.edges, boundary_mask
                )
                amplitude_row.update(
                    _metric_columns(estimator_name, metrics)
                )
            amplitude_rows.append(amplitude_row)

    seed_frame = pd.DataFrame(seed_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    common_frame = pd.DataFrame(common_rows)
    amplitude_frame = pd.DataFrame(amplitude_rows)

    estimator_source = inspect.getsource(estimator_module)
    test_report = output_dir / "test_report.txt"
    first_train = generate_split(
        cfg.simulation, cfg.train, 0, condition="intact"
    )
    second_train = generate_split(
        cfg.simulation, cfg.train, 0, condition="intact"
    )
    audits = {
        "pytest": _test_report_passed(test_report),
        "numerical_finiteness": _all_numeric_finite(
            seed_frame, thresholds_frame, common_frame, amplitude_frame
        ),
        "transition_alignment": bool(all(all_alignment_checks)),
        "seed_reproducibility": simulations_equal(first_train, second_train),
        "truth_blinding": "true_edges" not in estimator_source
        and "partition" not in estimator_source,
        "train_test_isolation": split_amplitudes_are_disjoint(
            cfg.train, cfg.validation, cfg.ood_test
        )
        and split_rng_streams_are_distinct(
            cfg.train, cfg.validation, cfg.ood_test
        ),
        "frozen_thresholds": bool(all(frozen_threshold_checks))
        and bool((thresholds_frame["threshold_source"] == "TRAIN").all()),
    }
    decision: DecisionResult = decide_phase0_r1(
        seed_frame["delta_mcc_cmi"].to_numpy(),
        seed_frame["delta_mcc_do"].to_numpy(),
        common_frame["dibt_boundary_fpr"].to_numpy(),
        seed_frame["dibt_boundary_edge_rate"].to_numpy(),
        audits,
        cfg.decision,
        cfg.experiment.bootstrap_repeats,
    )
    summary = {
        "phase": "Phase 0-r1",
        "run_mode": "confirmatory" if confirmatory else "smoke",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit_sha": commit_sha,
        "preregistration_commit_sha": preregistration_sha,
        "n_seeds": cfg.experiment.seeds,
        "mean_mcc_cmi": float(seed_frame["cmi_boundary_mcc"].mean()),
        "mean_mcc_do": float(seed_frame["do_boundary_mcc"].mean()),
        "mean_mcc_dibt": float(seed_frame["dibt_boundary_mcc"].mean()),
        "decision_statistics": decision.statistics,
        "gates": decision.gates,
        "audits": audits,
        "verdict": decision.verdict,
        "first_failed_gate": decision.first_failed_gate,
        "failed_gates": list(decision.failed_gates),
        "warnings": list(decision.warnings),
        "viability": {
            "threshold_source": "TRAIN_INTACT",
            "train_quantile": cfg.viability.train_quantile,
            "mean_threshold": float(
                seed_frame["viability_threshold"].mean()
            ),
            "mean_viability_intact_ood": float(
                seed_frame["viability_intact_ood"].mean()
            ),
            "mean_viability_boundary_removed_ood": float(
                seed_frame["viability_boundary_removed_ood"].mean()
            ),
            "primary_verdict_use": False,
        },
        "claim_firewall": CLAIM_FIREWALL,
    }

    seed_frame.to_csv(output_dir / "seed_results.csv", index=False)
    thresholds_frame.to_csv(output_dir / "thresholds_by_seed.csv", index=False)
    common_frame.to_csv(
        output_dir / "common_driver_boundary_fpr.csv", index=False
    )
    amplitude_frame.to_csv(
        output_dir / "ood_amplitude_breakdown.csv", index=False
    )
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
        f"implementation={commit_sha}\npreregistration={preregistration_sha}\n",
        encoding="utf-8",
    )
    (output_dir / "DECISION_NOTE.md").write_text(
        _decision_note(summary), encoding="utf-8"
    )
    if confirmatory:
        freeze_path.write_text(
            json.dumps(
                {
                    "verdict": decision.verdict,
                    "implementation_commit_sha": commit_sha,
                    "preregistration_commit_sha": preregistration_sha,
                    "frozen_at_utc": summary["completed_at_utc"],
                    "summary_file": "summary.json",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return seed_frame, summary


def update_decision_note_with_regeneration(
    output_dir: str | Path, regeneration_summary: dict
) -> None:
    output_path = Path(output_dir)
    summary = json.loads(
        (output_path / "summary.json").read_text(encoding="utf-8")
    )
    (output_path / "DECISION_NOTE.md").write_text(
        _decision_note(summary, regeneration_summary), encoding="utf-8"
    )
