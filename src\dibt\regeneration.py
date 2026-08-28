from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamics import build_adjacency
from .phase0_r1 import update_decision_note_with_regeneration
from .r1_config import Phase0R1Config


CONDITIONS = (
    "INTACT_NO_DAMAGE",
    "DAMAGE_NO_REPAIR",
    "DAMAGE_WITH_REPAIR",
)


@dataclass(frozen=True)
class RegenerationTrajectory:
    next_states: np.ndarray
    intervention_index: np.ndarray
    coupling_factor: np.ndarray
    partition: np.ndarray


def _coupling_factor(
    condition: str,
    time_index: int,
    steps: int,
    damage_time: int,
    repair_time: int,
    damage_factor: float,
) -> float:
    if condition == "INTACT_NO_DAMAGE" or time_index < damage_time:
        return 1.0
    if condition == "DAMAGE_NO_REPAIR" or time_index < repair_time:
        return damage_factor
    progress = (time_index - repair_time + 1) / max(1, steps - repair_time)
    return float(damage_factor + (1.0 - damage_factor) * min(1.0, progress))


def simulate_regeneration(
    cfg: Phase0R1Config, seed: int, condition: str
) -> RegenerationTrajectory:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown regeneration condition: {condition}")
    sim_cfg = replace(
        cfg.simulation,
        steps=cfg.regeneration.steps,
        burn_in=cfg.regeneration.burn_in,
        train_intervention_amplitudes=cfg.ood_test.intervention_amplitudes,
    )
    adjacency, boundary_edges, partition = build_adjacency(sim_cfg, "intact")
    rng = np.random.default_rng(seed + 3_000_000)
    state = rng.normal(0.0, 0.15, size=len(partition))
    damage_time = int(cfg.regeneration.damage_fraction * cfg.regeneration.steps)
    repair_time = int(cfg.regeneration.repair_fraction * cfg.regeneration.steps)
    states = []
    intervention_indices = []
    factors = []

    for raw_time in range(cfg.regeneration.burn_in + cfg.regeneration.steps):
        retained_time = raw_time - cfg.regeneration.burn_in
        factor = (
            1.0
            if retained_time < 0
            else _coupling_factor(
                condition,
                retained_time,
                cfg.regeneration.steps,
                damage_time,
                repair_time,
                cfg.regeneration.damage_factor,
            )
        )
        transition_input = state.copy()
        intervention_index = -1
        if rng.random() < sim_cfg.intervention_probability:
            intervention_index = int(rng.integers(0, len(partition)))
            transition_input[intervention_index] = float(
                rng.choice(cfg.ood_test.intervention_amplitudes)
            )
        effective_adjacency = adjacency.copy()
        effective_adjacency[boundary_edges] = (
            adjacency[boundary_edges] * factor
        )
        linear_next = effective_adjacency @ transition_input
        linear_next += rng.normal(0.0, sim_cfg.process_noise, size=len(partition))
        state = np.tanh(linear_next)
        if retained_time >= 0:
            states.append(state.copy())
            intervention_indices.append(intervention_index)
            factors.append(factor)
    return RegenerationTrajectory(
        next_states=np.asarray(states),
        intervention_index=np.asarray(intervention_indices),
        coupling_factor=np.asarray(factors),
        partition=partition,
    )


def run_regeneration(
    cfg: Phase0R1Config, *, output_dir: str | Path
) -> tuple[pd.DataFrame, dict]:
    output_path = Path(output_dir)
    freeze_path = output_path / "PRIMARY_VERDICT_FROZEN.json"
    if not freeze_path.exists():
        raise RuntimeError(
            "regeneration is prohibited until the primary verdict is frozen"
        )
    thresholds = pd.read_csv(output_path / "thresholds_by_seed.csv")
    rows = []
    damage_time = int(cfg.regeneration.damage_fraction * cfg.regeneration.steps)
    repair_time = int(cfg.regeneration.repair_fraction * cfg.regeneration.steps)

    for seed in range(cfg.experiment.seeds):
        trajectories = {
            condition: simulate_regeneration(cfg, seed, condition)
            for condition in CONDITIONS
        }
        intact = trajectories["INTACT_NO_DAMAGE"]
        boundary_indices = np.where(intact.partition == 1)[0]
        internal_mask = intact.partition == 2
        post_repair = np.arange(cfg.regeneration.steps) >= repair_time
        intact_response_mask = post_repair & np.isin(
            intact.intervention_index, boundary_indices
        )
        intact_response = float(
            np.mean(np.abs(intact.next_states[intact_response_mask][:, internal_mask]))
        ) if np.any(intact_response_mask) else 0.0
        viability_threshold = float(
            thresholds.loc[thresholds["seed"] == seed, "viability_threshold"].iloc[0]
        )

        for condition, trajectory in trajectories.items():
            response_mask = post_repair & np.isin(
                trajectory.intervention_index, boundary_indices
            )
            response = float(
                np.mean(
                    np.abs(trajectory.next_states[response_mask][:, internal_mask])
                )
            ) if np.any(response_mask) else 0.0
            recovery_candidates = np.where(
                (np.arange(cfg.regeneration.steps) >= repair_time)
                & (
                    trajectory.coupling_factor
                    >= cfg.regeneration.recovery_fraction
                )
            )[0]
            recovery_time = (
                int(recovery_candidates[0] - damage_time)
                if len(recovery_candidates)
                else np.nan
            )
            internal_norm = np.linalg.norm(
                trajectory.next_states[:, internal_mask], axis=1
            )
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "damage_time": damage_time,
                    "repair_time": repair_time,
                    "recovery_time_after_damage": recovery_time,
                    "boundary_edge_function_recovery": float(
                        trajectory.coupling_factor[-1]
                    ),
                    "ood_response": response,
                    "ood_response_recovery_ratio": (
                        response / intact_response if intact_response > 0 else 0.0
                    ),
                    "viability_threshold": viability_threshold,
                    "post_repair_viability": float(
                        np.mean(internal_norm[post_repair] <= viability_threshold)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    condition_summary = {}
    for condition in CONDITIONS:
        subset = frame[frame["condition"] == condition]
        recovery_values = subset["recovery_time_after_damage"].dropna()
        condition_summary[condition] = {
            "mean_recovery_time_after_damage": (
                float(recovery_values.mean()) if len(recovery_values) else None
            ),
            "mean_boundary_edge_function_recovery": float(
                subset["boundary_edge_function_recovery"].mean()
            ),
            "mean_ood_response_recovery_ratio": float(
                subset["ood_response_recovery_ratio"].mean()
            ),
            "mean_post_repair_viability": float(
                subset["post_repair_viability"].mean()
            ),
        }
    summary = {
        "track": "Phase 0-r1B",
        "status": "secondary_mechanistic",
        "executed_after_primary_freeze": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": condition_summary,
        "claim_firewall": (
            "Damage/repair dynamics are a synthetic mechanism diagnostic and "
            "do not establish regeneration of self, consciousness, or identity."
        ),
    }
    frame.to_csv(output_path / "regeneration_seed_results.csv", index=False)
    (output_path / "regeneration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_decision_note_with_regeneration(output_path, summary)
    return frame, summary
