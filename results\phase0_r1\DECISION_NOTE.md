# DIBT Phase 0-r1 Decision Note

## 1. Executive Verdict

**PHASE0_R1_OOD_CONDITIONAL_SUPPORT**

First failed gate: `None`  
All failed gates: `[]`

## 2. Implementation Audit

- pytest: PASS
- numerical_finiteness: PASS
- transition_alignment: PASS
- seed_reproducibility: PASS
- truth_blinding: PASS
- train_test_isolation: PASS
- frozen_thresholds: PASS

## 3. OOD Experimental Design

Thresholds were fit separately for each seed on TRAIN INTACT trajectories and
then frozen. Independent RNG streams generated VALIDATION and OOD_TEST data.
No OOD_TEST trajectory was used to fit a threshold or tune a gate.

## 4. CMI-only Baseline

Mean boundary MCC: `0.786996`.

## 5. DO-only Baseline

Mean boundary MCC: `1.000000`.

## 6. DIBT CMI-and-DO Result

Mean boundary MCC: `1.000000`.

## 7. Delta MCC and Bootstrap CI

- Mean ΔMCC vs CMI-only: `0.213004`
- Seed-bootstrap 95% CI: `[0.1845317372546629, 0.24385770145422514]`
- Mean ΔMCC vs DO-only: `0.000000`
- Secondary 95% CI: `[0.0, 0.0]`

## 8. Positive-Seed Consistency

`30 / 30` seeds were positive
(`1.000`).

## 9. COMMON_DRIVER Boundary-Specific Falsification

Mean DIBT boundary-specific FPR:
`0.000000`.

## 10. Causal-Graph Recovery vs Boundary Recovery

Primary MCC values use only the preregistered boundary-candidate mask. Full
observed causal-graph MCC values are retained separately in `seed_results.csv`.
The truth partition is used only by the evaluation layer.

## 11. Viability Diagnostic

The viability threshold is the fixed preregistered quantile of the TRAIN
INTACT internal-norm distribution for each seed. OOD intact and
boundary-removal values are reported as secondary exploratory diagnostics.

## 12. Regeneration Diagnostic

{
  "track": "Phase 0-r1B",
  "status": "secondary_mechanistic",
  "executed_after_primary_freeze": true,
  "completed_at_utc": "2026-08-28T15:49:19.040652+00:00",
  "conditions": {
    "INTACT_NO_DAMAGE": {
      "mean_recovery_time_after_damage": 301.0,
      "mean_boundary_edge_function_recovery": 1.0,
      "mean_ood_response_recovery_ratio": 1.0,
      "mean_post_repair_viability": 0.7612222222222224
    },
    "DAMAGE_NO_REPAIR": {
      "mean_recovery_time_after_damage": null,
      "mean_boundary_edge_function_recovery": 0.0,
      "mean_ood_response_recovery_ratio": 0.34066532213808587,
      "mean_post_repair_viability": 0.9002222222222224
    },
    "DAMAGE_WITH_REPAIR": {
      "mean_recovery_time_after_damage": 570.0,
      "mean_boundary_edge_function_recovery": 1.0,
      "mean_ood_response_recovery_ratio": 0.6706155828053206,
      "mean_post_repair_viability": 0.8617777777777779
    }
  },
  "claim_firewall": "Damage/repair dynamics are a synthetic mechanism diagnostic and do not establish regeneration of self, consciousness, or identity."
}

## 13. Adversarial Interpretation

Gate table:

- G0_IMPLEMENTATION: PASS
- G1_OOD_EFFECT_SIZE: PASS
- G2_OOD_CONFIDENCE: PASS
- G3_SEED_CONSISTENCY: PASS
- G4_COMMON_DRIVER: PASS

Warnings: `[]`. Precision, recall, specificity, edge density,
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

Under the preregistered synthetic system and held-out intervention regime, a PASS supports only improved recovery of boundary-candidate causal edges and survival of the specified common-driver falsification. It does not establish a real individual, metaphysical self-boundary, consciousness, first-person privacy, or substrate-independent identity.
