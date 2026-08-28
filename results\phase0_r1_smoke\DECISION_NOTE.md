# DIBT Phase 0-r1 Decision Note

## 1. Executive Verdict

**STOP_OOD_EFFECT_TOO_SMALL**

First failed gate: `G1_OOD_EFFECT_SIZE`  
All failed gates: `['G1_OOD_EFFECT_SIZE', 'G2_OOD_CONFIDENCE', 'G3_SEED_CONSISTENCY']`

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

Mean boundary MCC: `0.748328`.

## 5. DO-only Baseline

Mean boundary MCC: `0.258199`.

## 6. DIBT CMI-and-DO Result

Mean boundary MCC: `0.258199`.

## 7. Delta MCC and Bootstrap CI

- Mean ΔMCC vs CMI-only: `-0.490129`
- Seed-bootstrap 95% CI: `[-0.8164965809277259, 0.09146661817751023]`
- Mean ΔMCC vs DO-only: `0.000000`
- Secondary 95% CI: `[0.0, 0.0]`

## 8. Positive-Seed Consistency

`1 / 3` seeds were positive
(`0.333`).

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

Not yet executed. The extension may run only after the primary verdict is frozen.

## 13. Adversarial Interpretation

Gate table:

- G0_IMPLEMENTATION: PASS
- G1_OOD_EFFECT_SIZE: FAIL
- G2_OOD_CONFIDENCE: FAIL
- G3_SEED_CONSISTENCY: FAIL
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
