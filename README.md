# DIBT Phase 0 — Individuality Claim Falsification Suite

This repository implements the **Phase 0 confirmatory/falsification scaffold**
for Dynamic Individual Boundary Theory (DIBT).

The implementation follows the narrowest defensible DIBT claim:

> In a synthetic open dynamical system with a known causal boundary, under a
> preregistered spatial resolution, temporal window, admissible intervention
> set, and viability region, can a DIBT-style boundary estimator recover the
> true boundary edges better than a dynamic conditional-dependence baseline
> while rejecting a common-driver false boundary?

## Phase 0 scope

Primary:

- Known-boundary synthetic open system
- Primary endpoint: edge-level Matthews correlation coefficient (MCC)
- Baseline: Gaussian dynamic conditional-mutual-information approximation
- Primary falsification: `COMMON_DRIVER`
- Secondary mechanism check: `BOUNDARY_REMOVAL`

Not claimed:

- consciousness
- qualia
- first-person subjectivity
- soul
- metaphysical identity
- substrate-independent selfhood

## Important implementation note

The DIBT v1.0 paper specifies the experimental logic, endpoint, baseline
family, and falsification conditions, but it does not uniquely fix a numerical
estimator. This starter therefore implements a **reference estimator**:

1. lagged Gaussian conditional mutual information (CMI) proxy
2. direct-intervention effect size
3. an AND gate requiring both observational/conditional evidence and
   intervention-linked evidence

This is intentionally conservative against common-driver false positives. It
should not be described as *the* final DIBT estimator.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e ".[dev]"
python experiments/run_phase0.py --config configs/phase0.yaml
pytest -q
```

Outputs are written under `results/phase0/`.

For a short runtime check:

```bash
python experiments/run_phase0.py --config configs/smoke.yaml
```

## Phase 0-r1 held-out protocol

Phase 0-r1 repairs the original same-trajectory evaluation. It uses independent
TRAIN, VALIDATION, and OOD_TEST RNG streams and intervention amplitudes. CMI and
DO thresholds are fit on TRAIN INTACT data only and frozen before OOD scoring.
The held-out comparison includes CMI-only, DO-only, and the CMI∧DO reference
estimator.

The primary endpoint is MCC inside an evaluation-only boundary-candidate mask
(`E↔B` and `B↔I`). Full observed causal-graph MCC remains a separate secondary
output. The generator partition and edge truth never enter estimator inputs.

Run the r1 tests and a smoke check:

```bash
python experiments/validate_phase0_r1.py \
  --output-dir results/phase0_r1_smoke
python experiments/run_phase0_r1.py \
  --config configs/phase0_r1_smoke.yaml \
  --preregistration configs/preregistration_phase0_r1.yaml \
  --commit-sha SMOKE_UNFROZEN \
  --preregistration-sha <preregistration-commit> \
  --mode smoke
```

The confirmatory command requires the frozen implementation commit SHA and
refuses to overwrite an existing frozen result. After the primary verdict is
frozen, the secondary damage/repair extension can run with:

```bash
python experiments/run_regeneration.py \
  --config configs/phase0_r1.yaml \
  --output-dir results/phase0_r1
```

Smoke outcomes are implementation diagnostics, not scientific support. A STOP
outcome is retained without threshold or parameter tuning.

## Phase 0-r2A blind partition feasibility

r2A is a one-shot 10-seed development gate that runs before any r2B integrated
confirmatory experiment. Every trial independently permutes node order. The
blind estimator receives no generator partition, group sizes, true edges,
boundary mask, original node order, or generator-family label.

The fixed candidate concatenates observational CMI profiles with direct-
intervention response profiles, infers the number of groups from silhouette
scores over `k=2..5`, and compares against random, correlation-clustering, and
DO-profile baselines. The continuation endpoint is paired ARI improvement over
the best baseline plus all-pair end-to-end boundary-relation MCC.

```bash
python experiments/validate_phase0_r2a.py \
  --output-dir results/phase0_r2a
python experiments/run_phase0_r2a.py \
  --config configs/phase0_r2a.yaml \
  --preregistration configs/preregistration_phase0_r2a.yaml \
  --implementation-sha <frozen-r2a-implementation> \
  --preregistration-sha <r2a-preregistration>
```

If r2A returns `STOP_PARTITION_NOT_IDENTIFIED`, the protocol prohibits further
estimator search and r2B execution. A PASS authorizes only a separate r2B
preregistration; it does not establish integration value.

## Decision logic

Primary support requires all of:

1. mean ΔMCC = MCC_DIBT - MCC_baseline >= 0.05
2. seed-bootstrap 95% CI lower bound > 0
3. common-driver false-positive gate passes
4. numerical, transition-alignment, and seed-reproducibility audits pass

Boundary-removal viability effects are reported as mechanism diagnostics in
Phase 0 and are not promoted to consciousness or subjectivity claims.

The supplied `viability_radius` is checked against the theoretical maximum
internal-state norm induced by the `tanh` dynamics. If the radius makes the
viability fraction structurally equal to 1, the run reports a diagnostic
warning; this secondary diagnostic does not affect the primary verdict.

## Data alignment

Each simulated transition stores natural state `x(t)`, the possibly
intervened transition input, and the resulting state `x(t+1)` separately. This
prevents an intervention at `t+1` from contaminating the measured response to
an intervention at `t`.

## Repository layout

```text
DIBT_Phase0_Starter/
├── README.md
├── IMPLEMENTATION_SPEC.md
├── pyproject.toml
├── configs/
│   ├── phase0.yaml
│   ├── phase0_r1.yaml
│   ├── phase0_r1_smoke.yaml
│   ├── preregistration.yaml
│   ├── preregistration_phase0_r1.yaml
│   └── smoke.yaml
├── experiments/
│   ├── run_phase0.py
│   ├── run_phase0_r1.py
│   ├── run_regeneration.py
│   └── validate_phase0_r1.py
├── src/dibt/
│   ├── __init__.py
│   ├── config.py
│   ├── dynamics.py
│   ├── estimators.py
│   ├── metrics.py
│   ├── phase0.py
│   ├── phase0_r1.py
│   ├── evaluation.py
│   ├── decision.py
│   ├── splits.py
│   └── regeneration.py
└── tests/
    ├── test_alignment.py
    ├── test_estimators.py
    ├── test_metrics.py
    ├── test_nulls.py
    └── test_reproducibility.py
```
