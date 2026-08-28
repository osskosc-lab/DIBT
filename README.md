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
│   ├── preregistration.yaml
│   └── smoke.yaml
├── experiments/
│   └── run_phase0.py
├── src/dibt/
│   ├── __init__.py
│   ├── config.py
│   ├── dynamics.py
│   ├── estimators.py
│   ├── metrics.py
│   └── phase0.py
└── tests/
    ├── test_alignment.py
    ├── test_estimators.py
    ├── test_metrics.py
    ├── test_nulls.py
    └── test_reproducibility.py
```

