# Baseline Audit Before Phase 0-r1

Date: 2026-08-29

## Repository state

The target GitHub repository `osskosc-lab/DIBT` was empty when audited: it had
no commits, files, or branches. The baseline initialized here is the verified
DIBT Phase 0 starter produced immediately before the Phase 0-r1 request.

## Reproduction

- `python -m pytest -q`: 13 passed
- `python experiments/run_phase0.py --config configs/smoke.yaml`: completed
- Smoke verdict: `PHASE0_CONDITIONAL_SUPPORT`
- Smoke mean ΔMCC: `0.22353479633287124`
- Smoke bootstrap 95% CI: `[0.10556111258358869, 0.316515138991168]`
- Smoke common-driver all-edge rate: `0.029761904761904757`

## Known baseline limitations

- Estimator thresholds are fit and evaluated on the same trajectory.
- There is no DO-only baseline.
- MCC is evaluated across every ordered pair instead of a boundary-candidate
  evaluation mask.
- Common-driver falsification uses an all-edge prediction rate.
- The positive-seed consistency gate is absent.
- The supplied viability radius is structurally non-informative under the
  tanh-bounded three-dimensional internal state.
- The smoke run is not a preregistered 30-seed confirmatory result.

This snapshot is retained to make the Phase 0-r1 repair auditable.
