# DIBT Phase 0 Implementation Specification v0.1

## 1. Research target

Phase 0 does **not** test whether an entity "really is" an individual in a
metaphysical sense. It tests a narrower operational claim:

> Given a synthetic open system with a known causal cut, can the candidate
> boundary be recovered under fixed observation/intervention conditions, and
> can a common-driver configuration be rejected as a false boundary?

The simulation therefore separates:

- data generation
- estimation
- falsification
- decision gates

## 2. State partition

The synthetic state is

\[
Z_t = (E_t, B_t, I_t)
\]

with:

- `E`: external state
- `B`: boundary state
- `I`: internal state

The true boundary edge set is stored by the generator and never exposed to the
estimator.

## 3. Dynamics

Base dynamics:

\[
x_{t+1} = \tanh(A \tilde{x}_t + G c_t + \epsilon_t)
\]

where:

- `A` contains the true directed lag-1 causal graph
- `c_t` is an optional latent common driver
- `ε_t` is process noise
- `x̃_t` is `x_t` after any randomized direct intervention at time `t`

Each retained transition records `x_t`, `x̃_t`, `x_{t+1}`, the intervention
source, and its amplitude. Estimators consume only `x̃_t`, `x_{t+1}`, and the
same-transition intervention metadata.

### INTACT

Contains:

- E → B
- B → I
- I → B feedback
- internal and external self dynamics

### COMMON_DRIVER

All true E-B-I boundary-crossing causal edges are removed. A latent common
driver drives both E and I. This can create strong correlation without
boundary-mediated causation.

### BOUNDARY_REMOVAL

The true boundary-mediated edges are removed. It is a mechanism ablation, not
a null-equivalent model.

## 4. Primary endpoint

For all ordered candidate edges `i -> j`, compare estimated binary edge labels
to the generator truth and compute edge-level Matthews correlation coefficient:

\[
MCC(E_{\hat\Gamma}, E_{\Gamma_0})
\]

Primary effect:

\[
\Delta MCC = MCC_{DIBT} - MCC_{baseline}
\]

Preregistered support criterion from DIBT v1.0:

- mean ΔMCC >= 0.05
- seed-bootstrap 95% CI lower bound > 0

## 5. Baseline

The reference baseline is a Gaussian conditional mutual information
approximation. For each lagged edge `x_i(t) -> x_j(t+1)`:

1. regress the source on the remaining transition inputs at `t`
2. regress the target at `t+1` on the same controls
3. compute residual correlation `rho`
4. map it to Gaussian CMI

\[
I_G(X;Y|Z)=-\frac12\log(1-\rho^2)
\]

The edge threshold is estimated from permutation-null scores. This is a
concrete Gaussian approximation to the paper's broader dynamic
conditional-MI baseline family.

## 6. Reference DIBT estimator

Because v1.0 does not uniquely specify a numerical optimizer, this starter
uses a conservative two-evidence gate:

- observational/conditional evidence: Gaussian dynamic CMI
- interventional evidence: standardized response at `t+1` under direct
  interventions on the candidate source at `t`

An edge is accepted only if:

\[
CMI_{ij} > q_{obs}
\quad\text{AND}\quad
DO_{ij} > q_{do}
\]

Thresholds are estimated by within-dataset permutation nulls. The DIBT and
baseline estimators use the same observational null seed, giving them the same
CMI threshold on a paired trajectory.

## 7. Phase 0 conditions

Primary:

- INTACT
- COMMON_DRIVER

Secondary:

- BOUNDARY_REMOVAL

Later phases can add shuffled, reversed, source-swap, oracle-clone,
self-model, full-history, representation, scale, and reconstruction tests.

## 8. Leakage control

Estimator input at edge `i -> j` is restricted to:

- transition input at `t`
- target state at `t+1`
- intervention label/amplitude at `t`

No future transition input or state beyond `t+1` is used.

## 9. Seed design

Confirmatory run:

- 30 fixed seeds
- same seed set for DIBT and baseline
- same generated trajectory per paired comparison
- deterministic permutation thresholds
- seed-level bootstrap of the paired ΔMCC values

## 10. Claim firewall

Passing Phase 0 supports only boundary-edge recovery under this synthetic model
and intervention regime. It does not establish a unique observer-independent
individual boundary, consciousness, first-person privacy, selfhood,
ontological continuity, or fractality.
