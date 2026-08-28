# Phase 0-r2A Blind Partition Feasibility Decision

## Verdict

**STOP_PARTITION_NOT_IDENTIFIED**

## Fixed development design

Ten F0 development trajectories used independent r2A RNG streams. Node order
was independently permuted before estimator input. The estimator received only
transition arrays and intervention metadata.

## Implementation audits

- pytest: PASS
- blind_input_contract: PASS
- forbidden_count_blinding: PASS
- node_permutation_equivariance: PASS
- seed_reproducibility: PASS
- rng_isolation: PASS
- numerical_finiteness: PASS

## Partition results

- Mean candidate ARI: `0.051429`
- Mean best-baseline ARI: `0.156764`
- Mean paired ΔARI: `-0.105336`
- Bootstrap 95% CI: `[-0.18858194505872522, -0.034804760770705115]`
- Positive seeds: `2/10`

## End-to-end all-pair boundary relation

- Candidate mean MCC: `0.300000`
- Best baseline mean MCC: `0.026971`

## Gates

- G0_IMPLEMENTATION: PASS
- G1_MEAN_ARI_ADVANTAGE: FAIL
- G2_ARI_CONFIDENCE: FAIL
- G3_SEED_CONSISTENCY: FAIL
- G4_END_TO_END_BOUNDARY: PASS

## Continuation decision

`r2b_authorized = False`. If false, the protocol stops
without searching additional integrated estimators.

## Claim firewall

Even a PASS would support only blind recovery of synthetic partitions and
boundary relations in the fixed F0 development setting. It would not identify
a real individual, self, consciousness, subjectivity, or integration value.
