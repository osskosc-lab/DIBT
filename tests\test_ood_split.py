import numpy as np

from dibt.r1_config import load_phase0_r1_config
from dibt.splits import (
    generate_split,
    split_amplitudes_are_disjoint,
    split_rng_streams_are_distinct,
)


def test_ood_interventions_are_disjoint():
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    assert split_amplitudes_are_disjoint(
        cfg.train, cfg.validation, cfg.ood_test
    )
    assert not (
        set(cfg.train.intervention_amplitudes)
        & set(cfg.ood_test.intervention_amplitudes)
    )


def test_split_rng_streams_are_independent():
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    assert split_rng_streams_are_distinct(
        cfg.train, cfg.validation, cfg.ood_test
    )
    train = generate_split(cfg.simulation, cfg.train, 0, condition="intact")
    ood = generate_split(cfg.simulation, cfg.ood_test, 0, condition="intact")
    assert not np.array_equal(train.states, ood.states)
