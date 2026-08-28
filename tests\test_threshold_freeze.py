from dataclasses import asdict

import numpy as np
import pytest

from dibt.estimators import (
    FrozenThresholds,
    apply_dibt_reference,
    fit_thresholds,
    make_estimator_input,
    score_dataset,
)
from dibt.r1_config import load_phase0_r1_config
from dibt.splits import blind_for_estimation, generate_split


def _datasets():
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    train = generate_split(cfg.simulation, cfg.train, 2, condition="intact")
    ood = generate_split(cfg.simulation, cfg.ood_test, 2, condition="intact")
    return cfg, blind_for_estimation(train, "TRAIN"), blind_for_estimation(
        ood, "OOD_TEST"
    )


def test_thresholds_fit_on_train_only():
    cfg, train, ood = _datasets()
    thresholds = fit_thresholds(train, cfg.estimation, seed=2)
    assert thresholds.source_split == "TRAIN"
    with pytest.raises(ValueError, match="TRAIN"):
        fit_thresholds(ood, cfg.estimation, seed=2)


def test_frozen_threshold_application():
    cfg, train, ood = _datasets()
    thresholds = fit_thresholds(train, cfg.estimation, seed=2)
    snapshot = FrozenThresholds(**asdict(thresholds))
    changed_ood = make_estimator_input(
        ood.observational.transition_inputs,
        ood.observational.next_states * -5.0,
        ood.interventional.intervention_index,
        ood.interventional.intervention_value,
        split_name="OOD_TEST",
    )
    apply_dibt_reference(score_dataset(changed_ood, cfg.estimation), thresholds)
    assert thresholds == snapshot
