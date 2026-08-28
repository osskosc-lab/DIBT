import numpy as np

from dibt.estimators import (
    InterventionalInput,
    ObservationalInput,
    score_cmi,
    score_do,
)
from dibt.r1_config import load_phase0_r1_config
from dibt.splits import blind_for_estimation, generate_split


def _data():
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    sim = generate_split(cfg.simulation, cfg.train, 5, condition="intact")
    return cfg, blind_for_estimation(sim, "TRAIN")


def test_do_only_baseline():
    cfg, data = _data()
    original = score_do(data.interventional, cfg.estimation)
    # DO-only has no transition-input/CMI argument to observe.
    changed_observational = ObservationalInput(
        transition_inputs=np.full_like(
            data.observational.transition_inputs, 999.0
        ),
        next_states=data.observational.next_states,
    )
    assert changed_observational.transition_inputs[0, 0] == 999.0
    repeated = score_do(data.interventional, cfg.estimation)
    assert np.array_equal(original, repeated)


def test_cmi_only_baseline():
    cfg, data = _data()
    original = score_cmi(data.observational, cfg.estimation)
    # CMI-only has no intervention-label/amplitude argument to observe.
    changed_interventional = InterventionalInput(
        intervention_index=np.full_like(
            data.interventional.intervention_index, -1
        ),
        intervention_value=np.full_like(
            data.interventional.intervention_value, 777.0
        ),
        next_states=data.interventional.next_states,
    )
    assert changed_interventional.intervention_value[0] == 777.0
    repeated = score_cmi(data.observational, cfg.estimation)
    assert np.array_equal(original, repeated)
