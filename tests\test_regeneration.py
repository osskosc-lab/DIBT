from pathlib import Path

import pytest

from dibt.r1_config import load_phase0_r1_config
from dibt.regeneration import run_regeneration, simulate_regeneration


def test_regeneration_requires_frozen_primary_verdict(tmp_path: Path):
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    with pytest.raises(RuntimeError, match="primary verdict"):
        run_regeneration(cfg, output_dir=tmp_path)


def test_damage_and_repair_schedule_is_explicit():
    cfg = load_phase0_r1_config("configs/phase0_r1_smoke.yaml")
    no_repair = simulate_regeneration(cfg, 0, "DAMAGE_NO_REPAIR")
    with_repair = simulate_regeneration(cfg, 0, "DAMAGE_WITH_REPAIR")
    assert no_repair.coupling_factor[-1] == cfg.regeneration.damage_factor
    assert with_repair.coupling_factor[-1] == 1.0
