from dibt.r2a_config import load_phase0_r2a_config


def test_r2a_fixed_development_design():
    cfg = load_phase0_r2a_config("configs/phase0_r2a.yaml")
    assert cfg.experiment.seeds == 10
    assert cfg.clustering.min_clusters == 2
    assert cfg.clustering.max_clusters == 5
    assert len(
        {
            cfg.trajectory_seed_offset,
            cfg.node_permutation_seed_offset,
            cfg.random_baseline_seed_offset,
        }
    ) == 3
    assert cfg.trajectory_seed_offset not in {0, 1_000_000, 2_000_000}
