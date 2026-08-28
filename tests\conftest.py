from dibt.config import EstimationConfig, SimulationConfig


def simulation_config(**overrides) -> SimulationConfig:
    values = {
        "steps": 300,
        "burn_in": 50,
        "n_external": 3,
        "n_boundary": 2,
        "n_internal": 3,
        "process_noise": 0.08,
        "common_driver_strength": 0.85,
        "intervention_probability": 0.25,
        "train_intervention_amplitudes": (-1.0, -0.5, 0.5, 1.0),
        "ood_intervention_amplitudes": (-2.0, -1.5, 1.5, 2.0),
        "viability_radius": 2.5,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def estimation_config(**overrides) -> EstimationConfig:
    values = {
        "ridge": 0.01,
        "permutation_repeats": 12,
        "permutation_quantile": 0.95,
        "min_intervention_samples": 5,
    }
    values.update(overrides)
    return EstimationConfig(**values)
