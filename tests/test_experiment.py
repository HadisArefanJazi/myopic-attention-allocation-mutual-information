"""Tests for compact MI experiment execution."""

from __future__ import annotations

import numpy as np

from myopic_mi.experiment import ExperimentConfig, run_experiment


def test_short_experiment_shapes_and_benchmark_regret():
    config = ExperimentConfig(
        horizon=2,
        gamma_start=0.5,
        gamma_stop=0.5,
        gamma_step=0.5,
        gamma_target=0.5,
        grid_step=0.1,
        lmax=0.5,
    )

    result = run_experiment(config)

    assert result.trace_totals.shape == (3, 1)
    assert result.trace_paths.shape == (3, 1, 2)
    assert result.regrets.shape == (3, 1, 2)
    assert np.allclose(result.regrets[0, 0, :], 0.0)
