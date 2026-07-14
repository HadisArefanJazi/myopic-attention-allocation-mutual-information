"""Mutual-information myopic attention-allocation experiments."""

from myopic_mi.experiment import ExperimentConfig, ExperimentResult, run_experiment
from myopic_mi.solver import solve_stage_mi_exact_grid

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment", "solve_stage_mi_exact_grid"]
