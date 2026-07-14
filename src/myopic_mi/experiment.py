"""Experiment orchestration for mutual-information attention allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myopic_mi.costs import Matrix, frobenius_cost, off_diagonal_cost, symmetrize, trace_cost
from myopic_mi.solver import CostFunction, solve_stage_mi_exact_grid


@dataclass(frozen=True)
class Scenario:
    """A named stage-loss scenario."""

    name: str
    short_name: str
    loss_fn: CostFunction


@dataclass(frozen=True)
class ExperimentConfig:
    """Numerical settings for the mutual-information experiment."""

    horizon: int = 10
    gamma_start: float = 0.5
    gamma_stop: float = 2.5
    gamma_step: float = 0.5
    gamma_target: float = 0.5
    grid_step: float = 0.0005
    lmin: float = 0.0
    lmax: float = 2.0
    disc_tol: float = 1e-12
    mi_tol: float = 1e-10
    tie_tol: float = 1e-14

    @property
    def gamma_list(self) -> np.ndarray:
        """Return the gamma grid from the original script."""
        return np.arange(self.gamma_start, self.gamma_stop + 1e-12, self.gamma_step)


@dataclass(frozen=True)
class ExperimentResult:
    """Numerical output from a mutual-information experiment."""

    config: ExperimentConfig
    initial_covariance: Matrix
    scenario_names: list[str]
    trace_totals: np.ndarray
    trace_paths: np.ndarray
    instant_costs: np.ndarray
    regrets: np.ndarray
    covariance_paths: list[list[list[Matrix]]]
    action_paths: list[list[list[Matrix]]]


def default_initial_covariance() -> Matrix:
    """Return P0 = Q diag(5, 1) Q' with a 30-degree rotation."""
    theta = 30.0 * np.pi / 180.0
    q_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    return symmetrize(q_matrix @ np.diag([5.0, 1.0]) @ q_matrix.T)


def default_scenarios() -> list[Scenario]:
    """Return the three original loss scenarios."""
    return [
        Scenario("Scenario 1 (Benchmark)", "L^{(1)}", trace_cost),
        Scenario("Scenario 2", "L^{(2)}", off_diagonal_cost),
        Scenario("Scenario 3", "L^{(3)}", frobenius_cost),
    ]


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Run the mutual-information myopic attention experiment."""
    config = config or ExperimentConfig()
    scenarios = default_scenarios()
    gamma_list = config.gamma_list
    initial_covariance = default_initial_covariance()
    scenario_count = len(scenarios)
    gamma_count = len(gamma_list)

    trace_totals = np.zeros((scenario_count, gamma_count))
    trace_paths = np.zeros((scenario_count, gamma_count, config.horizon))
    instant_costs = np.zeros((scenario_count, gamma_count, config.horizon))
    covariance_paths: list[list[list[Matrix]]] = [[None for _ in range(gamma_count)] for _ in range(scenario_count)]  # type: ignore[list-item]
    action_paths: list[list[list[Matrix]]] = [[None for _ in range(gamma_count)] for _ in range(scenario_count)]  # type: ignore[list-item]

    for gamma_index, gamma in enumerate(gamma_list):
        alpha = gamma / config.horizon
        for scenario_index, scenario in enumerate(scenarios):
            covariance_sequence: list[Matrix] = [initial_covariance.copy()]
            action_sequence: list[Matrix] = []
            current_covariance = initial_covariance.copy()

            for stage in range(config.horizon):
                action, next_covariance, stage_value = solve_stage_mi_exact_grid(
                    current_covariance,
                    alpha,
                    scenario.loss_fn,
                    config.lmin,
                    config.lmax,
                    config.grid_step,
                    config.disc_tol,
                    config.mi_tol,
                    config.tie_tol,
                )
                action_sequence.append(action)
                covariance_sequence.append(next_covariance)
                trace_paths[scenario_index, gamma_index, stage] = np.trace(next_covariance)
                instant_costs[scenario_index, gamma_index, stage] = stage_value
                current_covariance = next_covariance

            trace_totals[scenario_index, gamma_index] = np.sum(trace_paths[scenario_index, gamma_index, :])
            covariance_paths[scenario_index][gamma_index] = covariance_sequence
            action_paths[scenario_index][gamma_index] = action_sequence

    regrets = np.zeros((scenario_count, gamma_count, config.horizon))
    for gamma_index in range(gamma_count):
        benchmark = trace_paths[0, gamma_index, :]
        for scenario_index in range(1, scenario_count):
            regrets[scenario_index, gamma_index, :] = (
                trace_paths[scenario_index, gamma_index, :] - benchmark
            ) / benchmark

    return ExperimentResult(
        config=config,
        initial_covariance=initial_covariance,
        scenario_names=[scenario.name for scenario in scenarios],
        trace_totals=trace_totals,
        trace_paths=trace_paths,
        instant_costs=instant_costs,
        regrets=regrets,
        covariance_paths=covariance_paths,
        action_paths=action_paths,
    )
