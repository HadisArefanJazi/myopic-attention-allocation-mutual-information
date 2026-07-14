"""Tests for MI stage-solver primitives."""

from __future__ import annotations

import numpy as np

from myopic_mi.costs import trace_cost
from myopic_mi.dynamics import action_from_l_factor, mutual_information_attention, posterior_covariance
from myopic_mi.solver import admissible_l2_roots_mi, solve_stage_mi_exact_grid


def test_action_and_posterior_are_symmetric():
    covariance = np.array([[2.0, 0.3], [0.3, 1.5]])
    action = action_from_l_factor(0.2, -0.1, 0.3)

    posterior = posterior_covariance(covariance, action)

    assert np.allclose(action, action.T)
    assert np.allclose(posterior, posterior.T)


def test_admissible_roots_satisfy_mi_constraint():
    covariance = np.eye(2)
    alpha = 0.05
    roots = admissible_l2_roots_mi(covariance, alpha, 0.0, 0.0, 0.0, 0.0, 1e-12)

    assert roots
    action = action_from_l_factor(0.0, roots[0], 0.0)
    assert np.isclose(mutual_information_attention(covariance, action), alpha)


def test_solve_stage_returns_feasible_action():
    covariance = np.eye(2)
    alpha = 0.05

    action, posterior, value = solve_stage_mi_exact_grid(
        covariance,
        alpha,
        trace_cost,
        lmin=0.0,
        lmax=0.5,
        grid_step=0.1,
        disc_tol=1e-12,
        mi_tol=1e-10,
        tie_tol=1e-14,
    )

    assert np.isclose(mutual_information_attention(covariance, action), alpha)
    assert value == trace_cost(posterior)
