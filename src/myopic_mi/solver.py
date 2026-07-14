"""Exact-grid stage solver for the mutual-information attention constraint."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from myopic_mi.costs import Matrix
from myopic_mi.dynamics import action_from_l_factor, mutual_information_attention, posterior_covariance

CostFunction = Callable[[Matrix], float]


def admissible_l2_roots_mi(
    covariance: Matrix,
    alpha: float,
    l1: float,
    l3: float,
    l1sq: float,
    l3sq: float,
    disc_tol: float,
) -> list[float]:
    """Return l2 roots that satisfy the per-stage MI equality for fixed l1 and l3."""
    p11 = covariance[0, 0]
    p12 = covariance[0, 1]
    p22 = covariance[1, 1]
    det_p = np.linalg.det(covariance)

    a = p22
    b = 2.0 * p12 * l1
    c = 1.0 + p11 * l1sq + p22 * l3sq + det_p * (l1sq * l3sq) - np.exp(2.0 * alpha)
    disc = b * b - 4.0 * a * c
    if disc < -disc_tol:
        return []

    disc = max(float(disc), 0.0)
    root_disc = np.sqrt(disc)
    r1 = float((-b - root_disc) / (2.0 * a))
    r2 = float((-b + root_disc) / (2.0 * a))
    if abs(r1 - r2) < 1e-14:
        return [r1]
    return [r1, r2]


def lexicographically_less(candidate: list[float], incumbent: list[float], tolerance: float) -> bool:
    """Compare deterministic tie-breaker keys with tolerance."""
    for candidate_value, incumbent_value in zip(candidate, incumbent):
        if candidate_value < incumbent_value - tolerance:
            return True
        if candidate_value > incumbent_value + tolerance:
            return False
    return False


def solve_stage_mi_exact_grid(
    covariance: Matrix,
    alpha: float,
    loss_fn: CostFunction,
    lmin: float,
    lmax: float,
    grid_step: float,
    disc_tol: float,
    mi_tol: float,
    tie_tol: float,
) -> tuple[Matrix, Matrix, float]:
    """Solve one myopic stage over the exact-grid MI action parameterization."""
    l_values = np.arange(lmin, lmax + 1e-12, grid_step)
    found_any = False
    best_action = np.zeros((2, 2))
    best_covariance = covariance.copy()
    best_value = np.inf
    best_key = [np.inf, np.inf, np.inf, np.inf, np.inf]

    for l1 in l_values:
        l1sq = float(l1 * l1)
        for l3 in l_values:
            l3sq = float(l3 * l3)
            roots_l2 = admissible_l2_roots_mi(covariance, alpha, float(l1), float(l3), l1sq, l3sq, disc_tol)
            for l2 in roots_l2:
                if not math.isfinite(l2):
                    continue

                action = action_from_l_factor(float(l1), l2, float(l3))
                mi_value = mutual_information_attention(covariance, action)
                if abs(mi_value - alpha) > mi_tol:
                    continue

                next_covariance = posterior_covariance(covariance, action)
                value = loss_fn(next_covariance)
                candidate_key = [value, float(np.trace(next_covariance)), float(l1), abs(l2), float(l3)]

                if (not found_any) or lexicographically_less(candidate_key, best_key, tie_tol):
                    found_any = True
                    best_action = action
                    best_covariance = next_covariance
                    best_value = value
                    best_key = candidate_key

    if not found_any:
        raise RuntimeError("No feasible action found at this stage.")
    return best_action, best_covariance, float(best_value)
