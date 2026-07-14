"""Covariance and mutual-information dynamics."""

from __future__ import annotations

import numpy as np

from myopic_mi.costs import Matrix, symmetrize


def posterior_covariance(previous_covariance: Matrix, action: Matrix) -> Matrix:
    """Compute P_t = (P_{t-1}^{-1} + A_t)^(-1) without explicit inversion."""
    return symmetrize(np.linalg.solve(np.eye(2) + previous_covariance @ action, previous_covariance))


def mutual_information_attention(previous_covariance: Matrix, action: Matrix) -> float:
    """Compute 0.5 * log(det(I + P A))."""
    return float(0.5 * np.log(np.linalg.det(np.eye(2) + previous_covariance @ action)))


def action_from_l_factor(l1: float, l2: float, l3: float) -> Matrix:
    """Build A = L L' with L = [[l1, 0], [l2, l3]]."""
    factor = np.array([[l1, 0.0], [l2, l3]], dtype=float)
    return symmetrize(factor @ factor.T)
