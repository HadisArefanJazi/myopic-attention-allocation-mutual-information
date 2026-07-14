"""Stage-cost functions for myopic mutual-information experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


def symmetrize(matrix: Matrix) -> Matrix:
    """Return the symmetric part of a matrix."""
    return 0.5 * (matrix + matrix.T)


def trace_cost(covariance: Matrix) -> float:
    """Benchmark trace cost."""
    return float(np.trace(covariance))


def off_diagonal_cost(covariance: Matrix) -> float:
    """Off-diagonal cost, 2 * |P_12|."""
    return float(2.0 * abs(covariance[0, 1]))


def frobenius_cost(covariance: Matrix) -> float:
    """Frobenius-norm stage cost."""
    return float(np.linalg.norm(covariance, ord="fro"))
