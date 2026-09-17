"""Exact and neural evaluation of probability-normalized Legendre polynomials."""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import torch

from ._root_factorized import _RootFactorizedPolynomialEmulator, _validate_degree
from .squaring import OPTIMAL_TANH_BIAS


def legendre_roots(degree: int) -> np.ndarray:
    """Return the roots of the classical Legendre polynomial ``P_degree``."""
    degree = _validate_degree(degree)
    if degree == 0:
        return np.empty(0, dtype=np.float64)
    if degree == 1:
        return np.array([0.0], dtype=np.float64)
    k = np.arange(1, degree, dtype=np.float64)
    off_diagonal = k / np.sqrt(4.0 * k**2 - 1.0)
    jacobi = np.diag(off_diagonal, -1) + np.diag(off_diagonal, 1)
    return np.sort(np.linalg.eigvalsh(jacobi))


def legendre_leading_coefficient(degree: int) -> float:
    """Leading coefficient of ``sqrt(2n+1) P_n`` for ``d rho = dx/2``."""
    degree = _validate_degree(degree)
    classical = math.comb(2 * degree, degree) / (2.0**degree)
    return math.sqrt(2.0 * degree + 1.0) * classical


@overload
def exact_legendre(x: torch.Tensor, degree: int) -> torch.Tensor: ...


@overload
def exact_legendre(x: np.ndarray, degree: int) -> np.ndarray: ...


def exact_legendre(x, degree: int):
    """Evaluate ``sqrt(2n+1) P_n(x)`` by the three-term recurrence.

    NumPy inputs produce NumPy outputs. PyTorch inputs preserve tensor dtype,
    device, and differentiability with respect to ``x``.
    """
    degree = _validate_degree(degree)
    if isinstance(x, torch.Tensor):
        if not torch.is_floating_point(x):
            raise TypeError("x must be floating point")
        if degree == 0:
            return torch.ones_like(x)
        p0 = torch.ones_like(x)
        p1 = x
        if degree == 1:
            return math.sqrt(3.0) * p1
        for n in range(1, degree):
            p0, p1 = p1, ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
        return math.sqrt(2.0 * degree + 1.0) * p1

    values = np.asarray(x)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    if degree == 0:
        return np.ones_like(values)
    p0 = np.ones_like(values)
    p1 = values.copy()
    if degree == 1:
        return math.sqrt(3.0) * p1
    for n in range(1, degree):
        p0, p1 = p1, ((2 * n + 1) * values * p1 - n * p0) / (n + 1)
    return math.sqrt(2.0 * degree + 1.0) * p1


class LegendreEmulator(_RootFactorizedPolynomialEmulator):
    """Root-factorized neural emulator of ``sqrt(2n+1) P_n`` for ``dx/2``."""

    def __init__(
        self,
        degree: int,
        product: str = "relu",
        num_layers: int = 8,
        tanh_step: float | None = None,
        tanh_bias: float = OPTIMAL_TANH_BIAS,
        tanh_tolerance: float = 1e-4,
    ):
        degree = _validate_degree(degree)
        super().__init__(
            legendre_roots(degree),
            legendre_leading_coefficient(degree),
            product=product,
            num_layers=num_layers,
            tanh_step=tanh_step,
            tanh_bias=tanh_bias,
            tanh_tolerance=tanh_tolerance,
        )
