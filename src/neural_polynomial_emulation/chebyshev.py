"""Exact and neural evaluation of probability-normalized Chebyshev polynomials."""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import torch

from ._root_factorized import _RootFactorizedPolynomialEmulator, _validate_degree
from .squaring import OPTIMAL_TANH_BIAS


def chebyshev_roots(degree: int) -> np.ndarray:
    """Return the roots of the first-kind Chebyshev polynomial ``T_degree``."""
    degree = _validate_degree(degree)
    if degree == 0:
        return np.empty(0, dtype=np.float64)
    indices = np.arange(1, degree + 1, dtype=np.float64)
    roots = np.cos((2.0 * indices - 1.0) * math.pi / (2.0 * degree))
    return np.sort(roots)


def chebyshev_l2_norm(degree: int) -> float:
    r"""Return the norm of ``T_degree`` for the Chebyshev probability measure.

    The measure is

    .. math::

        d\rho_C(x)=\frac{dx}{\pi\sqrt{1-x^2}}.

    The norm is one for degree zero and ``1 / sqrt(2)`` otherwise.
    """
    degree = _validate_degree(degree)
    if degree == 0:
        return 1.0
    return 1.0 / math.sqrt(2.0)


def chebyshev_leading_coefficient(degree: int) -> float:
    """Leading coefficient of the probability-orthonormal Chebyshev polynomial."""
    degree = _validate_degree(degree)
    classical = 1.0 if degree == 0 else 2.0 ** (degree - 1)
    return classical / chebyshev_l2_norm(degree)


@overload
def exact_chebyshev(x: torch.Tensor, degree: int) -> torch.Tensor: ...


@overload
def exact_chebyshev(x: np.ndarray, degree: int) -> np.ndarray: ...


def exact_chebyshev(x, degree: int):
    r"""Evaluate the probability-orthonormal Chebyshev polynomial by recurrence."""
    degree = _validate_degree(degree)
    normalization = 1.0 / chebyshev_l2_norm(degree)
    if isinstance(x, torch.Tensor):
        if not torch.is_floating_point(x):
            raise TypeError("x must be floating point")
        if degree == 0:
            return torch.ones_like(x)
        t0 = torch.ones_like(x)
        t1 = x
        if degree == 1:
            return normalization * t1
        for _ in range(1, degree):
            t0, t1 = t1, 2.0 * x * t1 - t0
        return normalization * t1

    values = np.asarray(x)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    if degree == 0:
        return np.ones_like(values)
    t0 = np.ones_like(values)
    t1 = values.copy()
    if degree == 1:
        return normalization * t1
    for _ in range(1, degree):
        t0, t1 = t1, 2.0 * values * t1 - t0
    return normalization * t1


class ChebyshevEmulator(_RootFactorizedPolynomialEmulator):
    r"""Root-factorized emulator of ``1`` or ``sqrt(2) T_n``."""

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
            chebyshev_roots(degree),
            chebyshev_leading_coefficient(degree),
            product=product,
            num_layers=num_layers,
            tanh_step=tanh_step,
            tanh_bias=tanh_bias,
            tanh_tolerance=tanh_tolerance,
        )
