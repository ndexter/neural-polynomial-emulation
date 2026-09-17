"""Numerical diagnostics intrinsic to polynomial emulation."""

from __future__ import annotations

import numpy as np
import torch

from .basis import emulated_legendre_basis, exact_legendre_basis
from .chebyshev_basis import emulated_chebyshev_basis, exact_chebyshev_basis
from .squaring import OPTIMAL_TANH_BIAS


def basis_error_by_degree(
    max_degree: int,
    *,
    family: str = "legendre",
    product: str = "relu",
    num_layers: int = 8,
    grid_size: int = 2001,
    tanh_step: float | None = None,
    tanh_bias: float = OPTIMAL_TANH_BIAS,
    tanh_tolerance: float = 1e-4,
) -> np.ndarray:
    """Return grid maximum errors for one polynomial family.

    ``family`` is ``"legendre"`` for the ``dx/2``-orthonormal basis or
    ``"chebyshev"`` for the Chebyshev-probability-orthonormal basis.
    """
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    x = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    family = str(family).lower()
    if family == "legendre":
        exact_basis = exact_legendre_basis
        emulated_basis = emulated_legendre_basis
    elif family == "chebyshev":
        exact_basis = exact_chebyshev_basis
        emulated_basis = emulated_chebyshev_basis
    else:
        raise ValueError("family must be 'legendre' or 'chebyshev'")
    exact = exact_basis(x, max_degree)
    with torch.no_grad():
        approximate = emulated_basis(
            x,
            max_degree,
            product=product,
            num_layers=num_layers,
            tanh_step=tanh_step,
            tanh_bias=tanh_bias,
            tanh_tolerance=tanh_tolerance,
        )
    return torch.amax(torch.abs(approximate - exact), dim=0).cpu().numpy()


def legendre_basis_error_by_degree(max_degree: int, **kwargs) -> np.ndarray:
    """Return Legendre-emulation grid errors through ``max_degree``."""
    return basis_error_by_degree(max_degree, family="legendre", **kwargs)


def chebyshev_basis_error_by_degree(
    max_degree: int,
    *,
    product: str = "relu",
    num_layers: int = 8,
    grid_size: int = 2001,
    tanh_step: float | None = None,
    tanh_bias: float = OPTIMAL_TANH_BIAS,
    tanh_tolerance: float = 1e-4,
) -> np.ndarray:
    """Return Chebyshev-emulation grid errors through ``max_degree``."""
    return basis_error_by_degree(
        max_degree,
        family="chebyshev",
        product=product,
        num_layers=num_layers,
        grid_size=grid_size,
        tanh_step=tanh_step,
        tanh_bias=tanh_bias,
        tanh_tolerance=tanh_tolerance,
    )


def relative_frobenius_mismatch(exact, approximate) -> float:
    """Return ``||approximate-exact||_F / ||exact||_F``."""
    exact_array = np.asarray(exact, dtype=np.float64)
    approximate_array = np.asarray(approximate, dtype=np.float64)
    if exact_array.shape != approximate_array.shape:
        raise ValueError("exact and approximate must have the same shape")
    denominator = float(np.linalg.norm(exact_array, ord="fro"))
    if denominator == 0.0:
        raise ValueError("exact must have nonzero Frobenius norm")
    return float(
        np.linalg.norm(approximate_array - exact_array, ord="fro") / denominator
    )
