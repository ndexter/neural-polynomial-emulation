"""Constructive neural emulators for probability-normalized polynomials."""

from .basis import LegendreBasisEmulator, emulated_legendre_basis, exact_legendre_basis
from .diagnostics import basis_error_by_degree, relative_frobenius_mismatch
from .legendre import (
    LegendreEmulator,
    exact_legendre,
    legendre_leading_coefficient,
    legendre_roots,
)
from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet
from .squaring import OPTIMAL_TANH_BIAS, ReLUSquaringNet, TanhSquaringNet

__all__ = [
    "OPTIMAL_TANH_BIAS",
    "LegendreBasisEmulator",
    "LegendreEmulator",
    "ReLUProductNet",
    "ReLUSquaringNet",
    "RePU2ProductNet",
    "TanhProductNet",
    "TanhSquaringNet",
    "basis_error_by_degree",
    "emulated_legendre_basis",
    "exact_legendre",
    "exact_legendre_basis",
    "legendre_leading_coefficient",
    "legendre_roots",
    "relative_frobenius_mismatch",
]

__version__ = "0.3.0"
