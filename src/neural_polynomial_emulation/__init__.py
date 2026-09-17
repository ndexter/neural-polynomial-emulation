"""Constructive neural emulators for probability-normalized polynomials."""

from .basis import LegendreBasisEmulator, emulated_legendre_basis, exact_legendre_basis
from .chebyshev import (
    ChebyshevEmulator,
    chebyshev_l2_norm,
    chebyshev_leading_coefficient,
    chebyshev_roots,
    exact_chebyshev,
)
from .chebyshev_basis import (
    ChebyshevBasisEmulator,
    emulated_chebyshev_basis,
    exact_chebyshev_basis,
)
from .diagnostics import (
    basis_error_by_degree,
    chebyshev_basis_error_by_degree,
    legendre_basis_error_by_degree,
    relative_frobenius_mismatch,
)
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
    "ChebyshevBasisEmulator",
    "ChebyshevEmulator",
    "LegendreBasisEmulator",
    "LegendreEmulator",
    "ReLUProductNet",
    "ReLUSquaringNet",
    "RePU2ProductNet",
    "TanhProductNet",
    "TanhSquaringNet",
    "basis_error_by_degree",
    "chebyshev_basis_error_by_degree",
    "chebyshev_l2_norm",
    "chebyshev_leading_coefficient",
    "chebyshev_roots",
    "emulated_chebyshev_basis",
    "emulated_legendre_basis",
    "exact_chebyshev",
    "exact_chebyshev_basis",
    "exact_legendre",
    "exact_legendre_basis",
    "legendre_basis_error_by_degree",
    "legendre_leading_coefficient",
    "legendre_roots",
    "relative_frobenius_mismatch",
]

__version__ = "0.4.0"
