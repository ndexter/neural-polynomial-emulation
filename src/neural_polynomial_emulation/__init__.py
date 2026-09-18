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
    emulator_complexity,
    legendre_basis_error_by_degree,
    relative_frobenius_mismatch,
)
from .feedforward import FeedforwardProductNet, RePU2Activation, to_feedforward
from .formulas import PolynomialFormula, basis_polynomial_formula
from .legendre import (
    LegendreEmulator,
    exact_legendre,
    legendre_leading_coefficient,
    legendre_roots,
)
from .multi_index import (
    hyperbolic_cross_indices,
    is_downward_closed,
    multi_index_set,
    total_degree_indices,
    validate_multi_index,
)
from .multivariate import (
    MultivariateBasisEmulator,
    emulated_multivariate_basis,
    exact_multivariate_basis,
)
from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet
from .squaring import OPTIMAL_TANH_BIAS, ReLUSquaringNet, TanhSquaringNet
from .visualization import interactive_network_figure

__all__ = [
    "OPTIMAL_TANH_BIAS",
    "ChebyshevBasisEmulator",
    "ChebyshevEmulator",
    "FeedforwardProductNet",
    "LegendreBasisEmulator",
    "LegendreEmulator",
    "MultivariateBasisEmulator",
    "PolynomialFormula",
    "ReLUProductNet",
    "ReLUSquaringNet",
    "RePU2Activation",
    "RePU2ProductNet",
    "TanhProductNet",
    "TanhSquaringNet",
    "basis_error_by_degree",
    "basis_polynomial_formula",
    "chebyshev_basis_error_by_degree",
    "chebyshev_l2_norm",
    "chebyshev_leading_coefficient",
    "chebyshev_roots",
    "emulated_chebyshev_basis",
    "emulated_legendre_basis",
    "emulated_multivariate_basis",
    "emulator_complexity",
    "exact_chebyshev",
    "exact_chebyshev_basis",
    "exact_legendre",
    "exact_legendre_basis",
    "exact_multivariate_basis",
    "hyperbolic_cross_indices",
    "interactive_network_figure",
    "is_downward_closed",
    "legendre_basis_error_by_degree",
    "legendre_leading_coefficient",
    "legendre_roots",
    "multi_index_set",
    "relative_frobenius_mismatch",
    "to_feedforward",
    "total_degree_indices",
    "validate_multi_index",
]

__version__ = "0.5.0"
