"""Tensor-product polynomial bases indexed by finite multi-index sets."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .basis import LegendreBasisEmulator, exact_legendre_basis
from .chebyshev_basis import ChebyshevBasisEmulator, exact_chebyshev_basis
from .multi_index import validate_multi_index
from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet
from .squaring import OPTIMAL_TANH_BIAS


def _family_apis(family: str):
    family = str(family).lower()
    if family == "legendre":
        return LegendreBasisEmulator, exact_legendre_basis
    if family == "chebyshev":
        return ChebyshevBasisEmulator, exact_chebyshev_basis
    raise ValueError("family must be 'legendre' or 'chebyshev'")


def _basis_bound(family: str, degree: int) -> float:
    if family == "legendre":
        return math.sqrt(2.0 * degree + 1.0)
    return 1.0 if degree == 0 else math.sqrt(2.0)


def _validate_points(x, dimension: int):
    if isinstance(x, torch.Tensor):
        if not torch.is_floating_point(x):
            raise TypeError("x must be floating point")
        if x.ndim != 2 or x.shape[1] != dimension:
            raise ValueError(f"x must have shape (n_samples, {dimension})")
        return x
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"x must have shape (n_samples, {dimension})")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    return values


def exact_multivariate_basis(x, multi_index, family: str = "legendre"):
    r"""Evaluate a probability-orthonormal tensor-product basis.

    For a multi-index ``alpha``, the corresponding column is

    .. math::

        \Psi_\alpha(x)=\prod_{j=1}^d \psi_{\alpha_j}(x_j).

    The input has shape ``(n_samples, d)`` and the output has shape
    ``(n_samples, len(multi_index))``.
    """
    indices = validate_multi_index(multi_index)
    family = str(family).lower()
    _, exact_basis = _family_apis(family)
    values = _validate_points(x, indices.shape[1])
    max_degrees = np.max(indices, axis=0)

    if isinstance(values, torch.Tensor):
        result = torch.ones(
            (values.shape[0], indices.shape[0]),
            dtype=values.dtype,
            device=values.device,
        )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=values.device)
        for coordinate, max_degree in enumerate(max_degrees):
            univariate = exact_basis(values[:, coordinate], int(max_degree))
            result = result * univariate[:, index_tensor[:, coordinate]]
        return result

    result = np.ones((values.shape[0], indices.shape[0]), dtype=values.dtype)
    for coordinate, max_degree in enumerate(max_degrees):
        univariate = exact_basis(values[:, coordinate], int(max_degree))
        result *= univariate[:, indices[:, coordinate]]
    return result


class MultivariateBasisEmulator(nn.Module):
    r"""Neural emulator of a tensor-product polynomial basis.

    Each coordinate polynomial uses the univariate root-factorized emulator.
    Additional fixed-weight multipliers form
    ``prod_j psi_{alpha_j}(x_j)`` for every requested multi-index.
    """

    def __init__(
        self,
        multi_index,
        family: str = "legendre",
        product: str = "relu",
        num_layers: int = 8,
        tanh_step: float | None = None,
        tanh_bias: float = OPTIMAL_TANH_BIAS,
        tanh_tolerance: float = 1e-4,
    ):
        super().__init__()
        indices = validate_multi_index(multi_index)
        family = str(family).lower()
        basis_emulator, _ = _family_apis(family)
        product = str(product).lower()
        if product not in {"relu", "tanh", "repu2"}:
            raise ValueError("product must be 'relu', 'tanh', or 'repu2'")
        num_layers = int(num_layers)
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")
        tanh_step = None if tanh_step is None else float(tanh_step)
        tanh_bias = float(tanh_bias)
        tanh_tolerance = float(tanh_tolerance)
        if tanh_step is not None and (not math.isfinite(tanh_step) or tanh_step <= 0.0):
            raise ValueError("tanh_step must be positive and finite")
        if not math.isfinite(tanh_tolerance) or tanh_tolerance <= 0.0:
            raise ValueError("tanh_tolerance must be positive and finite")

        self.family = family
        self.product = product
        self.dimension = indices.shape[1]
        self.n_features = indices.shape[0]
        self.num_layers = num_layers
        self.tanh_step = tanh_step
        self.tanh_bias = tanh_bias
        self.tanh_tolerance = tanh_tolerance
        self.register_buffer("multi_index", torch.as_tensor(indices, dtype=torch.int64))

        max_degrees = np.max(indices, axis=0)
        self.coordinate_emulators = nn.ModuleList(
            [
                basis_emulator(
                    int(max_degree),
                    product=product,
                    num_layers=num_layers,
                    tanh_step=tanh_step,
                    tanh_bias=tanh_bias,
                    tanh_tolerance=tanh_tolerance,
                )
                for max_degree in max_degrees
            ]
        )

        active_coordinates: list[tuple[int, ...]] = []
        cross_product_nets: list[nn.ModuleList] = []
        for alpha in indices:
            active = tuple(int(j) for j in np.flatnonzero(alpha))
            active_coordinates.append(active)
            bounds = [_basis_bound(family, int(alpha[j])) for j in active]
            modules: list[nn.Module] = []
            prefix_bound = bounds[0] if bounds else 1.0
            number_of_products = max(len(bounds) - 1, 1)
            for factor_number, next_bound in enumerate(bounds[1:], start=1):
                if product == "repu2":
                    modules.append(RePU2ProductNet())
                elif product == "relu":
                    radius = max(prefix_bound, next_bound)
                    modules.append(ReLUProductNet(-radius, radius, num_layers))
                else:
                    remaining_bound = math.prod(bounds[factor_number + 1 :])
                    amplification = max(
                        1.0, prefix_bound * next_bound * remaining_bound
                    )
                    local_tolerance = tanh_tolerance / (
                        number_of_products * max(1.0, remaining_bound)
                    )
                    modules.append(
                        TanhProductNet(
                            prefix_bound,
                            next_bound,
                            step=tanh_step,
                            bias=tanh_bias,
                            tolerance=local_tolerance,
                            roundoff_amplification=amplification,
                        )
                    )
                prefix_bound *= next_bound
            cross_product_nets.append(nn.ModuleList(modules))

        self.active_coordinates = tuple(active_coordinates)
        self.cross_product_nets = nn.ModuleList(cross_product_nets)

    def cross_tanh_steps(
        self, dtype: torch.dtype = torch.float64
    ) -> tuple[tuple[float, ...], ...]:
        """Return cross-coordinate tanh steps for each basis column."""
        if self.product != "tanh":
            return ()
        return tuple(
            tuple(module.effective_step(dtype) for module in modules)
            for modules in self.cross_product_nets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a floating-point torch.Tensor")
        values = _validate_points(x, self.dimension)
        coordinate_values = [
            emulator(values[:, coordinate])
            for coordinate, emulator in enumerate(self.coordinate_emulators)
        ]
        columns = []
        for feature, (active, modules) in enumerate(
            zip(self.active_coordinates, self.cross_product_nets)
        ):
            if not active:
                columns.append(torch.ones_like(values[:, 0]))
                continue
            alpha = self.multi_index[feature]
            result = coordinate_values[active[0]][:, alpha[active[0]]]
            for module, coordinate in zip(modules, active[1:]):
                result = module(
                    result, coordinate_values[coordinate][:, alpha[coordinate]]
                )
            columns.append(result)
        return torch.stack(columns, dim=-1)


def emulated_multivariate_basis(
    x,
    multi_index,
    family: str = "legendre",
    product: str = "relu",
    num_layers: int = 8,
    tanh_step: float | None = None,
    tanh_bias: float = OPTIMAL_TANH_BIAS,
    tanh_tolerance: float = 1e-4,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Evaluate a multivariate emulated basis matrix."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")
    model = MultivariateBasisEmulator(
        multi_index,
        family=family,
        product=product,
        num_layers=num_layers,
        tanh_step=tanh_step,
        tanh_bias=tanh_bias,
        tanh_tolerance=tanh_tolerance,
    ).to(dtype=x.dtype, device=x.device)
    if batch_size is None:
        return model(x)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(x) == 0:
        return model(x)
    return torch.cat(
        [
            model(x[start : start + batch_size])
            for start in range(0, len(x), batch_size)
        ],
        dim=0,
    )
