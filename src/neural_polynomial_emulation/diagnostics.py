"""Numerical diagnostics intrinsic to polynomial emulation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ._root_factorized import _canonical_roots, _RootFactorizedPolynomialEmulator
from .basis import emulated_legendre_basis, exact_legendre_basis
from .chebyshev_basis import emulated_chebyshev_basis, exact_chebyshev_basis
from .feedforward import FeedforwardProductNet
from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet
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


def emulator_complexity(module: nn.Module) -> dict[str, int]:
    """Count multiplication nodes and canonical feedforward realizations.

    ``activation_units_per_sample`` counts the scalar nonlinear units in the
    standard feedforward realization of every multiplier. A depth-``L`` ReLU
    multiplier has ``L`` hidden layers of width 12 when ``L >= 1`` (and one
    width-three layer when ``L == 0``); tanh and RePU-2 multipliers have one
    nonlinear layer of width six and four, respectively. The tanh realization
    uses affine widths ``2 -> 2 -> 6 -> 2 -> 1`` so that finite-difference
    cancellation follows the functional construction's numerical grouping.

    ``total_parameters`` counts all entries, including biases, in the dense
    affine layers of those realizations. ``nonzero_parameters`` counts only
    the entries used by their sparse analytic constructions. Counts are added
    over multiplier modules. Modular affine layers for root shifts, output
    scaling, and degree-zero constants are included. Artificial zero padding
    between independent branches is excluded. The implementation's registered
    buffers are reported separately.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    relu_products = [
        child for child in module.modules() if isinstance(child, ReLUProductNet)
    ]
    tanh_products = [
        child for child in module.modules() if isinstance(child, TanhProductNet)
    ]
    repu2_products = [
        child for child in module.modules() if isinstance(child, RePU2ProductNet)
    ]
    feedforward_products = [
        child for child in module.modules() if isinstance(child, FeedforwardProductNet)
    ]
    root_emulators = [
        child
        for child in module.modules()
        if isinstance(child, _RootFactorizedPolynomialEmulator)
    ]
    relu_units = sum(
        12 * child.num_layers if child.num_layers >= 1 else 3 for child in relu_products
    )
    tanh_units = 6 * len(tanh_products)
    repu2_units = 4 * len(repu2_products)

    relu_total_parameters = 0
    relu_nonzero_parameters = 0
    for child in relu_products:
        layers = child.num_layers
        if layers == 0:
            relu_total_parameters += 13  # 2 -> 3 -> 1
        else:
            relu_total_parameters += 36 + 156 * (layers - 1) + 13

        a = child.square_x.a
        b = child.square_x.b
        midpoint = 0.5 * (a + b)
        input_maps = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
        first_layer_nonzeros = 0
        for x_weight, y_weight in input_maps:
            affine_values = (
                (a + b) * x_weight,
                (a + b) * y_weight,
                -a * b,
            )
            first_layer_nonzeros += sum(value != 0.0 for value in affine_values)
            if layers >= 1:
                for threshold in (a, midpoint, b):
                    hinge_values = (x_weight, y_weight, -threshold)
                    first_layer_nonzeros += sum(value != 0.0 for value in hinge_values)
        if layers == 0:
            relu_nonzero_parameters += first_layer_nonzeros + 3
        else:
            relu_nonzero_parameters += first_layer_nonzeros + 45 * (layers - 1) + 12

    total_parameters = (
        relu_total_parameters
        + 41 * len(tanh_products)  # 2 -> 2 -> 6 -> 2 -> 1
        + 17 * len(repu2_products)  # 2 -> 4 -> 1
    )
    nonzero_parameters = (
        relu_nonzero_parameters + 22 * len(tanh_products) + 12 * len(repu2_products)
    )
    for child in root_emulators:
        if child.degree == 0:
            total_parameters += 2  # 1 -> 1 constant affine layer
            nonzero_parameters += 1
        else:
            total_parameters += 2 * child.degree + 2
            roots = _canonical_roots(child.roots.detach().cpu().numpy())
            nonzero_parameters += child.degree + int(np.count_nonzero(roots)) + 1
    for child in feedforward_products:
        child_parameters = list(child.parameters())
        total_parameters += sum(parameter.numel() for parameter in child_parameters)
        nonzero_parameters += sum(
            int(torch.count_nonzero(parameter.detach()).item())
            for parameter in child_parameters
        )
        if child.product == "relu":
            relu_units += child.activation_units
        elif child.product == "tanh":
            tanh_units += child.activation_units
        elif child.product == "repu2":
            repu2_units += child.activation_units
    registered_parameters = list(module.parameters())
    return {
        "product_nodes": (
            len(relu_products)
            + len(tanh_products)
            + len(repu2_products)
            + len(feedforward_products)
        ),
        "relu_product_nodes": len(relu_products)
        + sum(child.product == "relu" for child in feedforward_products),
        "tanh_product_nodes": len(tanh_products)
        + sum(child.product == "tanh" for child in feedforward_products),
        "repu2_product_nodes": len(repu2_products)
        + sum(child.product == "repu2" for child in feedforward_products),
        "activation_units_per_sample": relu_units + tanh_units + repu2_units,
        "total_parameters": total_parameters,
        "nonzero_parameters": nonzero_parameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in registered_parameters
            if parameter.requires_grad
        ),
        "buffer_scalars": sum(buffer.numel() for buffer in module.buffers()),
    }
