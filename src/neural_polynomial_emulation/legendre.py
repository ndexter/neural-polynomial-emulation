"""Exact and neural evaluation of probability-normalized Legendre polynomials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import overload

import numpy as np
import torch
from torch import nn

from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet
from .squaring import OPTIMAL_TANH_BIAS


@dataclass
class _FactorGroup:
    indices: tuple[int, ...]
    bound: float
    left: _FactorGroup | None = None
    right: _FactorGroup | None = None
    module_index: int | None = None
    sensitivity: float = 1.0


def _factor_group_bound(roots: np.ndarray) -> float:
    """Compute a safely inflated sup norm of a root product on ``[-1, 1]``."""
    if roots.size == 0:
        return 1.0
    polynomial = np.polynomial.Polynomial.fromroots(roots)
    critical = polynomial.deriv().roots()
    real_critical = [
        float(root.real)
        for root in critical
        if abs(root.imag) <= 1e-10 and -1.0 <= root.real <= 1.0
    ]
    points = np.asarray([-1.0, 1.0, *real_critical], dtype=np.float64)
    bound = float(np.max(np.abs(polynomial(points))))
    inflation = 128.0 * np.finfo(np.float64).eps * (roots.size + 1)
    return bound * (1.0 + inflation) + np.finfo(np.float64).tiny


def _validate_degree(degree: int) -> int:
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    return degree


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


class LegendreEmulator(nn.Module):
    """Root-factorized neural emulator of ``sqrt(2n+1) P_n``.

    Parameters
    ----------
    degree:
        Polynomial degree.
    product:
        ``"relu"`` or ``"tanh"`` for approximate multiplication, or
        ``"repu2"`` for the exact RePU-2 identity.
    num_layers:
        Depth parameter of every ReLU squaring module. It is retained for a
        uniform API for the other product types.
    tanh_step:
        Optional finite-difference step for the tanh construction. ``None``
        chooses a step separately at every tree node from ``tanh_tolerance``
        and the input dtype. A positive number overrides automatic selection.
    tanh_bias:
        Expansion point for the tanh finite difference. It must have nonzero
        second derivative.
    tanh_tolerance:
        Requested global absolute accuracy used by automatic step selection.
        This is a numerical allocation heuristic, not a certified error bound.

    Notes
    -----
    The roots and leading coefficient are registered as buffers. The module
    therefore follows ordinary PyTorch dtype/device conversions and has no
    trainable parameters.
    """

    def __init__(
        self,
        degree: int,
        product: str = "relu",
        num_layers: int = 8,
        tanh_step: float | None = None,
        tanh_bias: float = OPTIMAL_TANH_BIAS,
        tanh_tolerance: float = 1e-4,
    ):
        super().__init__()
        degree = _validate_degree(degree)
        num_layers = int(num_layers)
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")
        product = str(product).lower()
        if product not in {"relu", "repu2", "tanh"}:
            raise ValueError("product must be 'relu', 'repu2', or 'tanh'")
        tanh_step = None if tanh_step is None else float(tanh_step)
        tanh_bias = float(tanh_bias)
        tanh_tolerance = float(tanh_tolerance)
        if product == "tanh":
            if tanh_step is not None and (
                not math.isfinite(tanh_step) or tanh_step <= 0.0
            ):
                raise ValueError("tanh_step must be positive and finite")
            if not math.isfinite(tanh_bias):
                raise ValueError("tanh_bias must be finite")
            tanh_value = math.tanh(tanh_bias)
            tanh_second_derivative = -2.0 * tanh_value * (1.0 - tanh_value**2)
            if tanh_second_derivative == 0.0:
                raise ValueError(
                    "tanh must have nonzero second derivative at tanh_bias"
                )
            if not math.isfinite(tanh_tolerance) or tanh_tolerance <= 0.0:
                raise ValueError("tanh_tolerance must be positive and finite")

        self.degree = degree
        self.product = product
        self.num_layers = num_layers
        self.register_buffer(
            "roots", torch.as_tensor(legendre_roots(degree), dtype=torch.float64)
        )
        self.tanh_step = tanh_step
        self.tanh_bias = tanh_bias
        self.tanh_tolerance = tanh_tolerance
        self.register_buffer(
            "leading_coefficient",
            torch.tensor(legendre_leading_coefficient(degree), dtype=torch.float64),
        )

        modules: list[nn.Module] = []
        self._tanh_root: _FactorGroup | None = None
        self._tanh_level_module_indices: tuple[tuple[int, ...], ...] = ()
        self.tanh_group_indices: tuple[tuple[tuple[int, ...], ...], ...] = ()
        self.tanh_group_bounds: tuple[tuple[float, ...], ...] = ()
        if product == "tanh":
            self._build_tanh_tree(modules)
        else:
            for stage in range(max(degree - 1, 0)):
                if product == "repu2":
                    modules.append(RePU2ProductNet())
                else:
                    radius = 2.0 ** (stage + 1)
                    modules.append(ReLUProductNet(-radius, radius, num_layers))
        self.product_nets = nn.ModuleList(modules)

    def _build_tanh_tree(self, modules: list[nn.Module]) -> None:
        if self.degree == 0:
            return
        roots = self.roots.detach().cpu().numpy()
        cache: dict[tuple[int, ...], float] = {}

        def group_bound(indices: tuple[int, ...]) -> float:
            key = tuple(sorted(indices))
            if key not in cache:
                cache[key] = _factor_group_bound(roots[np.asarray(key)])
            return cache[key]

        groups = [
            _FactorGroup((index,), group_bound((index,)))
            for index in range(self.degree)
        ]
        module_nodes: list[_FactorGroup] = []
        level_modules: list[tuple[int, ...]] = []
        level_indices: list[tuple[tuple[int, ...], ...]] = []
        level_bounds: list[tuple[float, ...]] = []
        while len(groups) > 1:
            remaining = list(groups)
            next_groups: list[_FactorGroup] = []
            this_level_modules: list[int] = []
            while len(remaining) >= 2:
                candidates = []
                for left_index in range(len(remaining) - 1):
                    for right_index in range(left_index + 1, len(remaining)):
                        union = tuple(
                            sorted(
                                remaining[left_index].indices
                                + remaining[right_index].indices
                            )
                        )
                        candidates.append(
                            (group_bound(union), union, left_index, right_index)
                        )
                bound, union, left_index, right_index = min(candidates)
                right = remaining.pop(right_index)
                left = remaining.pop(left_index)
                node = _FactorGroup(
                    union,
                    bound,
                    left=left,
                    right=right,
                    module_index=len(module_nodes),
                )
                module_nodes.append(node)
                this_level_modules.append(node.module_index)
                next_groups.append(node)
            next_groups.extend(remaining)
            groups = sorted(next_groups, key=lambda group: group.indices)
            level_modules.append(tuple(this_level_modules))
            level_indices.append(tuple(group.indices for group in groups))
            level_bounds.append(tuple(group.bound for group in groups))

        self._tanh_root = groups[0]
        self._tanh_level_module_indices = tuple(level_modules)
        self.tanh_group_indices = tuple(level_indices)
        self.tanh_group_bounds = tuple(level_bounds)

        leading = float(self.leading_coefficient.item())

        def allocate(node: _FactorGroup, sensitivity: float) -> None:
            node.sensitivity = sensitivity
            if node.left is None or node.right is None:
                return
            allocate(node.left, sensitivity * node.right.bound)
            allocate(node.right, sensitivity * node.left.bound)

        allocate(self._tanh_root, leading)
        number_of_products = max(self.degree - 1, 1)
        for node in module_nodes:
            assert node.left is not None and node.right is not None
            amplification = max(
                1.0, node.sensitivity * node.left.bound * node.right.bound
            )
            local_tolerance = self.tanh_tolerance / (number_of_products * amplification)
            modules.append(
                TanhProductNet(
                    node.left.bound,
                    node.right.bound,
                    step=self.tanh_step,
                    bias=self.tanh_bias,
                    tolerance=local_tolerance,
                    roundoff_amplification=amplification,
                )
            )

    def tanh_steps(
        self, dtype: torch.dtype = torch.float64
    ) -> tuple[tuple[float, ...], ...]:
        """Return effective multiplier steps, grouped by product-tree level."""
        if self.product != "tanh":
            return ()
        return tuple(
            tuple(self.product_nets[index].effective_step(dtype) for index in level)
            for level in self._tanh_level_module_indices
        )

    def _evaluate_tanh_group(
        self, node: _FactorGroup, factors: list[torch.Tensor]
    ) -> torch.Tensor:
        if node.left is None or node.right is None:
            return factors[node.indices[0]]
        assert node.module_index is not None
        return self.product_nets[node.module_index](
            self._evaluate_tanh_group(node.left, factors),
            self._evaluate_tanh_group(node.right, factors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point torch.Tensor")
        if self.degree == 0:
            return torch.ones_like(x)

        roots = self.roots.to(dtype=x.dtype, device=x.device)
        leading = self.leading_coefficient.to(dtype=x.dtype, device=x.device)
        factors = [x - root for root in roots.unbind()]
        if self.degree == 1:
            return leading * factors[0]

        if self.product == "tanh":
            assert self._tanh_root is not None
            return leading * self._evaluate_tanh_group(self._tanh_root, factors)

        value = self.product_nets[0](factors[0], factors[1])
        for stage in range(1, self.degree - 1):
            value = self.product_nets[stage](value, factors[stage + 1])
        return leading * value
