"""Exact and neural bases for the Chebyshev probability measure."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .basis import _squeeze_coordinate_axis
from .chebyshev import ChebyshevEmulator, exact_chebyshev
from .squaring import OPTIMAL_TANH_BIAS


def exact_chebyshev_basis(x, max_degree: int):
    """Evaluate the normalized Chebyshev sequence through ``max_degree``."""
    max_degree = int(max_degree)
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    values = _squeeze_coordinate_axis(x)
    columns = [exact_chebyshev(values, degree) for degree in range(max_degree + 1)]
    if isinstance(values, torch.Tensor):
        return torch.stack(columns, dim=-1)
    return np.stack(columns, axis=-1)


class ChebyshevBasisEmulator(nn.Module):
    """Evaluate root-factorized neural Chebyshev polynomials through a degree."""

    def __init__(
        self,
        max_degree: int,
        product: str = "relu",
        num_layers: int = 8,
        tanh_step: float | None = None,
        tanh_bias: float = OPTIMAL_TANH_BIAS,
        tanh_tolerance: float = 1e-4,
    ):
        super().__init__()
        max_degree = int(max_degree)
        if max_degree < 0:
            raise ValueError("max_degree must be nonnegative")
        self.max_degree = max_degree
        self.product = str(product).lower()
        self.num_layers = int(num_layers)
        self.tanh_step = None if tanh_step is None else float(tanh_step)
        self.tanh_bias = float(tanh_bias)
        self.tanh_tolerance = float(tanh_tolerance)
        self.emulators = nn.ModuleList(
            [
                ChebyshevEmulator(
                    degree,
                    product=self.product,
                    num_layers=self.num_layers,
                    tanh_step=self.tanh_step,
                    tanh_bias=self.tanh_bias,
                    tanh_tolerance=self.tanh_tolerance,
                )
                for degree in range(max_degree + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = _squeeze_coordinate_axis(x)
        return torch.stack([emulator(values) for emulator in self.emulators], dim=-1)


def emulated_chebyshev_basis(
    x,
    max_degree: int,
    product: str = "relu",
    num_layers: int = 8,
    tanh_step: float | None = None,
    tanh_bias: float = OPTIMAL_TANH_BIAS,
    tanh_tolerance: float = 1e-4,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Evaluate the normalized Chebyshev sequence with neural multipliers."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")
    model = ChebyshevBasisEmulator(
        max_degree,
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
    if x.ndim == 0:
        return model(x)
    return torch.cat(
        [
            model(x[start : start + batch_size])
            for start in range(0, len(x), batch_size)
        ],
        dim=0,
    )
