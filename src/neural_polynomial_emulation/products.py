"""Constructive neural multiplication modules."""

from __future__ import annotations

import math

import torch
from torch import nn

from .squaring import OPTIMAL_TANH_BIAS, ReLUSquaringNet, TanhSquaringNet

_TANH_TRUNCATION_CONSTANT = 0.2
_MIN_TANH_STEP = 1e-4
_MAX_TANH_STEP = 0.5


class ReLUProductNet(nn.Module):
    """Approximate multiplication on ``[a, b] x [a, b]`` using ReLU squares."""

    def __init__(self, a: float = -1.0, b: float = 1.0, num_layers: int = 8):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.num_layers = int(num_layers)
        self.square_x = ReLUSquaringNet(a, b, num_layers)
        self.square_y = ReLUSquaringNet(a, b, num_layers)
        self.square_midpoint = ReLUSquaringNet(a, b, num_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (x + y)
        return 0.5 * (
            4.0 * self.square_midpoint(midpoint) - self.square_x(x) - self.square_y(y)
        )


class RePU2ProductNet(nn.Module):
    """Implement multiplication exactly using ``sigma_2(z)=relu(z)**2``."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.25 * (
            torch.relu(x + y).square()
            + torch.relu(-x - y).square()
            - torch.relu(x - y).square()
            - torch.relu(-x + y).square()
        )


class TanhProductNet(nn.Module):
    r"""Approximate multiplication with the six-neuron tanh construction.

    ``x_bound`` and ``y_bound`` describe the intended input box. Inputs are
    normalized to ``[-1, 1]`` before applying the polarization identity

    .. math:: xy = ((x+y)^2-(x-y)^2)/4.

    Each square is a three-neuron centered finite difference, so the resulting
    shallow multiplier has six tanh neurons and fixed analytic weights.
    """

    def __init__(
        self,
        x_bound: float = 1.0,
        y_bound: float | None = None,
        step: float | None = None,
        bias: float = OPTIMAL_TANH_BIAS,
        tolerance: float = 1e-4,
        roundoff_amplification: float = 1.0,
    ):
        super().__init__()
        self.x_bound = float(x_bound)
        self.y_bound = self.x_bound if y_bound is None else float(y_bound)
        if not math.isfinite(self.x_bound) or self.x_bound <= 0.0:
            raise ValueError("x_bound must be positive and finite")
        if not math.isfinite(self.y_bound) or self.y_bound <= 0.0:
            raise ValueError("y_bound must be positive and finite")
        self.step = None if step is None else float(step)
        self.bias = float(bias)
        self.tolerance = float(tolerance)
        self.roundoff_amplification = float(roundoff_amplification)
        if self.step is not None and (not math.isfinite(self.step) or self.step <= 0.0):
            raise ValueError("step must be positive and finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        if (
            not math.isfinite(self.roundoff_amplification)
            or self.roundoff_amplification <= 0.0
        ):
            raise ValueError("roundoff_amplification must be positive and finite")
        nominal_step = 0.1 if self.step is None else self.step
        self.square_sum = TanhSquaringNet(2.0, nominal_step, self.bias)
        self.square_difference = TanhSquaringNet(2.0, nominal_step, self.bias)

    def effective_step(self, dtype: torch.dtype) -> float:
        """Return the explicit or tolerance/dtype-selected finite-difference step.

        Automatic selection balances the fourth-order truncation error at the
        optimized bias against a sixth-root model of cancellation error.
        """
        if self.step is not None:
            return self.step
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        truncation_step = (self.tolerance / _TANH_TRUNCATION_CONSTANT) ** 0.25
        roundoff_step = (torch.finfo(dtype).eps * self.roundoff_amplification) ** (
            1.0 / 6.0
        )
        return min(
            _MAX_TANH_STEP,
            max(_MIN_TANH_STEP, truncation_step, roundoff_step),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x) or not torch.is_floating_point(y):
            raise TypeError("x and y must be floating-point torch.Tensors")
        step = self.effective_step(torch.promote_types(x.dtype, y.dtype))
        x_normalized = x / self.x_bound
        y_normalized = y / self.y_bound
        normalized_product = 0.25 * (
            self.square_sum.forward_with_step(x_normalized + y_normalized, step)
            - self.square_difference.forward_with_step(
                x_normalized - y_normalized, step
            )
        )
        return (self.x_bound * self.y_bound) * normalized_product
