"""Constructive neural approximations of the squaring map."""

from __future__ import annotations

import math

import torch
from torch import nn

OPTIMAL_TANH_BIAS = math.atanh(math.sqrt(2.0 / 3.0))


class ReLUSquaringNet(nn.Module):
    """Approximate ``x -> x**2`` on a compact interval.

    The construction is the iterated sawtooth approximation. On ``[a, b]``
    its uniform error is bounded by

    ``((b-a)**2 / 4) * 4**(-num_layers)``.

    The module has no trainable parameters and preserves the input tensor's
    shape, dtype, and device.
    """

    def __init__(self, a: float = -1.0, b: float = 1.0, num_layers: int = 8):
        super().__init__()
        a = float(a)
        b = float(b)
        num_layers = int(num_layers)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("a and b must be finite")
        if not a < b:
            raise ValueError("need a < b")
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")
        self.a = a
        self.b = b
        self.num_layers = num_layers
        self._scale = (b - a) ** 2 / 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point torch.Tensor")
        a, b = self.a, self.b
        f = torch.relu((a + b) * x - a * b)
        if self.num_layers == 0:
            return f

        g = (
            (2.0 / (b - a)) * torch.relu(x - a)
            - (4.0 / (b - a)) * torch.relu(x - (a + b) / 2.0)
            + (2.0 / (b - a)) * torch.relu(x - b)
        )
        f = f - self._scale * g
        for layer in range(2, self.num_layers + 1):
            g = (
                2.0 * torch.relu(g)
                - 4.0 * torch.relu(g - 0.5)
                + 2.0 * torch.relu(g - 1.0)
            )
            f = f - (self._scale / (4.0 ** (layer - 1))) * g
        return f

    def error_bound(self) -> float:
        """Return the analytic uniform error bound on ``[a, b]``."""
        return self._scale / (4.0**self.num_layers)


class TanhSquaringNet(nn.Module):
    r"""Approximate ``x -> x**2`` by a centered tanh finite difference.

    For ``|x| <= bound``, the construction applies

    .. math::

        q_h(z) = \frac{\tanh(b+hz)-2\tanh(b)+\tanh(b-hz)}
                       {h^2\tanh''(b)}

    to ``z = x / bound`` and returns ``bound**2 * q_h(z)``. This is the
    three-neuron square used in the tanh multiplication construction of
    De Ryck, Lanthaler, and Mishra (2021, Lemma 3.8).

    The default bias satisfies ``tanh''''(bias) = 0``, canceling the leading
    finite-difference defect and giving fourth-order rather than second-order
    truncation error. Smaller ``step`` increases the output weights like
    ``step**-2`` and eventually amplifies floating-point cancellation.
    """

    def __init__(
        self,
        bound: float = 1.0,
        step: float = 0.1,
        bias: float = OPTIMAL_TANH_BIAS,
    ):
        super().__init__()
        bound = float(bound)
        step = float(step)
        bias = float(bias)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("bound must be positive and finite")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step must be positive and finite")
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        tanh_bias = math.tanh(bias)
        second_derivative = -2.0 * tanh_bias * (1.0 - tanh_bias**2)
        if second_derivative == 0.0:
            raise ValueError("tanh must have nonzero second derivative at bias")

        self.bound = bound
        self.step = step
        self.bias = bias
        self._tanh_bias = tanh_bias
        self._second_derivative = second_derivative

    def forward_with_step(self, x: torch.Tensor, step: float) -> torch.Tensor:
        """Evaluate the same fixed-weight construction with a supplied step."""
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point torch.Tensor")
        step = float(step)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step must be positive and finite")
        z = x / self.bound
        offset = step * z
        numerator = (
            torch.tanh(self.bias + offset)
            - 2.0 * self._tanh_bias
            + torch.tanh(self.bias - offset)
        )
        return (self.bound**2) * numerator / (step**2 * self._second_derivative)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_step(x, self.step)
