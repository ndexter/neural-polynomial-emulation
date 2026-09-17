"""Materialized feedforward realizations of the analytic multipliers."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ._root_factorized import _canonical_roots, _RootFactorizedPolynomialEmulator
from .products import ReLUProductNet, RePU2ProductNet, TanhProductNet


class RePU2Activation(nn.Module):
    """Apply the quadratic rectifier ``relu(x)**2`` elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


class FeedforwardProductNet(nn.Module):
    """A frozen standard-feedforward realization of a scalar multiplier."""

    def __init__(
        self,
        network: nn.Sequential,
        *,
        product: str,
        activation_units: int,
        resolved_tanh_step: float | None = None,
    ):
        super().__init__()
        self.network = network
        self.product = str(product)
        self.activation_units = int(activation_units)
        self.resolved_tanh_step = resolved_tanh_step
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = torch.broadcast_tensors(x, y)
        inputs = torch.stack((x, y), dim=-1)
        return self.network(inputs).squeeze(-1)

    def effective_step(self, dtype: torch.dtype) -> float:
        """Return the step frozen into a converted tanh multiplier."""
        if self.product != "tanh" or self.resolved_tanh_step is None:
            raise RuntimeError("effective_step is only defined for tanh multipliers")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        return self.resolved_tanh_step


def _linear(weight, bias, *, dtype: torch.dtype) -> nn.Linear:
    weight_tensor = torch.as_tensor(weight, dtype=dtype)
    bias_tensor = torch.as_tensor(bias, dtype=dtype)
    layer = nn.Linear(
        weight_tensor.shape[1],
        weight_tensor.shape[0],
        bias=True,
        dtype=dtype,
    )
    with torch.no_grad():
        layer.weight.copy_(weight_tensor)
        layer.bias.copy_(bias_tensor)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return layer


def _repu2_feedforward(dtype: torch.dtype) -> FeedforwardProductNet:
    network = nn.Sequential(
        _linear(
            [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [0.0, 0.0, 0.0, 0.0],
            dtype=dtype,
        ),
        RePU2Activation(),
        _linear([[0.25, 0.25, -0.25, -0.25]], [0.0], dtype=dtype),
    )
    return FeedforwardProductNet(
        network,
        product="repu2",
        activation_units=4,
    )


def _tanh_feedforward(
    source: TanhProductNet, dtype: torch.dtype
) -> FeedforwardProductNet:
    step = source.effective_step(dtype)
    x_bound = source.x_bound
    y_bound = source.y_bound
    bias = source.bias
    hidden_weight = [
        [step / (2.0 * x_bound), step / (2.0 * y_bound)],
        [0.0, 0.0],
        [-step / (2.0 * x_bound), -step / (2.0 * y_bound)],
        [step / (2.0 * x_bound), -step / (2.0 * y_bound)],
        [0.0, 0.0],
        [-step / (2.0 * x_bound), step / (2.0 * y_bound)],
    ]
    tanh_bias = math.tanh(bias)
    second_derivative = -2.0 * tanh_bias * (1.0 - tanh_bias**2)
    output_scale = x_bound * y_bound / (step**2 * second_derivative)
    output_weight = [
        [
            output_scale,
            -2.0 * output_scale,
            output_scale,
            -output_scale,
            2.0 * output_scale,
            -output_scale,
        ]
    ]
    network = nn.Sequential(
        _linear(hidden_weight, [bias] * 6, dtype=dtype),
        nn.Tanh(),
        _linear(output_weight, [0.0], dtype=dtype),
    )
    return FeedforwardProductNet(
        network,
        product="tanh",
        activation_units=6,
        resolved_tanh_step=step,
    )


def _relu_feedforward(
    source: ReLUProductNet, dtype: torch.dtype
) -> FeedforwardProductNet:
    a = source.square_x.a
    b = source.square_x.b
    num_layers = source.num_layers
    input_maps = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))

    if num_layers == 0:
        hidden_weight = [
            [(a + b) * x_weight, (a + b) * y_weight]
            for x_weight, y_weight in input_maps
        ]
        network = nn.Sequential(
            _linear(hidden_weight, [-a * b] * 3, dtype=dtype),
            nn.ReLU(),
            _linear([[-0.5, -0.5, 2.0]], [0.0], dtype=dtype),
        )
        return FeedforwardProductNet(
            network,
            product="relu",
            activation_units=3,
        )

    midpoint = 0.5 * (a + b)
    hidden_weight = []
    hidden_bias = []
    for x_weight, y_weight in input_maps:
        hidden_weight.append([(a + b) * x_weight, (a + b) * y_weight])
        hidden_bias.append(-a * b)
        for threshold in (a, midpoint, b):
            hidden_weight.append([x_weight, y_weight])
            hidden_bias.append(-threshold)

    modules: list[nn.Module] = [
        _linear(hidden_weight, hidden_bias, dtype=dtype),
        nn.ReLU(),
    ]
    scale = (b - a) ** 2 / 4.0
    first_g_coefficients = (2.0 / (b - a), -4.0 / (b - a), 2.0 / (b - a))
    later_g_coefficients = (2.0, -4.0, 2.0)

    for hidden_layer in range(2, num_layers + 1):
        g_coefficients = (
            first_g_coefficients if hidden_layer == 2 else later_g_coefficients
        )
        correction = scale / (4.0 ** (hidden_layer - 2))
        block_weight = [
            [1.0, *(-correction * value for value in g_coefficients)],
            [0.0, *g_coefficients],
            [0.0, *g_coefficients],
            [0.0, *g_coefficients],
        ]
        block_bias = [0.0, 0.0, -0.5, -1.0]
        weight = torch.zeros((12, 12), dtype=dtype)
        for branch in range(3):
            block = torch.as_tensor(block_weight, dtype=dtype)
            start = 4 * branch
            weight[start : start + 4, start : start + 4] = block
        modules.extend(
            [
                _linear(weight, block_bias * 3, dtype=dtype),
                nn.ReLU(),
            ]
        )

    g_coefficients = first_g_coefficients if num_layers == 1 else later_g_coefficients
    correction = scale / (4.0 ** (num_layers - 1))
    square_output = torch.as_tensor(
        [1.0, *(-correction * value for value in g_coefficients)],
        dtype=dtype,
    )
    output_weight = torch.zeros((1, 12), dtype=dtype)
    for branch, polarization in enumerate((-0.5, -0.5, 2.0)):
        start = 4 * branch
        output_weight[0, start : start + 4] = polarization * square_output
    modules.append(_linear(output_weight, [0.0], dtype=dtype))
    return FeedforwardProductNet(
        nn.Sequential(*modules),
        product="relu",
        activation_units=12 * num_layers,
    )


def _convert_product(
    module: nn.Module, dtype: torch.dtype
) -> FeedforwardProductNet | None:
    if isinstance(module, ReLUProductNet):
        return _relu_feedforward(module, dtype)
    if isinstance(module, TanhProductNet):
        return _tanh_feedforward(module, dtype)
    if isinstance(module, RePU2ProductNet):
        return _repu2_feedforward(dtype)
    return None


def _first_device(module: nn.Module) -> torch.device:
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return torch.device("cpu")


def to_feedforward(
    module: nn.Module,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Return a converted copy with explicit frozen feedforward multipliers.

    The source module is not mutated. Root shifts, polynomial scaling, product
    trees, and basis assembly retain their existing functional wiring; every
    ReLU, tanh, or RePU-2 multiplier is replaced by an equivalent module made
    from fixed ``nn.Linear`` layers and elementwise activations.

    Automatic tanh steps are resolved for ``dtype`` at conversion time.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    target_device = _first_device(module) if device is None else torch.device(device)

    direct_conversion = _convert_product(module, dtype)
    if direct_conversion is not None:
        return direct_conversion.to(device=target_device, dtype=dtype)

    converted = copy.deepcopy(module)

    def replace_products(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            replacement = _convert_product(child, dtype)
            if replacement is None:
                replace_products(child)
            else:
                parent._modules[name] = replacement

    replace_products(converted)

    for child in list(converted.modules()):
        if not isinstance(child, _RootFactorizedPolynomialEmulator):
            continue
        if child.degree == 0:
            child.feedforward_constant_layer = _linear([[0.0]], [1.0], dtype=dtype)
            continue
        roots = torch.as_tensor(
            _canonical_roots(child.roots.detach().cpu().numpy()),
            dtype=dtype,
        )
        child.feedforward_factor_layer = _linear(
            torch.ones((child.degree, 1), dtype=dtype),
            -roots,
            dtype=dtype,
        )
        child.feedforward_output_layer = _linear(
            [[float(child.leading_coefficient.item())]],
            [0.0],
            dtype=dtype,
        )
    return converted.to(device=target_device, dtype=dtype)
