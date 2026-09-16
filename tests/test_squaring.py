import math

import pytest
import torch

from neural_polynomial_emulation import (
    OPTIMAL_TANH_BIAS,
    ReLUSquaringNet,
    TanhSquaringNet,
)


@pytest.mark.parametrize("num_layers", [1, 2, 4, 6, 8])
def test_squaring_respects_uniform_error_bound(num_layers):
    x = torch.linspace(-1.0, 1.0, 16385, dtype=torch.float64)
    net = ReLUSquaringNet(-1.0, 1.0, num_layers=num_layers)
    error = torch.max(torch.abs(net(x) - x.square())).item()
    assert error <= net.error_bound() * (1.0 + 1e-11) + 1e-14


def test_squaring_error_decreases_with_depth():
    x = torch.linspace(-1.0, 1.0, 8193, dtype=torch.float64)
    errors = [
        torch.max(torch.abs(ReLUSquaringNet(-1.0, 1.0, layers)(x) - x.square())).item()
        for layers in (2, 4, 6)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_squaring_validates_constructor_and_input():
    with pytest.raises(ValueError):
        ReLUSquaringNet(1.0, -1.0)
    with pytest.raises(ValueError):
        ReLUSquaringNet(num_layers=-1)
    with pytest.raises(TypeError):
        ReLUSquaringNet()(torch.arange(3))


def test_import_and_construction_do_not_change_default_dtype():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        ReLUSquaringNet()
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(old)


def test_tanh_squaring_improves_as_step_decreases_in_float64():
    x = torch.linspace(-1.0, 1.0, 4097, dtype=torch.float64)
    errors = [
        torch.max(torch.abs(TanhSquaringNet(step=step)(x) - x.square())).item()
        for step in (0.4, 0.2, 0.1)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.005


def test_optimized_tanh_bias_cancels_the_second_order_error_term():
    assert math.isclose(OPTIMAL_TANH_BIAS, math.atanh(math.sqrt(2.0 / 3.0)))
    x = torch.linspace(-1.0, 1.0, 4097, dtype=torch.float64)
    errors = [
        torch.max(torch.abs(TanhSquaringNet(step=step)(x) - x.square())).item()
        for step in (0.4, 0.2, 0.1)
    ]
    assert errors[0] / errors[1] > 12.0
    assert errors[1] / errors[2] > 12.0


def test_tanh_squaring_preserves_shape_dtype_and_gradients():
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64, requires_grad=True)
    out = TanhSquaringNet(bound=2.0, step=0.1)(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    out.sum().backward()
    assert x.grad is not None


def test_tanh_squaring_validates_constructor():
    with pytest.raises(ValueError):
        TanhSquaringNet(bound=0.0)
    with pytest.raises(ValueError):
        TanhSquaringNet(step=0.0)
    with pytest.raises(ValueError):
        TanhSquaringNet(bias=0.0)
