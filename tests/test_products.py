import torch

from neural_polynomial_emulation import (
    ReLUProductNet,
    RePU2ProductNet,
    TanhProductNet,
)


def test_repu2_product_is_exact_to_roundoff():
    generator = torch.Generator().manual_seed(7)
    x = 4.0 * torch.rand(2048, generator=generator, dtype=torch.float64) - 2.0
    y = 4.0 * torch.rand(2048, generator=generator, dtype=torch.float64) - 2.0
    torch.testing.assert_close(RePU2ProductNet()(x, y), x * y, rtol=2e-15, atol=2e-15)


def test_relu_product_improves_with_depth():
    x = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
    y = torch.flip(x, dims=(0,)) * 0.73
    shallow = torch.max(torch.abs(ReLUProductNet(-1.0, 1.0, 2)(x, y) - x * y))
    deep = torch.max(torch.abs(ReLUProductNet(-1.0, 1.0, 8)(x, y) - x * y))
    assert deep < shallow


def test_product_broadcasting():
    x = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    out = RePU2ProductNet()(x, y)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, x * y)


def test_tanh_product_improves_as_step_decreases_in_float64():
    x = torch.linspace(-2.0, 2.0, 2049, dtype=torch.float64)
    y = 3.0 * torch.sin(x)
    errors = [
        torch.max(torch.abs(TanhProductNet(2.0, 3.0, step=step)(x, y) - x * y)).item()
        for step in (0.4, 0.2, 0.1)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.03


def test_tanh_product_broadcasting():
    x = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    assert TanhProductNet()(x, y).shape == (2, 3)


def test_automatic_tanh_step_uses_tolerance_and_dtype():
    loose = TanhProductNet(step=None, tolerance=1e-4)
    tight = TanhProductNet(step=None, tolerance=1e-8)
    assert tight.effective_step(torch.float64) < loose.effective_step(torch.float64)
    assert tight.effective_step(torch.float32) > tight.effective_step(torch.float64)


def test_explicit_tanh_step_overrides_automatic_selection():
    net = TanhProductNet(step=0.037, tolerance=1e-12)
    assert net.effective_step(torch.float32) == 0.037
    assert net.effective_step(torch.float64) == 0.037
