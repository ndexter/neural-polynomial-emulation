import numpy as np
import pytest
import torch

from neural_polynomial_emulation import LegendreEmulator, exact_legendre


def test_degrees_zero_and_one_are_exact_under_probability_measure():
    x = torch.linspace(-1.0, 1.0, 31, dtype=torch.float64)
    torch.testing.assert_close(LegendreEmulator(0)(x), torch.ones_like(x))
    torch.testing.assert_close(LegendreEmulator(1)(x), np.sqrt(3.0) * x)


@pytest.mark.parametrize("degree", range(2, 9))
def test_repu2_matches_exact_legendre_at_moderate_degree(degree):
    x = torch.linspace(-0.95, 0.95, 257, dtype=torch.float64)
    predicted = LegendreEmulator(degree, product="repu2")(x)
    exact = exact_legendre(x, degree)
    torch.testing.assert_close(predicted, exact, rtol=2e-10, atol=2e-10)


def test_relu_emulation_improves_with_depth_at_fixed_degree():
    x = torch.linspace(-1.0, 1.0, 2049, dtype=torch.float64)
    exact = exact_legendre(x, 4)
    shallow = torch.max(
        torch.abs(LegendreEmulator(4, product="relu", num_layers=4)(x) - exact)
    )
    deep = torch.max(
        torch.abs(LegendreEmulator(4, product="relu", num_layers=12)(x) - exact)
    )
    assert deep < shallow


def test_tanh_emulation_uses_balanced_tree_and_improves_with_step():
    x = torch.linspace(-1.0, 1.0, 2049, dtype=torch.float64)
    exact = exact_legendre(x, 6)
    errors = [
        torch.max(
            torch.abs(LegendreEmulator(6, product="tanh", tanh_step=step)(x) - exact)
        ).item()
        for step in (0.2, 0.1, 0.05)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert len(LegendreEmulator(6, product="tanh").product_nets) == 5


def test_tanh_tree_pairs_symmetric_roots_and_uses_tight_bounds():
    model = LegendreEmulator(8, product="tanh")
    assert model.tanh_group_indices[0] == (
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
    )
    assert max(model.tanh_group_bounds[0]) < 1.0


def test_automatic_tanh_steps_are_per_level_and_dtype_aware():
    model = LegendreEmulator(8, product="tanh", tanh_tolerance=1e-6)
    float64_steps = model.tanh_steps(torch.float64)
    float32_steps = model.tanh_steps(torch.float32)
    assert tuple(map(len, float64_steps)) == (4, 2, 1)
    assert len({round(step, 12) for level in float64_steps for step in level}) > 1
    assert all(
        step32 >= step64
        for level32, level64 in zip(float32_steps, float64_steps)
        for step32, step64 in zip(level32, level64)
    )


def test_automatic_tanh_emulation_is_accurate_at_higher_degree():
    x = torch.linspace(-1.0, 1.0, 4097, dtype=torch.float64)
    model = LegendreEmulator(24, product="tanh", tanh_tolerance=1e-4)
    error = torch.max(torch.abs(model(x) - exact_legendre(x, 24))).item()
    assert error < 1e-4


def test_buffers_follow_dtype_conversion_and_serialize():
    model = LegendreEmulator(5, product="repu2").float()
    assert model.roots.dtype == torch.float32
    assert model.leading_coefficient.dtype == torch.float32
    assert {"roots", "leading_coefficient"}.issubset(model.state_dict())
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float32)
    assert model(x).dtype == torch.float32


def test_constructor_validation():
    with pytest.raises(ValueError):
        LegendreEmulator(-1)
    with pytest.raises(ValueError):
        LegendreEmulator(2, product="unknown")
    with pytest.raises(ValueError):
        LegendreEmulator(2, num_layers=-1)
    with pytest.raises(ValueError):
        LegendreEmulator(2, product="tanh", tanh_step=0.0)
    with pytest.raises(ValueError):
        LegendreEmulator(2, product="tanh", tanh_bias=0.0)
    with pytest.raises(ValueError):
        LegendreEmulator(2, product="tanh", tanh_tolerance=0.0)
