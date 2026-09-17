import math

import numpy as np
import pytest
import torch

from neural_polynomial_emulation import (
    ChebyshevBasisEmulator,
    ChebyshevEmulator,
    chebyshev_basis_error_by_degree,
    chebyshev_l2_norm,
    chebyshev_leading_coefficient,
    chebyshev_roots,
    emulated_chebyshev_basis,
    exact_chebyshev,
    exact_chebyshev_basis,
)


def test_chebyshev_probability_normalization():
    assert chebyshev_l2_norm(0) == 1.0
    assert chebyshev_l2_norm(4) == pytest.approx(1.0 / math.sqrt(2.0))
    nodes = np.cos((np.arange(4096) + 0.5) * np.pi / 4096)
    basis = exact_chebyshev_basis(nodes, 8)
    gram = basis.T @ basis / len(nodes)
    np.testing.assert_allclose(gram, np.eye(9), rtol=2e-14, atol=2e-14)


def test_chebyshev_roots_and_leading_coefficient():
    degree = 7
    expected_roots = np.sort(
        np.cos((2 * np.arange(1, degree + 1) - 1) * np.pi / (2 * degree))
    )
    np.testing.assert_allclose(chebyshev_roots(degree), expected_roots)
    assert chebyshev_leading_coefficient(0) == 1.0
    assert chebyshev_leading_coefficient(degree) == pytest.approx(
        math.sqrt(2.0) * 2.0 ** (degree - 1)
    )


def test_exact_chebyshev_matches_cosine_definition():
    x = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
    for degree in range(9):
        scale = 1.0 if degree == 0 else math.sqrt(2.0)
        expected = scale * torch.cos(degree * torch.acos(x))
        torch.testing.assert_close(
            exact_chebyshev(x, degree), expected, rtol=2e-13, atol=2e-13
        )


@pytest.mark.parametrize("degree", range(9))
def test_repu2_chebyshev_emulator_matches_reference(degree):
    x = torch.linspace(-0.95, 0.95, 257, dtype=torch.float64)
    torch.testing.assert_close(
        ChebyshevEmulator(degree, product="repu2")(x),
        exact_chebyshev(x, degree),
        rtol=4e-10,
        atol=4e-10,
    )


def test_tanh_chebyshev_emulator_is_accurate_at_degree_24():
    x = torch.linspace(-1.0, 1.0, 4097, dtype=torch.float64)
    error = torch.max(
        torch.abs(ChebyshevEmulator(24, product="tanh")(x) - exact_chebyshev(x, 24))
    ).item()
    assert error < 1e-4


def test_chebyshev_basis_interfaces_and_diagnostics():
    x = torch.linspace(-0.9, 0.9, 33, dtype=torch.float64).reshape(-1, 1)
    model = ChebyshevBasisEmulator(6, product="tanh").to(dtype=x.dtype)
    assert model(x).shape == (33, 7)
    batched = emulated_chebyshev_basis(x, 6, product="tanh", batch_size=7)
    torch.testing.assert_close(model(x), batched)
    errors = chebyshev_basis_error_by_degree(6, product="tanh", grid_size=257)
    assert errors.shape == (7,)
    assert np.all(np.isfinite(errors))


def test_invalid_chebyshev_parameters():
    with pytest.raises(ValueError):
        chebyshev_roots(-1)
    with pytest.raises(ValueError):
        ChebyshevBasisEmulator(-1)
    with pytest.raises(ValueError):
        emulated_chebyshev_basis(torch.ones(3), 2, batch_size=0)
