import numpy as np
import pytest
import torch

from neural_polynomial_emulation import (
    LegendreBasisEmulator,
    basis_error_by_degree,
    emulated_legendre_basis,
    exact_legendre_basis,
    relative_frobenius_mismatch,
)


def test_basis_emulator_shape_and_column_consistency():
    x = torch.linspace(-0.9, 0.9, 33, dtype=torch.float64).reshape(-1, 1)
    model = LegendreBasisEmulator(6, product="repu2").to(dtype=x.dtype)
    out = model(x)
    assert out.shape == (33, 7)
    for degree, emulator in enumerate(model.emulators):
        torch.testing.assert_close(out[:, degree], emulator(x[:, 0]))


def test_repu_basis_matches_exact_basis():
    x = np.linspace(-0.95, 0.95, 129)
    approximate = emulated_legendre_basis(x, 8, product="repu2")
    exact = exact_legendre_basis(x, 8)
    np.testing.assert_allclose(
        approximate.detach().numpy(), exact, rtol=2e-10, atol=2e-10
    )


def test_batched_convenience_matches_unbatched():
    x = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    full = emulated_legendre_basis(x, 5, product="relu", num_layers=8)
    batched = emulated_legendre_basis(x, 5, product="relu", num_layers=8, batch_size=13)
    torch.testing.assert_close(full, batched)


def test_diagnostics_return_finite_values():
    errors = basis_error_by_degree(6, product="relu", num_layers=8, grid_size=257)
    assert errors.shape == (7,)
    assert np.all(np.isfinite(errors))
    exact = np.eye(3)
    assert relative_frobenius_mismatch(exact, exact) == 0.0
    with pytest.raises(ValueError):
        relative_frobenius_mismatch(np.zeros((2, 2)), np.zeros((2, 2)))


def test_generic_diagnostics_support_both_families():
    for family in ("legendre", "chebyshev"):
        errors = basis_error_by_degree(4, family=family, product="repu2", grid_size=129)
        assert errors.shape == (5,)
    with pytest.raises(ValueError, match="family"):
        basis_error_by_degree(2, family="jacobi")


def test_invalid_basis_parameters():
    with pytest.raises(ValueError):
        LegendreBasisEmulator(-1)
    with pytest.raises(ValueError):
        emulated_legendre_basis(torch.ones(3), 2, batch_size=0)
