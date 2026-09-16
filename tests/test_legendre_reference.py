import numpy as np
import torch

from neural_polynomial_emulation import exact_legendre, exact_legendre_basis


def test_low_degree_probability_normalization():
    x = np.linspace(-1.0, 1.0, 17)
    np.testing.assert_allclose(exact_legendre(x, 0), 1.0)
    np.testing.assert_allclose(exact_legendre(x, 1), np.sqrt(3.0) * x)
    np.testing.assert_allclose(
        exact_legendre(x, 2), np.sqrt(5.0) * (3.0 * x**2 - 1.0) / 2.0
    )


def test_endpoint_values_and_shape():
    max_degree = 40
    basis = exact_legendre_basis(np.array([1.0]), max_degree)
    assert basis.shape == (1, max_degree + 1)
    np.testing.assert_allclose(
        basis[0], np.sqrt(2.0 * np.arange(max_degree + 1) + 1.0), rtol=2e-13, atol=2e-13
    )


def test_probability_orthonormality_by_gauss_legendre_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    basis = exact_legendre_basis(nodes, 12)
    gram = basis.T @ ((weights / 2.0)[:, None] * basis)
    np.testing.assert_allclose(gram, np.eye(13), rtol=2e-13, atol=2e-13)


def test_numpy_and_torch_reference_agree_and_preserve_tensor_gradient():
    x_np = np.linspace(-0.9, 0.9, 31)
    x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(
        exact_legendre(x_np, 9), exact_legendre(x_t, 9).detach().numpy()
    )
    exact_legendre(x_t, 9).sum().backward()
    assert torch.all(torch.isfinite(x_t.grad))


def test_trailing_singleton_coordinate_axis_is_removed():
    x = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    assert exact_legendre_basis(x, 4).shape == (9, 5)
