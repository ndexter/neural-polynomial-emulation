import math

import numpy as np
import pytest
import torch

from neural_polynomial_emulation import (
    MultivariateBasisEmulator,
    emulated_multivariate_basis,
    emulator_complexity,
    exact_chebyshev_basis,
    exact_legendre_basis,
    exact_multivariate_basis,
    hyperbolic_cross_indices,
    is_downward_closed,
    multi_index_set,
    total_degree_indices,
    validate_multi_index,
)


def test_total_degree_cardinality_ordering_and_downward_closure():
    indices = total_degree_indices(3, 4)
    assert indices.shape == (math.comb(7, 3), 3)
    keys = [(int(np.sum(alpha)), tuple(alpha)) for alpha in indices]
    assert keys == sorted(keys)
    assert is_downward_closed(indices)
    np.testing.assert_array_equal(indices, multi_index_set(3, 4, "total_degree"))


def test_hyperbolic_cross_convention_and_downward_closure():
    indices = hyperbolic_cross_indices(2, 184)
    assert indices.shape == (997, 2)
    assert np.all(np.prod(indices + 1, axis=1) <= 185)
    assert is_downward_closed(indices)
    np.testing.assert_array_equal(indices, multi_index_set(2, 184, "hyperbolic_cross"))
    assert hyperbolic_cross_indices(16, 16).shape == (8277, 16)


@pytest.mark.parametrize("family", ["legendre", "chebyshev"])
def test_exact_multivariate_basis_is_tensor_product(family):
    x = np.array([[0.25, -0.5], [0.75, 0.1]])
    indices = total_degree_indices(2, 3)
    actual = exact_multivariate_basis(x, indices, family=family)
    univariate = exact_legendre_basis if family == "legendre" else exact_chebyshev_basis
    first = univariate(x[:, 0], 3)
    second = univariate(x[:, 1], 3)
    expected = np.column_stack(
        [first[:, alpha[0]] * second[:, alpha[1]] for alpha in indices]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("family", ["legendre", "chebyshev"])
def test_tensor_product_basis_is_orthonormal_for_product_probability_measure(family):
    order = 8
    if family == "legendre":
        nodes, weights = np.polynomial.legendre.leggauss(order)
        weights = weights / 2.0
    else:
        nodes = np.cos((np.arange(order) + 0.5) * np.pi / order)
        weights = np.full(order, 1.0 / order)
    mesh = np.meshgrid(nodes, nodes, indexing="ij")
    points = np.column_stack([coordinate.reshape(-1) for coordinate in mesh])
    product_weights = np.outer(weights, weights).reshape(-1)
    indices = total_degree_indices(2, 3)
    basis = exact_multivariate_basis(points, indices, family=family)
    gram = basis.T @ (product_weights[:, None] * basis)
    np.testing.assert_allclose(gram, np.eye(len(indices)), rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("family", ["legendre", "chebyshev"])
@pytest.mark.parametrize("index_kind", ["total_degree", "hyperbolic_cross"])
def test_repu2_multivariate_emulator_matches_exact(family, index_kind):
    generator = np.random.default_rng(17)
    x = generator.uniform(-0.9, 0.9, size=(97, 3))
    indices = multi_index_set(3, 4, index_kind)
    exact = exact_multivariate_basis(x, indices, family=family)
    approximate = emulated_multivariate_basis(
        x, indices, family=family, product="repu2", batch_size=19
    )
    np.testing.assert_allclose(
        approximate.detach().numpy(), exact, rtol=2e-9, atol=2e-9
    )


def test_tanh_multivariate_emulator_and_step_introspection():
    generator = torch.Generator().manual_seed(9)
    x = 1.8 * torch.rand((129, 3), generator=generator, dtype=torch.float64) - 0.9
    indices = total_degree_indices(3, 3)
    model = MultivariateBasisEmulator(
        indices, family="chebyshev", product="tanh", tanh_tolerance=1e-4
    ).to(dtype=x.dtype)
    approximate = model(x)
    exact = exact_multivariate_basis(x, indices, family="chebyshev")
    assert torch.max(torch.abs(approximate - exact)).item() < 2e-4
    steps = model.cross_tanh_steps(torch.float64)
    assert len(steps) == len(indices)
    assert all(step > 0.0 for feature_steps in steps for step in feature_steps)


def test_multivariate_emulator_preserves_gradients():
    x = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    model = MultivariateBasisEmulator(
        total_degree_indices(2, 2), family="legendre", product="repu2"
    ).to(dtype=x.dtype)
    model(x).sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_complexity_counts_product_nodes_and_activation_units():
    indices = total_degree_indices(2, 2)
    expected_nodes = 3
    expectations = {
        "relu": (
            expected_nodes * 12 * 12,
            expected_nodes * 1765 + 24,
            expected_nodes * 528 + 16,
        ),
        "tanh": (18, 147, 82),
        "repu2": (12, 75, 52),
    }
    for product, (
        expected_units,
        expected_total,
        expected_nonzero,
    ) in expectations.items():
        model = MultivariateBasisEmulator(indices, product=product, num_layers=12)
        complexity = emulator_complexity(model)
        assert complexity["product_nodes"] == expected_nodes
        assert complexity["activation_units_per_sample"] == expected_units
        assert complexity["total_parameters"] == expected_total
        assert complexity["nonzero_parameters"] == expected_nonzero
        assert complexity["trainable_parameters"] == 0


def test_complexity_keeps_trainable_count_separate_from_realization_count():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        model.bias.copy_(torch.tensor([0.0, 3.0]))
    model.weight.requires_grad_(False)

    complexity = emulator_complexity(model)

    assert complexity["total_parameters"] == 0
    assert complexity["nonzero_parameters"] == 0
    assert complexity["trainable_parameters"] == 2


def test_multivariate_validation():
    with pytest.raises(ValueError, match="dimension"):
        total_degree_indices(0, 2)
    with pytest.raises(ValueError, match="order"):
        hyperbolic_cross_indices(2, -1)
    with pytest.raises(ValueError, match="kind"):
        multi_index_set(2, 2, "tensor_product")
    with pytest.raises(ValueError, match="integers"):
        validate_multi_index([[0.0, 0.5]])
    with pytest.raises(ValueError, match="distinct"):
        validate_multi_index([[0, 0], [0, 0]])
    indices = total_degree_indices(2, 2)
    with pytest.raises(ValueError, match="shape"):
        exact_multivariate_basis(np.ones((5, 3)), indices)
    with pytest.raises(ValueError, match="family"):
        MultivariateBasisEmulator(indices, family="hermite")
    with pytest.raises(ValueError, match="batch_size"):
        emulated_multivariate_basis(np.ones((5, 2)), indices, batch_size=0)
    empty = emulated_multivariate_basis(
        np.empty((0, 2)), indices, product="repu2", batch_size=3
    )
    assert empty.shape == (0, len(indices))
    with pytest.raises(TypeError, match="torch.Tensor"):
        MultivariateBasisEmulator(indices)(np.ones((5, 2)))
