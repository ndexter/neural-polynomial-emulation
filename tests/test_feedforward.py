import pytest
import torch

from neural_polynomial_emulation import (
    FeedforwardProductNet,
    MultivariateBasisEmulator,
    ReLUProductNet,
    RePU2ProductNet,
    TanhProductNet,
    emulator_complexity,
    to_feedforward,
    total_degree_indices,
)


@pytest.mark.parametrize(
    ("source", "tolerance"),
    [
        (ReLUProductNet(-1.0, 1.0, 12), 3e-15),
        (TanhProductNet(1.0, 1.0, tolerance=1e-6), 1e-12),
        (RePU2ProductNet(), 0.0),
    ],
)
def test_feedforward_product_matches_functional_product(source, tolerance):
    generator = torch.Generator().manual_seed(7)
    inputs = 2.0 * torch.rand((1024, 2), generator=generator, dtype=torch.float64) - 1.0
    converted = to_feedforward(source, dtype=torch.float64)

    assert isinstance(converted, FeedforwardProductNet)
    actual = converted(inputs[:, 0], inputs[:, 1])
    expected = source(inputs[:, 0], inputs[:, 1])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize(
    ("a", "b", "num_layers"),
    [(-1.0, 1.0, 0), (-2.0, 2.0, 1), (0.0, 2.0, 3)],
)
def test_relu_feedforward_realization_handles_supported_intervals(a, b, num_layers):
    source = ReLUProductNet(a, b, num_layers)
    converted = to_feedforward(source)
    inputs = torch.linspace(a, b, 129, dtype=torch.float64)
    x = inputs
    y = torch.flip(inputs, dims=(0,))

    torch.testing.assert_close(
        converted(x, y),
        source(x, y),
        rtol=0.0,
        atol=2e-13,
    )
    assert emulator_complexity(converted) == emulator_complexity(source)


@pytest.mark.parametrize("product", ["relu", "tanh", "repu2"])
def test_converted_multivariate_model_preserves_output_and_source(product):
    indices = total_degree_indices(2, 3)
    source = MultivariateBasisEmulator(
        indices,
        family="chebyshev",
        product=product,
        num_layers=12,
        tanh_tolerance=1e-6,
    )
    converted = to_feedforward(source, dtype=torch.float64)
    generator = torch.Generator().manual_seed(11)
    inputs = 2.0 * torch.rand((257, 2), generator=generator, dtype=torch.float64) - 1.0

    assert converted is not source
    assert sum(parameter.numel() for parameter in source.parameters()) == 0
    assert sum(parameter.numel() for parameter in converted.parameters()) > 0
    torch.testing.assert_close(
        converted(inputs),
        source(inputs),
        rtol=0.0,
        atol={"relu": 2e-13, "tanh": 2e-10, "repu2": 5e-15}[product],
    )


@pytest.mark.parametrize("product", ["relu", "tanh", "repu2"])
def test_materialized_counts_match_analytic_counts(product):
    source = MultivariateBasisEmulator(
        total_degree_indices(2, 2),
        product=product,
        num_layers=12,
    )
    converted = to_feedforward(source, dtype=torch.float64)
    source_counts = emulator_complexity(source)
    converted_counts = emulator_complexity(converted)

    assert converted_counts == source_counts
    assert converted_counts["total_parameters"] == sum(
        parameter.numel() for parameter in converted.parameters()
    )
    assert converted_counts["nonzero_parameters"] == sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in converted.parameters()
    )
    assert converted_counts["trainable_parameters"] == 0


def test_tanh_conversion_resolves_step_for_requested_dtype():
    source = TanhProductNet(tolerance=1e-8)
    converted = to_feedforward(source, dtype=torch.float32)
    inputs = torch.tensor([[0.25, -0.75], [0.5, 0.125]], dtype=torch.float32)

    assert next(converted.parameters()).dtype == torch.float32
    assert converted.effective_step(torch.float32) == source.effective_step(
        torch.float32
    )
    torch.testing.assert_close(
        converted(inputs[:, 0], inputs[:, 1]),
        source(inputs[:, 0], inputs[:, 1]),
        rtol=0.0,
        atol=5e-5,
    )


def test_converted_tanh_model_preserves_step_introspection():
    source = MultivariateBasisEmulator(
        total_degree_indices(2, 3),
        product="tanh",
        tanh_tolerance=1e-6,
    )
    converted = to_feedforward(source, dtype=torch.float64)

    assert converted.cross_tanh_steps(torch.float64) == source.cross_tanh_steps(
        torch.float64
    )
    for converted_basis, source_basis in zip(
        converted.coordinate_emulators, source.coordinate_emulators
    ):
        for converted_polynomial, source_polynomial in zip(
            converted_basis.emulators, source_basis.emulators
        ):
            assert converted_polynomial.tanh_steps(
                torch.float64
            ) == source_polynomial.tanh_steps(torch.float64)


def test_feedforward_conversion_preserves_input_gradients():
    source = RePU2ProductNet()
    converted = to_feedforward(source)
    inputs = torch.tensor([[0.2, -0.7], [0.8, 0.3]], dtype=torch.float64)
    inputs.requires_grad_()

    converted(inputs[:, 0], inputs[:, 1]).sum().backward()

    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))


def test_feedforward_conversion_validates_inputs():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        to_feedforward(object())
    with pytest.raises(TypeError, match="floating point"):
        to_feedforward(RePU2ProductNet(), dtype=torch.int64)
