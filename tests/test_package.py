import neural_polynomial_emulation as npe


def test_public_api_and_version():
    expected = {
        "ChebyshevBasisEmulator",
        "ChebyshevEmulator",
        "FeedforwardProductNet",
        "LegendreBasisEmulator",
        "LegendreEmulator",
        "MultivariateBasisEmulator",
        "ReLUProductNet",
        "TanhProductNet",
        "TanhSquaringNet",
        "RePU2ProductNet",
        "RePU2Activation",
        "ReLUSquaringNet",
        "emulated_chebyshev_basis",
        "emulated_legendre_basis",
        "emulated_multivariate_basis",
        "emulator_complexity",
        "exact_chebyshev_basis",
        "exact_legendre_basis",
        "exact_multivariate_basis",
        "hyperbolic_cross_indices",
        "legendre_basis_error_by_degree",
        "total_degree_indices",
        "to_feedforward",
        "chebyshev_basis_error_by_degree",
    }
    assert expected.issubset(set(npe.__all__))
    assert npe.__version__ == "0.5.0"
    assert npe.OPTIMAL_TANH_BIAS > 1.0
