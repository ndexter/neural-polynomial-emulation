import neural_polynomial_emulation as npe


def test_public_api_and_version():
    expected = {
        "ChebyshevBasisEmulator",
        "ChebyshevEmulator",
        "LegendreBasisEmulator",
        "LegendreEmulator",
        "ReLUProductNet",
        "TanhProductNet",
        "TanhSquaringNet",
        "RePU2ProductNet",
        "ReLUSquaringNet",
        "emulated_chebyshev_basis",
        "emulated_legendre_basis",
        "exact_chebyshev_basis",
        "exact_legendre_basis",
        "legendre_basis_error_by_degree",
        "chebyshev_basis_error_by_degree",
    }
    assert expected.issubset(set(npe.__all__))
    assert npe.__version__ == "0.4.0"
    assert npe.OPTIMAL_TANH_BIAS > 1.0
