import neural_polynomial_emulation as npe


def test_public_api_and_version():
    expected = {
        "LegendreBasisEmulator",
        "LegendreEmulator",
        "ReLUProductNet",
        "TanhProductNet",
        "TanhSquaringNet",
        "RePU2ProductNet",
        "ReLUSquaringNet",
        "emulated_legendre_basis",
        "exact_legendre_basis",
    }
    assert expected.issubset(set(npe.__all__))
    assert npe.__version__ == "0.3.0"
    assert npe.OPTIMAL_TANH_BIAS > 1.0
