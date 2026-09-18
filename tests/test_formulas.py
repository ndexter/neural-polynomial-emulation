import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from neural_polynomial_emulation import (
    basis_polynomial_formula,
    exact_chebyshev_basis,
    exact_legendre_basis,
)

REFERENCE_DATA = (
    Path(__file__).parents[1]
    / "notebooks"
    / "data"
    / "orthogonal_polynomial_coefficients.json"
)


def _sympy_coefficients(polynomial, variable):
    coefficients = reversed(sp.Poly(polynomial, variable, domain=sp.QQ).all_coeffs())
    return tuple(
        Fraction(int(coefficient.p), int(coefficient.q))
        for coefficient in coefficients
    )


def test_known_legendre_formulas():
    assert basis_polynomial_formula("legendre", 0).coefficients == (Fraction(1),)
    assert basis_polynomial_formula("legendre", 2).coefficients == (
        Fraction(-1, 2),
        Fraction(0),
        Fraction(3, 2),
    )
    formula = basis_polynomial_formula("legendre", 3)
    assert formula.coefficients == (
        Fraction(0),
        Fraction(-3, 2),
        Fraction(0),
        Fraction(5, 2),
    )
    assert formula.normalization_radicand == 7


def test_known_chebyshev_formulas():
    assert basis_polynomial_formula("chebyshev", 0).coefficients == (Fraction(1),)
    assert basis_polynomial_formula("chebyshev", 3).coefficients == (
        Fraction(0),
        Fraction(-3),
        Fraction(0),
        Fraction(4),
    )
    formula = basis_polynomial_formula("chebyshev", 4)
    assert formula.coefficients == (
        Fraction(1),
        Fraction(0),
        Fraction(-8),
        Fraction(0),
        Fraction(8),
    )
    assert formula.normalization_radicand == 2


@pytest.mark.parametrize(
    ("family", "sympy_family"),
    [("legendre", sp.legendre), ("chebyshev", sp.chebyshevt)],
)
def test_generated_coefficients_agree_with_sympy(family, sympy_family):
    variable = sp.Symbol("x")
    for degree in range(33):
        expected = _sympy_coefficients(sympy_family(degree, variable), variable)
        assert basis_polynomial_formula(family, degree).coefficients == expected


def test_fixed_reference_coefficients_agree_with_sympy():
    reference = json.loads(REFERENCE_DATA.read_text())
    variable = sp.Symbol("x")
    for family, sympy_family in (
        ("legendre", sp.legendre),
        ("chebyshev", sp.chebyshevt),
    ):
        assert len(reference[family]) == reference["max_degree"] + 1
        for degree, entry in enumerate(reference[family]):
            expected = _sympy_coefficients(sympy_family(degree, variable), variable)
            assert tuple(map(Fraction, entry["coefficients"])) == expected
            expected_radicand = (
                2 * degree + 1
                if family == "legendre"
                else (1 if degree == 0 else 2)
            )
            assert entry["normalization_radicand"] == expected_radicand


@pytest.mark.parametrize(
    ("family", "recurrence"),
    [
        ("legendre", exact_legendre_basis),
        ("chebyshev", exact_chebyshev_basis),
    ],
)
def test_explicit_formulas_agree_with_recurrence(family, recurrence):
    points = np.linspace(-1.0, 1.0, 101)
    expected = recurrence(points, 16)
    explicit = np.column_stack(
        [basis_polynomial_formula(family, degree)(points) for degree in range(17)]
    )

    np.testing.assert_allclose(explicit, expected, rtol=2e-10, atol=2e-10)


def test_formula_latex_and_validation():
    assert (
        basis_polynomial_formula("legendre", 2).latex()
        == r"\sqrt{5}\left(\frac{3}{2}x^{2} - \frac{1}{2}\right)"
    )
    assert basis_polynomial_formula("chebyshev", 1).latex() == r"\sqrt{2}x"
    with pytest.raises(ValueError, match="family"):
        basis_polynomial_formula("jacobi", 2)
    with pytest.raises(ValueError, match="nonnegative"):
        basis_polynomial_formula("legendre", -1)
