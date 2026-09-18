"""Exact monomial formulas for the supported orthonormal polynomial bases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._root_factorized import _validate_degree


@dataclass(frozen=True)
class PolynomialFormula:
    r"""Exact rational polynomial with a square-root normalization.

    The represented basis function is

    .. math::

        \psi_n(x)=\sqrt{r}\sum_{k=0}^n c_k x^k,

    where ``normalization_radicand`` is :math:`r` and ``coefficients[k]`` is
    the exact rational number :math:`c_k`.
    """

    family: str
    degree: int
    coefficients: tuple[Fraction, ...]
    normalization_radicand: int

    @property
    def normalization(self) -> float:
        """Return the floating-point square-root normalization."""
        return math.sqrt(self.normalization_radicand)

    def __call__(self, x) -> np.ndarray:
        """Evaluate the explicit monomial formula in NumPy float64 arithmetic."""
        values = np.asarray(x, dtype=np.float64)
        coefficients = np.asarray(
            [float(coefficient) for coefficient in self.coefficients],
            dtype=np.float64,
        )
        return self.normalization * np.polynomial.polynomial.polyval(
            values, coefficients
        )

    def latex(self, variable: str = "x") -> str:
        """Return a LaTeX expression for the exact normalized polynomial."""
        polynomial = _polynomial_latex(self.coefficients, variable)
        if self.normalization_radicand == 1:
            return polynomial
        normalization = rf"\sqrt{{{self.normalization_radicand}}}"
        nonzero_terms = sum(coefficient != 0 for coefficient in self.coefficients)
        if nonzero_terms == 1:
            return normalization + polynomial
        return normalization + rf"\left({polynomial}\right)"


def _legendre_coefficients(degree: int) -> tuple[Fraction, ...]:
    coefficients = [Fraction(0) for _ in range(degree + 1)]
    for index in range(degree // 2 + 1):
        power = degree - 2 * index
        numerator = (-1) ** index * math.factorial(2 * degree - 2 * index)
        denominator = (
            2**degree
            * math.factorial(index)
            * math.factorial(degree - index)
            * math.factorial(power)
        )
        coefficients[power] = Fraction(numerator, denominator)
    return tuple(coefficients)


def _chebyshev_coefficients(degree: int) -> tuple[Fraction, ...]:
    if degree == 0:
        return (Fraction(1),)
    coefficients = [Fraction(0) for _ in range(degree + 1)]
    for index in range(degree // 2 + 1):
        power = degree - 2 * index
        numerator = (
            (-1) ** index * degree * math.factorial(degree - index - 1) * 2**power
        )
        denominator = 2 * math.factorial(index) * math.factorial(degree - 2 * index)
        coefficients[power] = Fraction(numerator, denominator)
    return tuple(coefficients)


def basis_polynomial_formula(family: str, degree: int) -> PolynomialFormula:
    r"""Return the exact monomial formula for a probability-orthonormal mode.

    Legendre modes are orthonormal for :math:`dx/2`. First-kind Chebyshev
    modes are orthonormal for :math:`dx/(\pi\sqrt{1-x^2})`.
    """
    degree = _validate_degree(degree)
    family = str(family).lower()
    if family == "legendre":
        return PolynomialFormula(
            family=family,
            degree=degree,
            coefficients=_legendre_coefficients(degree),
            normalization_radicand=2 * degree + 1,
        )
    if family == "chebyshev":
        return PolynomialFormula(
            family=family,
            degree=degree,
            coefficients=_chebyshev_coefficients(degree),
            normalization_radicand=1 if degree == 0 else 2,
        )
    raise ValueError("family must be 'legendre' or 'chebyshev'")


def _fraction_magnitude_latex(value: Fraction) -> str:
    magnitude = abs(value)
    if magnitude.denominator == 1:
        return str(magnitude.numerator)
    return rf"\frac{{{magnitude.numerator}}}{{{magnitude.denominator}}}"


def _polynomial_latex(coefficients: tuple[Fraction, ...], variable: str) -> str:
    terms: list[tuple[str, str]] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if coefficient == 0:
            continue
        sign = "-" if coefficient < 0 else "+"
        if power == 0:
            body = _fraction_magnitude_latex(coefficient)
        else:
            magnitude = (
                "" if abs(coefficient) == 1 else _fraction_magnitude_latex(coefficient)
            )
            variable_power = variable if power == 1 else rf"{variable}^{{{power}}}"
            body = magnitude + variable_power
        terms.append((sign, body))
    if not terms:
        return "0"
    first_sign, first_body = terms[0]
    expression = ("-" if first_sign == "-" else "") + first_body
    for sign, body in terms[1:]:
        expression += f" {sign} {body}"
    return expression
