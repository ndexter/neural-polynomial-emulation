"""Downward-closed multi-index sets for tensor-product polynomial bases."""

from __future__ import annotations

import math

import numpy as np


def _validate_dimension_order(dimension: int, order: int) -> tuple[int, int]:
    dimension = int(dimension)
    order = int(order)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if order < 0:
        raise ValueError("order must be nonnegative")
    return dimension, order


def _sort_indices(indices: list[tuple[int, ...]]) -> np.ndarray:
    """Sort first by total degree and then lexicographically."""
    return np.asarray(
        sorted(indices, key=lambda alpha: (sum(alpha), alpha)), dtype=np.int64
    )


def total_degree_indices(dimension: int, degree: int) -> np.ndarray:
    r"""Return ``alpha`` with ``sum(alpha) <= degree``.

    The cardinality is ``binomial(dimension + degree, dimension)``.
    """
    dimension, degree = _validate_dimension_order(dimension, degree)
    indices: list[tuple[int, ...]] = []

    def append_indices(
        prefix: tuple[int, ...], remaining_dimension: int, remaining_degree: int
    ) -> None:
        if remaining_dimension == 1:
            indices.extend(prefix + (value,) for value in range(remaining_degree + 1))
            return
        for value in range(remaining_degree + 1):
            append_indices(
                prefix + (value,), remaining_dimension - 1, remaining_degree - value
            )

    append_indices((), dimension, degree)
    result = _sort_indices(indices)
    expected_size = math.comb(dimension + degree, dimension)
    assert result.shape == (expected_size, dimension)
    return result


def hyperbolic_cross_indices(dimension: int, order: int) -> np.ndarray:
    r"""Return ``alpha`` with ``prod(alpha_j + 1) <= order + 1``.

    This convention makes ``order=0`` the constant index set and agrees with
    the common definition of the lower hyperbolic cross of order ``order``.
    """
    dimension, order = _validate_dimension_order(dimension, order)
    product_limit = order + 1
    indices: list[tuple[int, ...]] = []

    def append_indices(
        prefix: tuple[int, ...], remaining_dimension: int, remaining_product: int
    ) -> None:
        if remaining_dimension == 0:
            indices.append(prefix)
            return
        for shifted_value in range(1, remaining_product + 1):
            append_indices(
                prefix + (shifted_value - 1,),
                remaining_dimension - 1,
                remaining_product // shifted_value,
            )

    append_indices((), dimension, product_limit)
    return _sort_indices(indices)


def multi_index_set(
    dimension: int, order: int, kind: str = "total_degree"
) -> np.ndarray:
    """Construct a supported multi-index set in deterministic order."""
    kind = str(kind).lower()
    if kind == "total_degree":
        return total_degree_indices(dimension, order)
    if kind == "hyperbolic_cross":
        return hyperbolic_cross_indices(dimension, order)
    raise ValueError("kind must be 'total_degree' or 'hyperbolic_cross'")


def validate_multi_index(multi_index) -> np.ndarray:
    """Return a validated integer multi-index array without reordering it."""
    values = np.asarray(multi_index)
    if values.ndim != 2:
        raise ValueError("multi_index must be a two-dimensional array")
    if values.shape[0] == 0:
        raise ValueError("multi_index must contain at least one index")
    if values.shape[1] == 0:
        raise ValueError("multi_index dimension must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("multi_index must be finite")
    integer_values = values.astype(np.int64)
    if not np.array_equal(values, integer_values):
        raise ValueError("multi_index entries must be integers")
    if np.any(integer_values < 0):
        raise ValueError("multi_index entries must be nonnegative")
    if len({tuple(row) for row in integer_values.tolist()}) != len(integer_values):
        raise ValueError("multi_index entries must be distinct")
    return integer_values.copy()


def is_downward_closed(multi_index) -> bool:
    """Return whether every coordinatewise predecessor is present."""
    values = validate_multi_index(multi_index)
    index_set = {tuple(row) for row in values.tolist()}
    for alpha in index_set:
        for coordinate, value in enumerate(alpha):
            if value == 0:
                continue
            predecessor = list(alpha)
            predecessor[coordinate] -= 1
            if tuple(predecessor) not in index_set:
                return False
    return True
