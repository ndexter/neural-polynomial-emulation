"""Measure exact-versus-emulated basis-matrix mismatch for both families."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neural_polynomial_emulation import (
    emulated_chebyshev_basis,
    emulated_legendre_basis,
    exact_chebyshev_basis,
    exact_legendre_basis,
    relative_frobenius_mismatch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", default="4,8,12,16")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("legendre", "chebyshev"),
        default=("legendre", "chebyshev"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("example_outputs/basis_mismatch.csv")
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    family_apis = {
        "legendre": (
            lambda: rng.uniform(-1.0, 1.0, size=args.samples),
            exact_legendre_basis,
            emulated_legendre_basis,
        ),
        "chebyshev": (
            lambda: np.cos(rng.uniform(0.0, np.pi, size=args.samples)),
            exact_chebyshev_basis,
            emulated_chebyshev_basis,
        ),
    }
    rows = []
    for family in args.families:
        sampler, exact_basis, emulated_basis = family_apis[family]
        x = sampler()
        for max_degree in (int(item) for item in args.degrees.split(",")):
            exact = exact_basis(x, max_degree)
            for product in ("relu", "tanh", "repu2"):
                approximate = (
                    emulated_basis(
                        x,
                        max_degree,
                        product=product,
                        num_layers=args.num_layers,
                    )
                    .detach()
                    .numpy()
                )
                rows.append(
                    {
                        "family": family,
                        "max_degree": max_degree,
                        "product": product,
                        "num_layers": args.num_layers,
                        "relative_frobenius_mismatch": relative_frobenius_mismatch(
                            exact, approximate
                        ),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
