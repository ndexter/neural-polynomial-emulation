"""Measure exact-versus-emulated Legendre basis-matrix mismatch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neural_polynomial_emulation import (
    emulated_legendre_basis,
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
        "--output", type=Path, default=Path("example_outputs/basis_mismatch.csv")
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-1.0, 1.0, size=args.samples)
    rows = []
    for max_degree in (int(item) for item in args.degrees.split(",")):
        exact = exact_legendre_basis(x, max_degree)
        for product in ("relu", "repu2"):
            approximate = (
                emulated_legendre_basis(
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
