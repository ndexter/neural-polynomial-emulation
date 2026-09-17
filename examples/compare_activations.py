"""Compare ReLU, tanh, and RePU-2 emulation for both polynomial families."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_polynomial_emulation import basis_error_by_degree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-degree", type=int, default=12)
    parser.add_argument("--tanh-step", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=2001)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("legendre", "chebyshev"),
        default=("legendre", "chebyshev"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("example_outputs/activations")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in args.families:
        for product in ("relu", "tanh", "repu2"):
            errors = basis_error_by_degree(
                args.max_degree,
                family=family,
                product=product,
                num_layers=args.num_layers,
                tanh_step=args.tanh_step,
                grid_size=args.grid_size,
            )
            rows.extend(
                {
                    "family": family,
                    "degree": degree,
                    "product": product,
                    "sup_error": error,
                }
                for degree, error in enumerate(errors)
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "basis_errors.csv", index=False)
    fig, axes = plt.subplots(
        1, len(args.families), figsize=(7 * len(args.families), 4.5), squeeze=False
    )
    for ax, family in zip(axes[0], args.families):
        family_frame = frame[frame["family"] == family]
        for product, group in family_frame.groupby("product"):
            ax.semilogy(
                group["degree"],
                np.maximum(group["sup_error"], 1e-18),
                marker="o",
                label=product,
            )
        ax.set(
            xlabel="degree",
            ylabel="grid maximum error",
            title=f"Probability-normalized {family.title()} emulation",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "basis_errors.png", dpi=180)


if __name__ == "__main__":
    main()
