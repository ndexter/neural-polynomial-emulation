# Source notes

The initial mathematical constructions were consolidated from the local
`CAREER_legwork/Polynomial_emulation_test` experiments, principally:

- `legendre_emulation_core.py`;
- `legendre_notebook_utils.py`;
- `Polynomial_emulation_DNN_experiments_diagnostics_run.ipynb`;
- `Polynomial_emulation_ReLU_vs_RePU_diagnostics.ipynb`.

The consolidation changes the basis normalization from orthonormality under
Lebesgue measure `dx` to orthonormality under the uniform probability measure
`dx/2`. It also removes the global PyTorch default-dtype mutation, separates
exact recurrence evaluation from neural emulation, and adds package-level
tests.

Recovery solvers, Christoffel sampling, experiment archives, generated plots,
and machine-specific launch scripts were deliberately excluded.

The tanh construction follows the source chain

- Ben Adcock et al., *Near-optimal learning of Banach-valued,
  high-dimensional functions via deep neural networks* (2025), Lemma 7.1;
- Tim De Ryck, Samuel Lanthaler, and Siddhartha Mishra, *On the approximation
  of functions by tanh neural networks*, Neural Networks 143 (2021), Lemma
  3.8.

The implementation uses the explicit centered second finite difference for
squaring, the polarization identity for two-factor multiplication, and a
balanced tree for products of multiple factors. It contains no fitted weights.

