# Source notes

The initial mathematical constructions were consolidated from the local
`Polynomial_emulation_test` experiments, principally:

- `legendre_emulation_core.py`;
- `legendre_notebook_utils.py`;
- `Polynomial_emulation_DNN_experiments_diagnostics_run.ipynb`;
- `Polynomial_emulation_ReLU_vs_RePU_diagnostics.ipynb`.

The consolidation changes the Legendre normalization from orthonormality
under Lebesgue measure `dx` to orthonormality under the uniform probability
measure `dx/2`. Chebyshev polynomials use the Chebyshev probability measure
`dx / (pi sqrt(1-x^2))`, giving `psi_0 = 1` and
`psi_n = sqrt(2) T_n` for `n >= 1`. The package removes the global PyTorch
default-dtype mutation, separates exact recurrence evaluation from neural
emulation, and adds package-level tests.

The multivariate extension forms tensor products of these probability-
orthonormal univariate bases. It provides total-degree sets
`sum(alpha) <= p` and lower hyperbolic crosses
`prod(alpha_j + 1) <= n + 1`. The implementation evaluates the univariate
neural constructions coordinatewise and uses the selected fixed-weight
multiplier for the cross-coordinate products.

Recovery solvers, Christoffel sampling, experiment archives, generated plots,
and machine-specific launch scripts were deliberately excluded.

The tanh construction follows the source chain

- Ben Adcock et al., *Near-optimal learning of Banach-valued,
  high-dimensional functions via deep neural networks* (2025), Lemma 7.1 and
  Theorem 7.4;
- Tim De Ryck, Samuel Lanthaler, and Siddhartha Mishra, *On the approximation
  of functions by tanh neural networks*, Neural Networks 143 (2021), Lemma
  3.8.

The implementation uses the explicit centered second finite difference for
squaring, the polarization identity for two-factor multiplication, and a
bound-aware tree for products of multiple factors. Legendre and Chebyshev
emulators share this root-product engine. It contains no fitted weights.

Complexity diagnostics preserve these functional implementations. Their
parameter counts instead describe a modular feedforward embedding of each
scalar multiplier: dense weight-and-bias slots are reported as total
parameters, while the analytic sparse realization determines the nonzero
count. Counts include modular affine layers for root shifts, final polynomial
scalings, and degree-zero constants. Artificial padding between independent
branches is excluded.

The optional feedforward conversion materializes these multiplier and affine
blocks as frozen `torch.nn.Linear` layers. It deep-copies the enclosing
emulator, leaving the original functional construction unchanged. The tanh
step is resolved for the requested floating-point dtype at conversion time.
Its consecutive affine layers preserve the functional construction's
sum/difference and centered-finite-difference grouping, avoiding a single
cancellation-heavy output reduction in low precision.
