# Neural Polynomial Emulation

`neural-polynomial-emulation` provides constructive PyTorch networks for
Legendre and first-kind Chebyshev tensor-product bases on `[-1, 1]^d`. The
networks have fixed analytic weights: they are mathematical constructions
rather than models trained from data.

Each family uses its natural probability measure. For Legendre polynomials,

$$
d\rho_L(x)=\frac{dx}{2},
$$

the orthonormal Legendre basis is

$$
\psi_n(x)=\sqrt{2n+1}\,P_n(x).
$$

Thus `psi_0 = 1`, `psi_n(1) = sqrt(2n+1)`, and the intrinsic weight is
`sqrt(2n+1)`. For Chebyshev polynomials,

$$
d\rho_C(x)=\frac{dx}{\pi\sqrt{1-x^2}},
$$

and the orthonormal basis is

$$
\psi_0(x)=1, \qquad \psi_n(x)=\sqrt{2}\,T_n(x),\quad n\geq 1.
$$

These family-specific probability normalizations are fixed; there is no
alternative normalization option.

For a multi-index $\nu\in\mathbb{N}_0^d$, the multivariate basis function is

$$
\Psi_\nu(x)=\prod_{j=1}^d\psi_{\nu_j}(x_j).
$$

It is orthonormal for the corresponding product probability measure. The
package supplies the total-degree and lower-hyperbolic-cross index sets

```math
\Lambda^{\mathrm{TD}}_{d,p}
= \lbrace \nu\in\mathbb{N}_0^d : \lVert\nu\rVert_1\leq p \rbrace,
\qquad
\Lambda^{\mathrm{HC}}_{d,n}
= \lbrace \nu\in\mathbb{N}_0^d :
\prod_{j=1}^d(\nu_j+1)\leq n+1 \rbrace.
```

## Components

- `ReLUSquaringNet`: a depth-controlled piecewise-linear approximation of
  squaring, with its analytic uniform error bound.
- `ReLUProductNet`: approximate multiplication obtained from three squaring
  networks.
- `TanhSquaringNet`: a three-neuron approximation of squaring obtained from a
  centered second finite difference of `tanh`.
- `TanhProductNet`: the corresponding six-neuron approximate multiplier,
  obtained by polarization.
- `RePU2ProductNet`: exact multiplication, up to floating-point roundoff,
  using the activation `relu(z)**2`.
- `LegendreEmulator`: root-factorized neural emulation of one normalized
  Legendre polynomial.
- `ChebyshevEmulator`: root-factorized neural emulation of one normalized
  first-kind Chebyshev polynomial.
- `LegendreBasisEmulator`: simultaneous evaluation of degrees zero through
  `max_degree`.
- `ChebyshevBasisEmulator`: the corresponding Chebyshev sequence evaluator.
- `exact_legendre_basis` and `exact_chebyshev_basis`: stable three-term
  recurrences used as references.
- `basis_polynomial_formula`: exact rational monomial coefficients and
  probability-normalization factors for both supported families.
- `total_degree_indices` and `hyperbolic_cross_indices`: deterministic,
  downward-closed multi-index generators ordered by total degree and then
  lexicographically.
- `MultivariateBasisEmulator`: tensor-product neural emulation on an explicit
  multi-index set.
- `exact_multivariate_basis` and `emulated_multivariate_basis`: exact and
  neural multivariate design matrices.
- `emulator_complexity`: construction-aware counts of multiplication nodes,
  scalar activation units, dense total parameters, nonzero parameters,
  trainable implementation parameters, and fixed buffers.
- `to_feedforward`: nonmutating conversion to frozen, explicit `nn.Linear`
  multiplier realizations suitable for direct evaluation, serialization, and
  comparison with the functional constructions. Its tanh realization keeps
  the sum/difference and centered-finite-difference reductions in separate
  affine layers for stable float32 evaluation across linear-algebra backends.
- `interactive_network_figure`: a rotatable Plotly X-ray of every affine
  layer, activation, computational branch, and basis output in a converted
  multivariate emulator.

## Example

```python
import torch

from neural_polynomial_emulation import (
    MultivariateBasisEmulator,
    exact_multivariate_basis,
    hyperbolic_cross_indices,
    to_feedforward,
)

x = 2.0 * torch.rand(512, 3, dtype=torch.float64) - 1.0
multi_index = hyperbolic_cross_indices(dimension=3, order=8)
emulator = MultivariateBasisEmulator(
    multi_index,
    family="chebyshev",
    product="tanh",
    tanh_tolerance=1e-4,
).to(dtype=x.dtype)

approx = emulator(x)
exact = exact_multivariate_basis(x, multi_index, family="chebyshev")
error = torch.max(torch.abs(approx - exact), dim=0).values

# Materialize frozen Linear/activation multiplier blocks without changing emulator.
feedforward = to_feedforward(emulator, dtype=x.dtype)
feedforward_approx = feedforward(x)
```

The root-factorized construction is intentionally exposed as the object of
study. The ReLU and RePU-2 paths use sequential multiplication. The tanh path
uses a binary product tree whose greedy grouping favors small intermediate
root-product sup norms. Each group bound is computed from the endpoints and
the real critical points of that partial polynomial on `[-1, 1]`.
Approximate multiplication errors and conditioning of the factorized
polynomial can grow with degree. The diagnostics report this behavior; the
package does not conceal it with an exact recurrence.

The multivariate emulator first constructs each required univariate basis
sequence. It then applies additional fixed-weight multipliers across active
coordinates. RePU-2 gives the tensor product exactly up to floating-point
roundoff. ReLU and tanh introduce approximation error in both the univariate
root products and the cross-coordinate products.

`emulator_complexity` does not materialize or replace these functional
implementations. For size comparisons, it embeds each scalar multiplier in a
standard two-input, scalar-output feedforward network and adds the counts over
multiplier nodes. The dense total includes every weight and bias slot; the
nonzero count uses the sparse analytic realization. Modular affine layers for
root shifts, final polynomial scaling, and degree-zero constants are included;
only artificial padding between independent branches is excluded.

`to_feedforward(model, dtype=...)` materializes that same modular embedding in
a deep copy of the model. It replaces each multiplier with frozen linear layers
and the corresponding elementwise activation. Root shifts, output scaling, and
degree-zero constants are also materialized as frozen affine layers, while the
product-tree wiring and basis assembly are retained. Automatic tanh steps are
resolved for the requested dtype. Roundoff-scale representations of
analytically zero roots are set to exact zero in the copy so sparsity masks
reflect the mathematical construction. The source model is not modified.

## Tanh construction

`TanhSquaringNet` approximates the square by a centered second finite
difference of `tanh`. Its default bias is
`atanh(sqrt(2/3))`, where the fourth derivative of `tanh` vanishes. This
cancels the leading finite-difference error and changes the exact-arithmetic
truncation error from second to fourth order in the step.
`TanhProductNet` combines two such squares using
`xy = ((x+y)^2 - (x-y)^2)/4`. Composing these multipliers in a balanced tree
gives the tanh polynomial emulators. This implements the Legendre and
Chebyshev root-factorized construction in Theorem 7.4 of Adcock et al. (2025),
using the tanh multiplier from its Lemma 7.1 and De Ryck, Lanthaler, and
Mishra (2021, Lemma 3.8).

By default, `tanh_step=None` selects a step independently for every product
node. The selector allocates `tanh_tolerance` over the tree and combines its
fourth-order truncation model with a dtype-dependent cancellation floor.
Consequently float32 uses larger steps than float64, and different tree levels
can use different steps. This is a practical numerical heuristic rather than
a certified global error bound. Pass a positive `tanh_step` to override it for
experiments such as a step-size sweep. The individual emulator classes expose
`tanh_steps(dtype)`, which reports the effective steps grouped by tree level.

## Development

The notebooks `notebooks/multivariate_basis_diagnostics.ipynb`,
`notebooks/multivariate_accuracy_complexity.ipynb`, and
`notebooks/feedforward_embedding_diagnostics.ipynb` compare basis columns,
sweep accuracy and size controls, and verify the executable feedforward
conversion. `notebooks/feedforward_weight_structure.ipynb` visualizes signed
weights, biases, exact nonzero masks, root/normalization structure, and an
interactive three-dimensional X-ray of the complete branched network.
`notebooks/formula_reference_diagnostics.ipynb` lists the exact normalized
monomial formulas and compares recurrence, direct monomial evaluation, and
all three neural constructions against a high-precision formula reference as
order and dimension increase. Its fixed coefficient fixture is generated from
SymPy's exact Legendre and Chebyshev polynomials and is independent of the
package's runtime coefficient generator. The test suite checks both sources
against SymPy.
`notebooks/dimension_roundoff_accumulation.ipynb` compares float32 and float64
precision sensitivity with increasing dimension for all three multiplication
constructions and both functional and feedforward implementations. Reported
maximum errors are maxima over the displayed grid or fixed random sample, not
certified uniform-error bounds.

```bash
python -m pip install -e ".[test,notebook]"
python -m pytest -q
python examples/compare_activations.py
```
