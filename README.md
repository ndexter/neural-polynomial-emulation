# Neural Polynomial Emulation

`neural-polynomial-emulation` provides constructive PyTorch networks that
emulate Legendre and first-kind Chebyshev polynomials on `[-1, 1]`. The
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

## Example

```python
import torch

from neural_polynomial_emulation import (
    ChebyshevBasisEmulator,
    exact_chebyshev_basis,
)

x = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
emulator = ChebyshevBasisEmulator(
    max_degree=8,
    product="tanh",
    tanh_tolerance=1e-4,
).to(dtype=x.dtype)

approx = emulator(x)
exact = exact_chebyshev_basis(x, max_degree=8)
error = torch.max(torch.abs(approx - exact), dim=0).values
```

The root-factorized construction is intentionally exposed as the object of
study. The ReLU and RePU-2 paths use sequential multiplication. The tanh path
uses a binary product tree whose greedy grouping favors small intermediate
root-product sup norms. Each group bound is computed from the endpoints and
the real critical points of that partial polynomial on `[-1, 1]`.
Approximate multiplication errors and conditioning of the factorized
polynomial can grow with degree. The diagnostics report this behavior; the
package does not conceal it with an exact recurrence.

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

```bash
python -m pip install -e ".[test,examples]"
python -m pytest -q
python examples/compare_activations.py
```
