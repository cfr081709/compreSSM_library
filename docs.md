# CompreSSM API Documentation

This document describes the API implemented in [`src/core.py`](src/core.py).
The library computes balanced-truncation reductions for stable discrete-time
linear state-space models, with support for general and diagonal state
transitions and adapters for compatible PyTorch modules.

## Installation

```bash
python -m pip install compreSSM-library
```

The package requires Python 3.10 or later. Its public module is named `core`:

```python
from core import CompressionConfig, DiscreteSSM, compress_numpy
```

For development, install the test extra and run the test suite:

```bash
python -m pip install ".[test]"
python -m pytest
```

## State-space model

The implementation uses the discrete-time system

```text
x[k + 1] = A x[k] + B u[k]
y[k]     = C x[k] + D u[k]
```

Here, `x` is the state, `u` the input, and `y` the output. `A` has shape
`(n, n)`, `B` has shape `(n, m)`, `C` has shape `(p, n)`, and `D` describes
direct input-to-output feedthrough. Matrices may be real or complex.

For a diagonal system, `A` is represented by its diagonal entries `lam`, a
one-dimensional array of length `n`. The other matrix dimensions are unchanged.

The reduction uses controllability and observability Gramians and the Hankel
singular values (HSVs) derived from them. It keeps the smallest leading set of
HSVs whose sum reaches the configured retention threshold. The reduced `D`
matrix is always the original `D`.

## Configuration and schedules

### `CompressionConfig`

Pydantic model that sets options for a compression operation.

```python
CompressionConfig(dt=1.0, tol=0.01, max_ratio=0.95)
```

Fields:

- `dt`: Positive sample interval, used when converting a continuous-time
  diagonal system or extracting one from the supported PyTorch module format.
  Defaults to `1.0`.
- `tol`: Fraction of the total HSV sum allowed to be discarded, in `[0, 1]`.
  The retained fraction is `1 - tol`. Defaults to `0.01`.
- `max_ratio`: Upper bound on the ratio of reduced to original state count.
  A reduction is returned only when `0 < r < max_ratio * n`. Defaults to
  `0.95`.

#### `CompressionConfig.threshold`

Read-only property returning `1 - tol`, the fraction of total HSV mass that
rank selection must retain.

### `ReductionSchedule`

Pydantic model describing the step window and interval for scheduled
reductions. `red_start` and `red_end` are non-negative step indices and
`red_interval` must be positive.

#### `ReductionSchedule.fires_at(step)`

Returns `True` if `step` is within the half-open interval
`[red_start, red_end)` and `(step + 1)` is divisible by `red_interval`.
Otherwise returns `False`.

For example, `ReductionSchedule(red_start=0, red_end=20_000,
red_interval=5_000)` fires at steps `4999`, `9999`, `14999`, and `19999`.

## System representations

### `DiscreteSSM`

Frozen dataclass containing dense state-space matrices `A`, `B`, `C`, and `D`.
The `n` property returns the state count (`A.shape[0]`).

#### `DiscreteSSM.from_numpy(*, A, B, C, D)`

Converts the supplied values to NumPy arrays and constructs a `DiscreteSSM`.
It checks that:

- `A` is a non-empty square matrix.
- `B` has one row per state and `C` has one column per state.
- All four arrays contain finite values.
- `A` is strictly stable, with spectral radius less than one.

Invalid input raises `ValueError`. This constructor does not separately
validate the output/input dimensions of `D` against `B` and `C`.

#### `DiscreteSSM.from_continuous_diagonal_numpy(*, a, B, C, D, dt)`

Discretizes a continuous-time diagonal system using zero-order hold (ZOH).
`a` is a non-empty one-dimensional array of continuous-time rates; each rate
must have a strictly negative real part. `dt` must be finite and positive. The
discrete diagonal is `exp(dt * a)`, and the input matrix is scaled by the ZOH
integral factor `(exp(dt * a) - 1) / a`. Zero rates are rejected by validation.
The result is checked through `DiscreteSSM.from_numpy`.

#### `DiscreteSSM.from_tensors(*, A_log, B, C, D, dt)`

Converts the expected PyTorch SSM tensors to NumPy and creates a discrete
system. It interprets continuous-time rates as `a = -exp(A_log)`, reshapes `B`
to `(n, -1)` and `C` to `(-1, n)`, and delegates discretization and validation
to `from_continuous_diagonal_numpy`. Tensors are detached and moved to CPU
before conversion.

#### `DiscreteSSM.from_module(module, *, dt)`

Extracts tensors named `ssm.A_log`, `ssm.B`, `ssm.C`, and `ssm.D` from
`module.state_dict()` and passes them to `from_tensors`. The module must expose
those exact state-dictionary keys. This operation only reads the module.

### `DiagonalDiscreteSSM`

Frozen dataclass for systems whose state transition is diagonal. Fields are:

- `lam`: Discrete eigenvalues, shape `(n,)`.
- `B`: Input matrix, shape `(n, m)`.
- `C`: Output matrix, shape `(p, n)`.
- `D`: Direct-feedthrough matrix or value.

The `n` property returns the number of eigenvalues.

#### `DiagonalDiscreteSSM.from_numpy(*, lam, B, C, D)`

Converts the inputs to NumPy arrays and validates the diagonal state count,
matrix dimensions, finite values, and strict stability (`abs(lam) < 1` for
every eigenvalue). Invalid inputs raise `ValueError`. `D` is checked for
finite values, but its dimensions are not independently checked against `B`
and `C`.

#### `DiagonalDiscreteSSM.from_lru_numpy(*, nu_log, theta_log, B_re, B_im, gamma_log, C_re, C_im, D)`

Builds a diagonal system from the expected LRU parameter arrays. It computes
the eigenvalues as

```text
lam = exp(-exp(nu_log) + 1j * exp(theta_log))
```

and reconstructs complex `B` from `B_re`, `B_im`, and the per-state scale
`exp(gamma_log)`. It constructs complex `C` from `C_re` and `C_im`, then uses
`from_numpy` for validation. The parameter arrays must have compatible shapes.

#### `DiagonalDiscreteSSM.from_lru_module(module)`

Reads named parameters `nu_log`, `theta_log`, `B_re`, `B_im`, `gamma_log`,
`C_re`, `C_im`, and `D` from a PyTorch module, converts them to NumPy, and
delegates to `from_lru_numpy`. The module must expose all of those parameter
names. This operation does not mutate the module.

### Reduction result dataclasses

#### `ReducedSSM`

Contains a dense reduced system: `A`, `B`, `C`, and unchanged `D`, together
with `r` (the reduced state count) and `error_bound` (the computed balanced
truncation bound).

#### `ReducedDiagonalSSM`

Contains a diagonal reduced system: `lam`, `B`, `C`, and unchanged `D`,
together with `r` and `error_bound`.

#### `Balancing`

Stores intermediate balancing results:

- `Lp`, `Lq`: Cholesky factors of the controllability and observability
  Gramians.
- `U`, `Vh`: Left and right singular vectors of `Lq.conj().T @ Lp`.
- `sigma`: Hankel singular values, in descending order.

These are primarily useful when inspecting or composing the lower-level
`CompreSSM` operations.

## Conversion helper

### `tensor_to_numpy(x)`

Returns a detached, CPU-resident copy of a PyTorch tensor as a NumPy array.
The copy is independent of the original tensor storage.

## `CompreSSM` numerical operations

`CompreSSM` is a namespace-style class containing static methods used by the
high-level compression functions. You can use these methods individually for
analysis or a custom reduction pipeline.

### `CompreSSM.calculate_gramians_diagonal(lam, *, B, C)`

Calculates controllability (`P`) and observability (`Q`) Gramians for a
diagonal system without constructing or solving a dense Lyapunov equation.
Returns `(P, Q)`. The results are explicitly Hermitian-symmetrized to reduce
floating-point asymmetry. The system should be stable.

### `CompreSSM.calculate_gramians(A, *, B, C)`

Solves the discrete Lyapunov equations for a dense system and returns
`(P, Q)`, the Hermitian-symmetrized controllability and observability Gramians.
The system matrix should be stable.

### `CompreSSM.diagonalize_reduced(Ar, Br, Cr)`

Computes an eigendecomposition of the reduced state matrix `Ar` and returns
`(lam_r, B_diag, C_diag)` transformed into diagonal-state coordinates. It uses
`np.linalg.solve` on the eigenvector matrix, so a defective or numerically
singular eigenvector matrix can cause a linear algebra error or poor numerical
results.

### `CompreSSM.cholesky_with_jitter(M, *, name, jitter=1e-12, max_tries=8)`

Attempts a lower-triangular Cholesky factorization of `M`. The first attempt
uses the original matrix; later attempts add progressively larger diagonal
jitter, scaled by the mean real trace of `M`. Returns the factor on success.
After all attempts fail, raises `numpy.linalg.LinAlgError` including `name`.

### `CompreSSM.calculate_sqrt(P, *, Q)`

Computes Cholesky factors for `P` and `Q` using `cholesky_with_jitter`, and
returns `(Lp, Lq)`. Despite the method name, the returned factors are
Cholesky factors, not general matrix square roots.

### `CompreSSM.HSVs(Lp, *, Lq)`

Computes and returns the singular value decomposition of `Lq.conj().T @ Lp`,
as provided by SciPy: `(U, sigma, Vh)`. The singular values `sigma` are the
Hankel singular values, ordered from largest to smallest.

### `CompreSSM.balance(P, *, Q)`

Combines Cholesky factorization and HSV computation. Returns a `Balancing`
instance containing `Lp`, `Lq`, `U`, `sigma`, and `Vh`.

### `CompreSSM.rank_selection(sigma, *, threshold)`

Returns the smallest rank whose cumulative HSV sum is at least
`threshold * sum(sigma)`. `sigma` must be a one-dimensional array of finite,
non-negative values, and `threshold` must be in `[0, 1]`; invalid inputs raise
`ValueError`. An empty HSV array returns `0`. If the total is zero or less,
the full length is returned. The function assumes the HSVs are already sorted
in descending order; it does not sort them.

### `CompreSSM.should_compress(r, n, *, max_ratio)`

Returns whether a reduction should be applied: `True` exactly when
`0 < r < max_ratio * n`. The upper comparison is strict.

### `CompreSSM.calculate_transform_matrix(Lp, *, r, sigma, Vh)`

Builds the balancing transform `T` for the first `r` singular directions and
returns `(T, diag_entries)`, where `diag_entries = sigma[:r] ** -0.5`. Raises
`numpy.linalg.LinAlgError` if any retained singular value is non-positive.

### `CompreSSM.calculate_projection_matrix(Lq, *, r, diag_entries, U)`

Builds and returns the matching projection matrix `S` from the observability
factor and the first `r` left singular vectors. With the corresponding `T`,
the projection is intended to satisfy `S @ T = I` up to numerical precision.

### `CompreSSM.truncate_A(A, *, S, T)`

Returns the reduced transition matrix `S @ A @ T`.

### `CompreSSM.truncate_B(S, *, B)`

Returns the reduced input matrix `S @ B`.

### `CompreSSM.truncate_C(C, *, T)`

Returns the reduced output matrix `C @ T`.

### `CompreSSM.truncate(A, B, C, *, balancing, r)`

Constructs the balancing transform and projection from `balancing`, applies
them to `A`, `B`, and `C`, and returns `(Ar, Br, Cr)`. `D` is not an argument
and is therefore handled unchanged by the higher-level compression functions.

## High-level compression functions

### `compress_from_gramians(A, B, C, *, P, Q, config)`

Balances a system using precomputed Gramians, selects a rank using
`config.threshold`, and checks whether the reduction meets `config.max_ratio`.
Returns `None` when the selected rank is not a worthwhile reduction. Otherwise
returns `(Ar, Br, Cr, r, error_bound)`, where the bound is twice the sum of the
discarded HSVs:

```text
error_bound = 2 * sum(sigma[r:])
```

The standard interpretation of this bound assumes the system is stable and
minimal.

### `compress_numpy(system, *, config)`

Computes Gramians for a `DiscreteSSM`, then applies balanced truncation. Returns
a `ReducedSSM`, or `None` when `max_ratio` says the reduction is not worthwhile.
The direct-feedthrough matrix `D` is copied unchanged. The reported error bound
is the standard balanced-truncation `H-infinity` bound when the input system
is stable and minimal.

### `compress_diagonal_numpy(system, *, config)`

Computes diagonal-system Gramians, applies balanced truncation, diagonalizes
the reduced transition matrix, and returns a `ReducedDiagonalSSM`. Returns
`None` when the reduction is not worthwhile. Preserves `D` unchanged. The
reported bound has the same stable, minimal system assumption as
`compress_numpy`.

### `compress_module(module, *, config)`

Reads the fixed-step diagonal SSM format from a PyTorch module using
`DiscreteSSM.from_module`, compresses it with `compress_numpy`, and returns a
`ReducedSSM` or `None`. It does not write parameters back to the module.
`config.dt` is used during continuous-to-discrete conversion.

### `compress_lru_module(module, *, config)`

Reads a diagonal LRU parameterization from a PyTorch module using
`DiagonalDiscreteSSM.from_lru_module` and compresses it. Returns a
`ReducedDiagonalSSM` or `None`. It does not mutate the module.

### `lru_arrays_from_diagonal(reduced)`

Converts a `ReducedDiagonalSSM` into a dictionary of NumPy arrays suitable for
writing into the supported LRU parameterization. The dictionary contains
`nu_log`, `theta_log`, `gamma_log`, `B_re`, `B_im`, `C_re`, and `C_im`.
Eigenvalue magnitude and phase are clipped away from singular logarithm
boundaries for conversion. The direct-feedthrough parameter `D` is not
included because it is unchanged.

### `replace_parameter(module, name, array)`

Replaces the named attribute on a PyTorch module with a new `nn.Parameter`
created from `array`. The new tensor is made contiguous and uses the existing
parameter's dtype, device, and `requires_grad` value. The module must already
have the named parameter; this function does not handle missing attributes.

### `write_back_lru(module, *, reduced)`

Converts `reduced` with `lru_arrays_from_diagonal` and replaces each matching
LRU parameter using `replace_parameter`. It mutates the module. The `D`
parameter remains untouched.

### `find_blocks(model, *, block_cls)`

Returns a list of all modules in `model.modules()` that are instances of
`block_cls`, including `model` itself if it matches.

### `reduce_lru_blocks(blocks, *, config)`

Compresses each module in `blocks` in order. When a block can be reduced,
updates that block in place using `write_back_lru`. Returns a list with one
entry per input block: a `ReducedDiagonalSSM` for blocks that were reduced, or
`None` for blocks for which no reduction was applied.

### `run_scheduled_reduction(*, step, schedule, blocks, config, make_optimizer)`

Checks whether `schedule.fires_at(step)` is true. If it is false, returns
`None` without processing blocks. If true, calls `reduce_lru_blocks`, which
may mutate reduced blocks. When no block was reduced, returns `None`. When at
least one block was reduced, calls `make_optimizer()` and returns the newly
created `torch.optim.Optimizer`. Rebuilding the optimizer allows callers to
account for changed parameter shapes.

## Errors and numerical considerations

- The system constructors reject unstable state transitions and non-finite
  state-space arrays.
- Gramian calculations require a stable system for the discrete Lyapunov
  equations to have the intended finite solution.
- Cholesky factorization may fail even after jitter, raising
  `numpy.linalg.LinAlgError`.
- Diagonalizing the reduced transition matrix requires a sufficiently
  well-conditioned eigenvector basis.
- Rank selection and compression use the HSV sum; `tol=0` requests retention
  of all HSV mass, which commonly means no reduction is returned.
- `D` is passed through rather than optimized or reduced.
- The error bound is a mathematical guarantee under the assumptions of
  stable, minimal LTI systems; finite precision and model violations can affect
  its practical interpretation.

## Citation

This implementation is based on:

> Makram Chahine, Philipp Nazari, Daniela Rus, and T. Konstantin Rusch.
> “The Curious Case of In-Training Compression of State Space Models.” 2026.
> [arXiv:2510.02823](https://arxiv.org/abs/2510.02823).

```bibtex
@misc{chahine2026curiouscaseintrainingcompression,
  title={The Curious Case of In-Training Compression of State Space Models},
  author={Makram Chahine and Philipp Nazari and Daniela Rus and T. Konstantin Rusch},
  year={2026},
  eprint={2510.02823},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.02823}
}
```