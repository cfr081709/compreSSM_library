# CompreSSM Library

CompreSSM provides balanced-truncation tools for compressing stable discrete-time
state-space models, including diagonal systems used by Linear Recurrent Units
(LRUs). It computes controllability and observability Gramians, Hankel singular
values, reduced systems, and an error bound for each reduction.

## Installation

```bash
python -m pip install compressm
```

The package requires Python 3.10 or later. PyTorch is a runtime dependency
because the module also provides helpers for extracting and updating PyTorch
modules.

## Quick start

The published module is named `core`:

```python
import numpy as np

from core import (
    CompressionConfig,
    DiagonalDiscreteSSM,
    compress_diagonal_numpy,
)

system = DiagonalDiscreteSSM.from_numpy(
    lam=np.array([0.5, 0.8], dtype=complex),
    B=np.array([[1.0], [0.01]], dtype=complex),
    C=np.array([[1.0, 0.01]], dtype=complex),
    D=np.array([0.0]),
)

reduced = compress_diagonal_numpy(
    system,
    config=CompressionConfig(tol=0.05, max_ratio=0.95),
)

if reduced is None:
    print("The requested tolerance does not justify reducing this system.")
else:
    print(f"Reduced state dimension: {reduced.r} (from {system.n})")
    print(f"Balanced-truncation error bound: {reduced.error_bound}")
```

`tol` is the fraction of total Hankel singular value mass that may be discarded.
`max_ratio` prevents accepting a reduction when the resulting state count is not
small enough compared with the original system.

## Main API

- `DiscreteSSM` and `compress_numpy` handle general stable discrete-time systems.
- `DiagonalDiscreteSSM` and `compress_diagonal_numpy` handle diagonal systems,
  including LRU parameterizations.
- `CompressionConfig` controls the sample time, retained Hankel singular value
  mass, and minimum worthwhile reduction.
- `compress_module` and `compress_lru_module` adapt compatible PyTorch modules.
- `run_scheduled_reduction` applies reduction to blocks on a training schedule.

The `D` (direct-feedthrough) matrix is retained unchanged during reduction. The
reported balanced-truncation error bound assumes a stable, minimal system.

For details on every class and function, see the [full API documentation](docs.md).

## Development

Install the package and test dependencies, then run the test suite:

```bash
python -m pip install ".[test]"
python -m pytest
```

## Citation

This library is based on:

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
