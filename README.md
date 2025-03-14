# cuTWED v3.0.0

A linear memory algorithm for Time Warp Edit Distance with multiple backends

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/cutwed.svg)](https://badge.fury.io/py/cutwed)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## About

cuTWED is a fast implementation of the Time Warp Edit Distance algorithm, which measures similarity between time series. This version has been completely refactored to:

1. Support multiple backends (CUDA, CuPy, PyTorch, JAX, NumPy)
2. Improve code maintainability and robustness
3. Enhance compatibility with modern CUDA versions
4. Add modern Python packaging

The original algorithm was described by Marteau in "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching (2009)". 

## Features

- **Multiple Backends**: Works on various hardware (CPU, NVIDIA GPU, TPU) through different backend implementations:
  - CUDA (original C++ implementation, fastest on NVIDIA GPUs)
  - CuPy (CUDA-accelerated Python, easy to use on NVIDIA GPUs)
  - PyTorch (works on CPU or GPU, integrates with deep learning workflows)
  - JAX (works on CPU, GPU, or TPU, with JIT compilation)
  - NumPy (reference implementation, works everywhere)

- **Linear Memory**: Uses O(n) memory instead of O(n²)
- **Massive Parallelization**: Exploits GPU cores for significant speedups
- **Batch Processing**: Efficiently computes distance matrices
- **Unified API**: Consistent interface across all backends

## Quick Start

### Installation

From PyPI (CPU-only version):

```bash
pip install cutwed
```

For GPU acceleration, install with the appropriate extra:

```bash
# For CUDA backend (requires CUDA toolkit)
pip install cutwed[cuda]

# For CuPy backend
pip install cutwed[cupy]

# For PyTorch backend
pip install cutwed[torch]

# For JAX backend
pip install cutwed[jax]

# For all backends
pip install cutwed[all]
```

### Usage

```python
import numpy as np
from cutwed import twed, set_backend

# Create two time series
A = np.array([1.0, 2.0, 3.0, 4.0])
TA = np.array([0.0, 1.0, 2.0, 3.0])
B = np.array([1.0, 3.0, 5.0])
TB = np.array([0.0, 1.0, 2.0])

# Parameters
nu = 1.0      # Elasticity parameter
lamb = 1.0    # Stiffness parameter
degree = 2    # LP-norm degree (typically 2 for Euclidean)

# Compute TWED distance
distance = twed(A, TA, B, TB, nu, lamb, degree)
print(f"TWED distance: {distance}")

# Explicitly select a backend
set_backend('numpy')  # Options: 'numpy', 'pytorch', 'jax', 'cupy', 'cuda'
distance = twed(A, TA, B, TB, nu, lamb, degree)
```

### Batch Processing

```python
import numpy as np
from cutwed import twed_batch, TriangleOpt

# Create batch time series data (10 series of length 100 with 3 dimensions)
AA = np.random.randn(10, 100, 3)
TAA = np.tile(np.arange(100), (10, 1))
BB = np.random.randn(5, 100, 3)
TBB = np.tile(np.arange(100), (5, 1))

# Compute distance matrix (10x5)
distances = twed_batch(AA, TAA, BB, TBB, nu=1.0, lamb=1.0, degree=2)
print(f"Distance matrix shape: {distances.shape}")
```

### Backends with GPU Support

```python
# PyTorch backend with GPU
import torch
from cutwed import twed, set_backend

set_backend('pytorch')
A = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
TA = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
B = torch.tensor([1.0, 3.0, 5.0], device='cuda')
TB = torch.tensor([0.0, 1.0, 2.0], device='cuda')

distance = twed(A, TA, B, TB, nu=1.0, lamb=1.0, degree=2)
```

```python
# CuPy backend
import cupy as cp
from cutwed import twed, set_backend

set_backend('cupy')
A = cp.array([1.0, 2.0, 3.0, 4.0])
TA = cp.array([0.0, 1.0, 2.0, 3.0])
B = cp.array([1.0, 3.0, 5.0])
TB = cp.array([0.0, 1.0, 2.0])

distance = twed(A, TA, B, TB, nu=1.0, lamb=1.0, degree=2)
```

## Building from Source

### Prerequisites

- CUDA Toolkit 10.0 or later (for CUDA/CuPy backends)
- CMake 3.18 or later
- C++14 compatible compiler
- Python 3.7 or later
- NumPy

### Build Process

```bash
# Clone the repository
git clone https://github.com/garrettwrong/cuTWED.git
cd cuTWED/refactored

# Create a build directory
mkdir build
cd build

# Configure the build
cmake ..

# Build the library
make -j

# Install
make install

# Build and install Python package
pip install -e ..
```

## Google Colab

You can try cuTWED directly in Google Colab without installing anything:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garrettwrong/cuTWED/blob/master/refactored/examples/cutwed_colab_demo.ipynb)

## Documentation

For detailed documentation, please refer to:

- [API Reference](docs/api-reference.md)
- [Architecture Overview](docs/architecture.md)
- [Installation Guide](docs/installation.md)

## Performance

cuTWED achieves significant speedups compared to CPU implementations. With GPU acceleration:

- Single-precision (float32) computations can be 10-100x faster than CPU
- Batch processing can handle hundreds of time series efficiently
- Memory usage scales linearly with input size, not quadratically

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

GPLv3

Copyright 2020-2025 cuTWED Contributors