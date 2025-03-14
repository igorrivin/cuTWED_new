# cuTWED Documentation

## Introduction

cuTWED is a CUDA-accelerated implementation of the Time Warp Edit Distance algorithm, providing significant speedups compared to CPU implementations.

This documentation provides an overview of the library, its API, and usage examples.

## Table of Contents

1. [Installation](installation.md)
2. [API Reference](api-reference.md)
3. [Examples](examples.md)
4. [Algorithm Details](algorithm.md)
5. [Performance](performance.md)
6. [CuPy Integration](cupy-integration.md)
7. [Contributing](contributing.md)

## Quick Start

```python
import numpy as np
from cutwed import twed

# Create time series data
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
```

Using CuPy for GPU acceleration:

```python
import numpy as np
import cupy as cp
from cutwed import twed_cupy

# Create time series data on GPU
A = cp.array([1.0, 2.0, 3.0, 4.0])
TA = cp.array([0.0, 1.0, 2.0, 3.0])
B = cp.array([1.0, 3.0, 5.0])
TB = cp.array([0.0, 1.0, 2.0])

# Compute TWED distance using GPU
distance = twed_cupy(A, TA, B, TB, 1.0, 1.0, 2)
print(f"TWED distance: {distance}")
```

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.