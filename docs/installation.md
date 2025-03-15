# Installation Guide

This guide provides instructions for installing the cuTWED library and its Python bindings.

## Quick Install

The easiest way to install cuTWED is via pip:

```bash
pip install cutwed
```

This will install the package with all CPU backends (NumPy, PyTorch, JAX). CUDA support is automatically enabled if an NVIDIA GPU is detected.

## Available Backends

cuTWED supports multiple backends:

1. `cuda`: Original CUDA C++ implementation (fastest, requires NVIDIA GPU)
2. `cupy`: CuPy implementation (fast, requires NVIDIA GPU)
3. `jax`: JAX implementation (works on CPU/GPU/TPU)
4. `pytorch`: PyTorch implementation (works on CPU/GPU)
5. `numpy`: NumPy implementation (works on CPU only)

The backend is automatically selected based on availability and performance, but you can manually select it:

```python
import cutwed
cutwed.set_backend('jax')  # Use JAX backend
```

## Prerequisites for Building from Source

- CMake 3.18 or later
- C++14 compatible compiler
- Python 3.7 or later (for Python bindings)
- NumPy (for Python bindings)
- CUDA Toolkit 10.0 or later (optional, for CUDA backend)

## Installing from Source

### Step 1: Clone the Repository

```bash
git clone https://github.com/garrettwrong/cuTWED.git
cd cuTWED/refactored
```

### Step 2: Configure with CMake

Create a build directory and configure the project:

```bash
mkdir build
cd build
cmake ..
```

You can customize the build with the following options:

- `-DCUTWED_BUILD_TESTS=ON/OFF`: Build tests (default: ON)
- `-DCUTWED_USE_DOUBLE=ON/OFF`: Use double precision (default: ON)
- `-DCUTWED_BUILD_PYTHON=ON/OFF`: Build Python bindings (default: ON)
- `-DCMAKE_INSTALL_PREFIX=/path/to/install`: Set installation directory

For example:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCUTWED_BUILD_TESTS=ON
```

### Step 3: Build and Install

Compile the library:

```bash
make -j
```

Install the library:

```bash
make install
```

This will install the library to the specified installation directory (or the default location if not specified).

### Step 4: Build and Install Python Bindings

After installing the C++ library, you can build and install the Python bindings:

```bash
make python_package
make python_install
```

Alternatively, you can use pip directly:

```bash
cd /path/to/install/python
pip install -e .
```

## Testing the Installation

### C++ Tests

Run the C++ tests to verify the installation:

```bash
cd build
ctest
```

### Python Tests

To run the Python tests:

```bash
cd tests
python -m pytest python_test.py -v
```

## Using with Conda

You can create a Conda environment with all the required dependencies:

```bash
conda create -n cutwed python=3.9 numpy pytest cmake
conda activate cutwed
conda install -c conda-forge cudatoolkit
```

If you want to use CuPy:

```bash
conda install -c conda-forge cupy
```

Then follow the installation steps above.

## Docker Installation

A Dockerfile is provided to create a containerized environment:

```bash
cd cuTWED/refactored/docker
docker build -t cutwed .
```

To run the container:

```bash
docker run -it --gpus all cutwed
```

## Installation Options

### Basic Installation (CPU only)

```bash
pip install cutwed
```

### With CUDA Support

For NVIDIA GPU support, you need to have CUDA installed. The build system will automatically detect NVIDIA GPUs on Linux systems.

To force CUDA compilation even if no GPU is detected:

```bash
FORCE_CUDA=1 pip install cutwed
```

### Install Optional Dependencies

To install with specific backend dependencies:

```bash
# Install with PyTorch support
pip install cutwed[torch]

# Install with JAX support
pip install cutwed[jax]

# Install with CuPy support
pip install cutwed[cupy]

# Install all backends
pip install cutwed[all]

# Install development dependencies
pip install cutwed[dev]
```

## Google Colab Installation

cuTWED works well in Google Colab. When using Colab with a GPU runtime, CUDA support is automatically enabled.

```python
!pip install cutwed

import cutwed
print(f"Available backends: {cutwed.get_available_backends()}")
print(f"Currently using: {cutwed.get_backend_name()}")
```

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

On Apple Silicon Macs, JAX is significantly faster than PyTorch for cuTWED operations, despite PyTorch having MPS (Metal Performance Shaders) support. The package will automatically prioritize JAX on these platforms.

If you get errors with JAX on Apple Silicon, make sure you're using the correct JAX version:

```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

### NVIDIA GPUs

For best performance on NVIDIA GPUs, ensure you have the latest CUDA drivers installed and use the CUDA backend:

```python
import cutwed
cutwed.set_backend('cuda')  # Use native CUDA implementation
```

## Troubleshooting

### Common Issues

#### Missing CUDA Toolkit

If CMake cannot find the CUDA Toolkit, make sure it is installed and the `CUDA_HOME` environment variable is set:

```bash
export CUDA_HOME=/path/to/cuda
```

#### Compilation Errors

If you encounter compilation errors, check that your CUDA Toolkit and compiler are compatible. The library requires CUDA 10.0 or later and a C++14 compatible compiler.

#### Python Import Errors

If you get errors importing the Python module, make sure the library is in your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/install/lib
```

#### Backend Availability

If you encounter issues with backend availability, check which backends are available:

```python
import cutwed
print(cutwed.get_available_backends())
```

If a backend is missing, install the corresponding dependencies:

- For PyTorch: `pip install torch`
- For JAX: `pip install jax jaxlib`
- For CuPy: `pip install cupy-cuda11x` (adjust for your CUDA version)

### Getting Help

If you encounter any issues during installation, please open an issue on GitHub with a detailed description of the problem and the steps to reproduce it.