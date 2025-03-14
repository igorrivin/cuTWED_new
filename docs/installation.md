# Installation Guide

This guide provides instructions for installing the cuTWED library and its Python bindings.

## Prerequisites

- CUDA Toolkit 10.0 or later
- CMake 3.18 or later
- C++14 compatible compiler
- Python 3.7 or later (for Python bindings)
- NumPy (for Python bindings)
- CuPy (optional, for enhanced GPU integration)

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

### Getting Help

If you encounter any issues during installation, please open an issue on GitHub with a detailed description of the problem and the steps to reproduce it.