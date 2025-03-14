#!/bin/bash
set -e

# Display CUDA and Python information
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Python version: $(python3 --version)"
echo "CuPy version: $(python3 -c 'import cupy; print(cupy.__version__)')"

# Print installation information
echo "cuTWED is installed at: /opt/cuTWED"
echo "Library path: /opt/cuTWED/lib"
echo "Python path: /opt/cuTWED/python"

# Run the provided command
exec "$@"