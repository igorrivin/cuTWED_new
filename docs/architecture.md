# Architecture

This document provides an overview of the cuTWED architecture and how the different components interact.

## Component Overview

The cuTWED library is organized as follows:

```
cuTWED/
├── include/               # Public headers
│   ├── cuTWED.hpp         # Main API definition
│   └── cuda_utils.hpp     # CUDA utility functions
├── src/                   # Implementation files
│   └── cuTWED.cu          # CUDA implementation
├── python/                # Python bindings
│   ├── cutwed.py          # Python API
│   └── setup.py           # Python package setup
└── tests/                 # Tests
    ├── test_cutwed.cu     # C++ tests
    └── python_test.py     # Python tests
```

## Architecture Diagram

```
+---------------------------+     +---------------------------+
|                           |     |                           |
|  C++ Application          |     |  Python Application       |
|                           |     |                           |
+-------------+-------------+     +-------------+-------------+
              |                                 |
              | Include                         | Import
              v                                 v
+-------------+-------------+     +-------------+-------------+
|                           |     |                           |
|  cuTWED.hpp               |     |  cutwed.py               |
|  (C++ API)                |     |  (Python API)            |
|                           |     |                           |
+-------------+-------------+     +-------------+-------------+
              |                                 |
              | Template-based                  | Foreign Function
              | Implementation                  | Interface (ctypes)
              v                                 v
+-------------+-------------+     +-------------+-------------+
|                           |     |                           |
|  cuTWED.cu                +<----+  libcuTWED.so            |
|  (CUDA Implementation)    |     |  (Shared Library)        |
|                           |     |                           |
+-------------+-------------+     +-------------+-------------+
              |                                 |
              | Uses                            | Optional
              v                                 v
+-------------+-------------+     +-------------+-------------+
|                           |     |                           |
|  cuda_utils.hpp           |     |  CuPy Integration        |
|  (CUDA Utilities)         |     |  (GPU Array Interface)   |
|                           |     |                           |
+---------------------------+     +---------------------------+
```

## Key Components

### C++ API (cuTWED.hpp)

- Defines the template class `TWED<T>` for both float and double precision
- Provides static methods for computing TWED distances
- Includes C-style interface for backward compatibility

### CUDA Implementation (cuTWED.cu)

- Implements the TWED algorithm using CUDA kernels
- Manages device memory and data transfers
- Handles batch processing and optimization

### CUDA Utilities (cuda_utils.hpp)

- Provides RAII wrappers for CUDA resources
- Handles error checking and exception management
- Simplifies memory management with templated classes

### Python API (cutwed.py)

- Provides a Pythonic interface to the C++ library
- Uses ctypes to interface with the shared library
- Supports NumPy arrays for input/output

### CuPy Integration

- Optional integration with CuPy for GPU array handling
- Provides seamless use of GPU memory through CuPy arrays
- Implements optimized batch processing for CuPy arrays

## Data Flow

1. The user provides time series data (either in host memory for C++ or as NumPy/CuPy arrays for Python)
2. The API validates the input data and prepares it for processing
3. For GPU computation, data is transferred to device memory (if not already there)
4. CUDA kernels compute the TWED distances in parallel
5. Results are transferred back to host memory (if needed)
6. The API returns the TWED distance(s) to the user

## Memory Management

The refactored implementation uses RAII (Resource Acquisition Is Initialization) principles to ensure proper management of CUDA resources:

- `DeviceMemory<T>`: Template class for managing device memory
- `CudaStream`: Class for managing CUDA streams
- `CublasHandle`: Class for managing cuBLAS handles

These classes automatically handle resource allocation and deallocation, preventing memory leaks and ensuring proper cleanup even in case of errors.

## Error Handling

The implementation uses a consistent error handling approach:

1. C++ exceptions are used within the C++ API
2. The C interface translates exceptions to error codes
3. The Python API converts error codes back to exceptions

This approach ensures that errors are properly propagated and can be handled at the appropriate level.