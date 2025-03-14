# API Reference

This document describes the API for the cuTWED library, including both the C++ and Python interfaces.

## C++ API

The C++ API is implemented as a template class in the `cutwed` namespace.

### cutwed::TWED<T>

Template class for computing Time Warp Edit Distance, where `T` is the precision type (either `float` or `double`).

#### Static Methods

##### compute()

```cpp
static T compute(
    const T* A, int nA, const T* TA,
    const T* B, int nB, const T* TB,
    T nu, T lambda, int degree, int dim
);
```

Computes the TWED distance between two time series.

**Parameters:**
- `A`: First time series values (nA x dim)
- `nA`: Length of first time series
- `TA`: First time series timestamps (nA)
- `B`: Second time series values (nB x dim)
- `nB`: Length of second time series
- `TB`: Second time series timestamps (nB)
- `nu`: Elasticity parameter
- `lambda`: Stiffness parameter
- `degree`: LP-norm degree (default=2)
- `dim`: Dimensionality of time series points

**Returns:** The TWED distance

##### compute_dev()

```cpp
static T compute_dev(
    const T* A_dev, int nA, const T* TA_dev,
    const T* B_dev, int nB, const T* TB_dev,
    T nu, T lambda, int degree, int dim
);
```

Computes the TWED distance between two time series in device memory.

**Parameters:** Same as `compute()`, but pointers should reference device memory

**Returns:** The TWED distance

##### batch_compute()

```cpp
static int batch_compute(
    const T* AA, int nA, const T* TAA,
    const T* BB, int nB, const T* TBB,
    T nu, T lambda, int degree, int dim,
    int nAA, int nBB, T* RRes, TriangleOpt tri = TriangleOpt::NOPT
);
```

Computes TWED distances between multiple time series.

**Parameters:**
- `AA`: Batch of first time series (nAA x nA x dim)
- `nA`: Length of first time series
- `TAA`: Batch of first timestamps (nAA x nA)
- `BB`: Batch of second time series (nBB x nB x dim)
- `nB`: Length of second time series
- `TBB`: Batch of second timestamps (nBB x nB)
- `nu`: Elasticity parameter
- `lambda`: Stiffness parameter
- `degree`: LP-norm degree (default=2)
- `dim`: Dimensionality of time series points
- `nAA`: Number of time series in first batch
- `nBB`: Number of time series in second batch
- `RRes`: Result matrix (nAA x nBB), must be pre-allocated
- `tri`: Triangle optimization option (default=NOPT)

**Returns:** Error code (0 for success)

##### batch_compute_dev()

```cpp
static int batch_compute_dev(
    const T* AA_dev, int nA, const T* TAA_dev,
    const T* BB_dev, int nB, const T* TBB_dev,
    T nu, T lambda, int degree, int dim,
    int nAA, int nBB, T* RRes, TriangleOpt tri = TriangleOpt::NOPT
);
```

Computes TWED distances between multiple time series in device memory.

**Parameters:** Same as `batch_compute()`, but pointers should reference device memory

**Returns:** Error code (0 for success)

##### malloc_dev()

```cpp
static void malloc_dev(
    int nA, T** A_dev, T** TA_dev,
    int nB, T** B_dev, T** TB_dev,
    int dim, int nAA = 1, int nBB = 1
);
```

Allocates device memory for TWED computation.

**Parameters:**
- `nA`: Length of first time series
- `A_dev`: Pointer to device memory for first time series
- `TA_dev`: Pointer to device memory for first timestamps
- `nB`: Length of second time series
- `B_dev`: Pointer to device memory for second time series
- `TB_dev`: Pointer to device memory for second timestamps
- `dim`: Dimensionality of time series points
- `nAA`: Number of time series in first batch (default=1)
- `nBB`: Number of time series in second batch (default=1)

##### free_dev()

```cpp
static void free_dev(
    T* A_dev, T* TA_dev,
    T* B_dev, T* TB_dev
);
```

Frees device memory allocated by `malloc_dev()`.

**Parameters:**
- `A_dev`: Device memory for first time series
- `TA_dev`: Device memory for first timestamps
- `B_dev`: Device memory for second time series
- `TB_dev`: Device memory for second timestamps

##### copy_to_dev()

```cpp
static void copy_to_dev(
    int nA, const T* A, T* A_dev, const T* TA, T* TA_dev,
    int nB, const T* B, T* B_dev, const T* TB, T* TB_dev,
    int dim, int nAA = 1, int nBB = 1
);
```

Copies host data to device memory.

**Parameters:**
- `nA`: Length of first time series
- `A`: Host memory for first time series
- `A_dev`: Device memory for first time series
- `TA`: Host memory for first timestamps
- `TA_dev`: Device memory for first timestamps
- `nB`: Length of second time series
- `B`: Host memory for second time series
- `B_dev`: Device memory for second time series
- `TB`: Host memory for second timestamps
- `TB_dev`: Device memory for second timestamps
- `dim`: Dimensionality of time series points
- `nAA`: Number of time series in first batch (default=1)
- `nBB`: Number of time series in second batch (default=1)

### C-style Interface

For backward compatibility, the following C-style functions are provided:

#### Double Precision Functions

```cpp
double twed(double A[], int nA, double TA[],
            double B[], int nB, double TB[],
            double nu, double lambda, int degree, int dim);
```

```cpp
double twed_dev(double A_dev[], int nA, double TA_dev[],
                double B_dev[], int nB, double TB_dev[],
                double nu, double lambda, int degree, int dim);
```

```cpp
int twed_batch(double AA[], int nA, double TAA[],
               double BB[], int nB, double TBB[],
               double nu, double lambda, int degree, int dim,
               int nAA, int nBB, double* RRes, int tri);
```

```cpp
int twed_batch_dev(double AA_dev[], int nA, double TAA_dev[],
                   double BB_dev[], int nB, double TBB_dev[],
                   double nu, double lambda, int degree, int dim,
                   int nAA, int nBB, double* RRes, int tri);
```

```cpp
void twed_malloc_dev(int nA, double **A_dev, double **TA_dev,
                     int nB, double **B_dev, double **TB_dev,
                     int dim, int nAA, int nBB);
```

```cpp
void twed_free_dev(double *A_dev, double *TA_dev,
                   double *B_dev, double *TB_dev);
```

```cpp
void twed_copy_to_dev(int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                      int nB, double B[], double B_dev[], double TB[], double TB_dev[],
                      int dim, int nAA, int nBB);
```

#### Single Precision Functions

The same functions are available with `float` instead of `double`, with an `f` suffix (e.g., `twedf`, `twed_devf`, etc.).

## Python API

The Python API provides a simple interface to the C++ library.

### Functions

#### twed()

```python
def twed(A, TA, B, TB, nu, lamb, degree=2):
    """
    Compute Time Warp Edit Distance between two time series.
    
    Parameters
    ----------
    A : array_like
        First time series values, shape (nA, dim) or (nA,)
    TA : array_like
        First time series timestamps, shape (nA,)
    B : array_like
        Second time series values, shape (nB, dim) or (nB,)
    TB : array_like
        Second time series timestamps, shape (nB,)
    nu : float
        Elasticity parameter
    lamb : float
        Stiffness parameter
    degree : int, optional
        Power used in the Lp norm, default is 2
        
    Returns
    -------
    float
        The TWED distance
    """
```

#### twed_batch()

```python
def twed_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=TriangleOpt.NOPT):
    """
    Compute batch Time Warp Edit Distances between multiple time series.
    
    Parameters
    ----------
    AA : array_like
        Batch of first time series, shape (nAA, nA, dim) or (nAA, nA)
    TAA : array_like
        Batch of first timestamps, shape (nAA, nA)
    BB : array_like
        Batch of second time series, shape (nBB, nB, dim) or (nBB, nB)
    TBB : array_like
        Batch of second timestamps, shape (nBB, nB)
    nu : float
        Elasticity parameter
    lamb : float
        Stiffness parameter
    degree : int, optional
        Power used in the Lp norm, default is 2
    tri : TriangleOpt, optional
        Triangle optimization option, default is NOPT
        
    Returns
    -------
    ndarray
        Distance matrix of shape (nAA, nBB)
    """
```

### CuPy Integration

If CuPy is installed, the following functions are available:

#### twed_cupy()

```python
def twed_cupy(A, TA, B, TB, nu, lamb, degree=2):
    """
    CuPy-accelerated version of TWED.
    
    Parameters
    ----------
    Same as twed function
    
    Returns
    -------
    float
        The TWED distance
    """
```

#### twed_batch_cupy()

```python
def twed_batch_cupy(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=TriangleOpt.NOPT):
    """
    CuPy-accelerated version of batch TWED.
    
    Parameters
    ----------
    Same as twed_batch function
    
    Returns
    -------
    ndarray
        Distance matrix
    """
```

### Enumerations

#### TriangleOpt

```python
class TriangleOpt(IntEnum):
    """Triangle optimization options for batch processing."""
    TRIU = -2  # Upper triangular optimization
    TRIL = -1  # Lower triangular optimization
    NOPT = 0   # No optimization
```