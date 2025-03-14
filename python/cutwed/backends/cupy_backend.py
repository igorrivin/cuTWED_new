"""CuPy implementation of the TWED algorithm."""

import numpy as np
try:
    import cupy as cp
except ImportError:
    raise ImportError("CuPy backend requires CuPy to be installed. Please install it with: pip install cupy")


# Kernels for TWED computation
_lpnorm_kernel = cp.RawKernel(r'''
extern "C" __global__
void lpnorm_kernel(const float* X, int n, int dim, int degree, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float val = fabsf(X[tid * dim + d]);
        sum += powf(val, (float)degree);
    }
    
    result[tid] = powf(sum, 1.0f / (float)degree);
}
''', 'lpnorm_kernel')

_lpnorm_kernel_double = cp.RawKernel(r'''
extern "C" __global__
void lpnorm_kernel_double(const double* X, int n, int dim, int degree, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) {
        double val = fabs(X[tid * dim + d]);
        sum += pow(val, (double)degree);
    }
    
    result[tid] = pow(sum, 1.0 / (double)degree);
}
''', 'lpnorm_kernel_double')

_twed_kernel = cp.RawKernel(r'''
extern "C" __global__
void twed_kernel(const float* A, const float* TA, int nA, 
                 const float* B, const float* TB, int nB,
                 int dim, float nu, float lambda, int degree,
                 float* dp) {
    // Initialize dp matrix
    dp[0] = 0.0f;
    
    // Fill first row and column with infinity
    for (int i = 1; i <= nA; ++i) {
        dp[i * (nB + 1)] = INFINITY;
    }
    for (int j = 1; j <= nB; ++j) {
        dp[j] = INFINITY;
    }
    
    // Precompute local costs
    float da[10000];  // Assuming max nA is 10000
    float db[10000];  // Assuming max nB is 10000
    
    // Initialize da
    for (int i = 1; i <= nA; ++i) {
        if (i == 1) {
            // Calculate norm of A[0]
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float val = fabsf(A[d]);
                sum += powf(val, (float)degree);
            }
            da[i] = powf(sum, 1.0f / (float)degree);
        } else {
            // Calculate norm of A[i-1] - A[i-2]
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float val = fabsf(A[(i-1)*dim + d] - A[(i-2)*dim + d]);
                sum += powf(val, (float)degree);
            }
            da[i] = powf(sum, 1.0f / (float)degree);
        }
    }
    
    // Initialize db
    for (int j = 1; j <= nB; ++j) {
        if (j == 1) {
            // Calculate norm of B[0]
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float val = fabsf(B[d]);
                sum += powf(val, (float)degree);
            }
            db[j] = powf(sum, 1.0f / (float)degree);
        } else {
            // Calculate norm of B[j-1] - B[j-2]
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float val = fabsf(B[(j-1)*dim + d] - B[(j-2)*dim + d]);
                sum += powf(val, (float)degree);
            }
            db[j] = powf(sum, 1.0f / (float)degree);
        }
    }
    
    // Fill dp matrix
    for (int i = 1; i <= nA; ++i) {
        for (int j = 1; j <= nB; ++j) {
            // Calculate cost between A[i-1] and B[j-1]
            float cost = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float val = fabsf(A[(i-1)*dim + d] - B[(j-1)*dim + d]);
                cost += powf(val, (float)degree);
            }
            cost = powf(cost, 1.0f / (float)degree);
            
            // Additional cost for alignment of previous elements
            if (i > 1 && j > 1) {
                float prev_cost = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float val = fabsf(A[(i-2)*dim + d] - B[(j-2)*dim + d]);
                    prev_cost += powf(val, (float)degree);
                }
                cost += powf(prev_cost, 1.0f / (float)degree);
            }
            
            // Time differences
            float htrans = fabsf(TA[i-1] - TB[j-1]);
            if (i > 1 && j > 1) {
                htrans += fabsf(TA[i-2] - TB[j-2]);
            }
            
            // Case 1: Match A[i-1] and B[j-1]
            float c1 = dp[(i-1) * (nB + 1) + (j-1)] + cost + nu * htrans;
            
            // Case 2: Delete A[i-1]
            float htrans2;
            if (i > 1) {
                htrans2 = TA[i-1] - TA[i-2];
            } else {
                htrans2 = TA[i-1];
            }
            
            float c2 = dp[(i-1) * (nB + 1) + j] + da[i] + lambda + nu * htrans2;
            
            // Case 3: Delete B[j-1]
            float htrans3;
            if (j > 1) {
                htrans3 = TB[j-1] - TB[j-2];
            } else {
                htrans3 = TB[j-1];
            }
            
            float c3 = dp[i * (nB + 1) + (j-1)] + db[j] + lambda + nu * htrans3;
            
            // Take the minimum
            dp[i * (nB + 1) + j] = fminf(c1, fminf(c2, c3));
        }
    }
}
''', 'twed_kernel')

_twed_kernel_double = cp.RawKernel(r'''
extern "C" __global__
void twed_kernel_double(const double* A, const double* TA, int nA, 
                        const double* B, const double* TB, int nB,
                        int dim, double nu, double lambda, int degree,
                        double* dp) {
    // Initialize dp matrix
    dp[0] = 0.0;
    
    // Fill first row and column with infinity
    for (int i = 1; i <= nA; ++i) {
        dp[i * (nB + 1)] = INFINITY;
    }
    for (int j = 1; j <= nB; ++j) {
        dp[j] = INFINITY;
    }
    
    // Precompute local costs
    double da[10000];  // Assuming max nA is 10000
    double db[10000];  // Assuming max nB is 10000
    
    // Initialize da
    for (int i = 1; i <= nA; ++i) {
        if (i == 1) {
            // Calculate norm of A[0]
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = fabs(A[d]);
                sum += pow(val, (double)degree);
            }
            da[i] = pow(sum, 1.0 / (double)degree);
        } else {
            // Calculate norm of A[i-1] - A[i-2]
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = fabs(A[(i-1)*dim + d] - A[(i-2)*dim + d]);
                sum += pow(val, (double)degree);
            }
            da[i] = pow(sum, 1.0 / (double)degree);
        }
    }
    
    // Initialize db
    for (int j = 1; j <= nB; ++j) {
        if (j == 1) {
            // Calculate norm of B[0]
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = fabs(B[d]);
                sum += pow(val, (double)degree);
            }
            db[j] = pow(sum, 1.0 / (double)degree);
        } else {
            // Calculate norm of B[j-1] - B[j-2]
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = fabs(B[(j-1)*dim + d] - B[(j-2)*dim + d]);
                sum += pow(val, (double)degree);
            }
            db[j] = pow(sum, 1.0 / (double)degree);
        }
    }
    
    // Fill dp matrix
    for (int i = 1; i <= nA; ++i) {
        for (int j = 1; j <= nB; ++j) {
            // Calculate cost between A[i-1] and B[j-1]
            double cost = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = fabs(A[(i-1)*dim + d] - B[(j-1)*dim + d]);
                cost += pow(val, (double)degree);
            }
            cost = pow(cost, 1.0 / (double)degree);
            
            // Additional cost for alignment of previous elements
            if (i > 1 && j > 1) {
                double prev_cost = 0.0;
                for (int d = 0; d < dim; ++d) {
                    double val = fabs(A[(i-2)*dim + d] - B[(j-2)*dim + d]);
                    prev_cost += pow(val, (double)degree);
                }
                cost += pow(prev_cost, 1.0 / (double)degree);
            }
            
            // Time differences
            double htrans = fabs(TA[i-1] - TB[j-1]);
            if (i > 1 && j > 1) {
                htrans += fabs(TA[i-2] - TB[j-2]);
            }
            
            // Case 1: Match A[i-1] and B[j-1]
            double c1 = dp[(i-1) * (nB + 1) + (j-1)] + cost + nu * htrans;
            
            // Case 2: Delete A[i-1]
            double htrans2;
            if (i > 1) {
                htrans2 = TA[i-1] - TA[i-2];
            } else {
                htrans2 = TA[i-1];
            }
            
            double c2 = dp[(i-1) * (nB + 1) + j] + da[i] + lambda + nu * htrans2;
            
            // Case 3: Delete B[j-1]
            double htrans3;
            if (j > 1) {
                htrans3 = TB[j-1] - TB[j-2];
            } else {
                htrans3 = TB[j-1];
            }
            
            double c3 = dp[i * (nB + 1) + (j-1)] + db[j] + lambda + nu * htrans3;
            
            // Take the minimum
            dp[i * (nB + 1) + j] = fmin(c1, fmin(c2, c3));
        }
    }
}
''', 'twed_kernel_double')


def lpnorm(x, p=2, axis=-1):
    """Compute the Lp norm along the specified axis using CuPy."""
    return cp.power(cp.sum(cp.power(cp.abs(x), p), axis=axis), 1/p)


def twed(A, TA, B, TB, nu, lamb, degree=2):
    """Compute Time Warp Edit Distance between two time series using CuPy.
    
    Args:
        A: First time series values, shape (nA, dim) or (nA,)
        TA: First time series timestamps, shape (nA,)
        B: Second time series values, shape (nB, dim) or (nB,)
        TB: Second time series timestamps, shape (nB,)
        nu: Elasticity parameter
        lamb: Stiffness parameter
        degree: Power used in the Lp norm, default is 2
        
    Returns:
        The TWED distance as a scalar
    """
    # Convert inputs to CuPy arrays
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)
    if not isinstance(TA, cp.ndarray):
        TA = cp.asarray(TA)
    if not isinstance(B, cp.ndarray):
        B = cp.asarray(B)
    if not isinstance(TB, cp.ndarray):
        TB = cp.asarray(TB)
    
    # Ensure inputs have the same dtype
    dtype = cp.result_type(A.dtype, B.dtype, TA.dtype, TB.dtype)
    A = A.astype(dtype)
    TA = TA.astype(dtype)
    B = B.astype(dtype)
    TB = TB.astype(dtype)
    
    # Ensure A and B are 2D
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    # Check dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same dimensions, got {A.shape[1]} and {B.shape[1]}")
    
    # Get dimensions
    nA, dim = A.shape
    nB = B.shape[0]
    
    # Allocate dp matrix
    dp = cp.zeros((nA + 1) * (nB + 1), dtype=dtype)
    
    # Choose the appropriate kernel based on data type
    if dtype == cp.float64:
        kernel = _twed_kernel_double
    else:
        kernel = _twed_kernel
    
    # Launch kernel
    threads_per_block = 1
    blocks_per_grid = 1
    kernel((blocks_per_grid,), (threads_per_block,), 
           (A, TA, nA, B, TB, nB, dim, nu, lamb, degree, dp))
    
    # Return the final result
    return float(dp[(nA + 1) * (nB + 1) - 1])


def twed_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=0):
    """Compute batch Time Warp Edit Distances using CuPy.
    
    Args:
        AA: Batch of first time series, shape (nAA, nA, dim) or (nAA, nA)
        TAA: Batch of first timestamps, shape (nAA, nA)
        BB: Batch of second time series, shape (nBB, nB, dim) or (nBB, nB)
        TBB: Batch of second timestamps, shape (nBB, nB)
        nu: Elasticity parameter
        lamb: Stiffness parameter
        degree: Power used in the Lp norm, default is 2
        tri: Triangle optimization option (0: none, -1: lower, -2: upper)
        
    Returns:
        Distance matrix of shape (nAA, nBB)
    """
    # Convert inputs to CuPy arrays
    if not isinstance(AA, cp.ndarray):
        AA = cp.asarray(AA)
    if not isinstance(TAA, cp.ndarray):
        TAA = cp.asarray(TAA)
    if not isinstance(BB, cp.ndarray):
        BB = cp.asarray(BB)
    if not isinstance(TBB, cp.ndarray):
        TBB = cp.asarray(TBB)
    
    # Ensure inputs have the same dtype
    dtype = cp.result_type(AA.dtype, BB.dtype, TAA.dtype, TBB.dtype)
    AA = AA.astype(dtype)
    TAA = TAA.astype(dtype)
    BB = BB.astype(dtype)
    TBB = TBB.astype(dtype)
    
    # Ensure AA and BB are 3D
    if AA.ndim == 2:
        AA = AA.reshape(AA.shape[0], AA.shape[1], 1)
    if BB.ndim == 2:
        BB = BB.reshape(BB.shape[0], BB.shape[1], 1)
    
    # Get dimensions
    nAA, nA, dim = AA.shape
    nBB, nB = BB.shape[:2]
    
    # Initialize result matrix
    result = cp.zeros((nAA, nBB), dtype=dtype)
    
    # Determine which pairs to compute based on triangle optimization
    if tri == -1 and nAA == nBB:  # Lower triangular
        pairs = [(i, j) for i in range(nAA) for j in range(min(i+1, nBB))]
    elif tri == -2 and nAA == nBB:  # Upper triangular
        pairs = [(i, j) for i in range(nAA) for j in range(i, nBB)]
    else:  # Full matrix
        pairs = [(i, j) for i in range(nAA) for j in range(nBB)]
    
    # Compute distances
    for i, j in pairs:
        result[i, j] = twed(AA[i], TAA[i], BB[j], TBB[j], nu, lamb, degree)
        
        # For triangular optimization with symmetric inputs, fill the other half
        if tri != 0 and nAA == nBB and i != j:
            result[j, i] = result[i, j]
    
    return result