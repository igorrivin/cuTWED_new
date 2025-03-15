#!/usr/bin/env python
"""
Copyright 2022-2025, cuTWED Contributors

This file is part of cuTWED.

cuTWED is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
cuTWED is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with cuTWED.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import ctypes
from enum import IntEnum
import warnings

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found. GPU-accelerated operations will not be available.")

# Load the C library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/libcuTWED.so')

try:
    _lib = ctypes.CDLL(_lib_path)
except OSError:
    # Try to find the library in standard locations
    try:
        _lib = ctypes.CDLL('libcuTWED.so')
    except OSError as e:
        raise ImportError(f"Failed to load the cuTWED library. Make sure it is installed correctly. Error: {e}")

# Define C function prototypes
_lib.twed.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int
]
_lib.twed.restype = ctypes.c_double

_lib.twedf.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
]
_lib.twedf.restype = ctypes.c_float

_lib.twed_batch.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int
]
_lib.twed_batch.restype = ctypes.c_int

_lib.twed_batchf.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
_lib.twed_batchf.restype = ctypes.c_int


class TriangleOpt(IntEnum):
    """Triangle optimization options for batch processing."""
    TRIU = -2  # Upper triangular optimization
    TRIL = -1  # Lower triangular optimization
    NOPT = 0   # No optimization


def _prepare_time_series(A, TA, B, TB):
    """Prepare time series data for computation."""
    # Ensure input arrays are numpy arrays
    A = np.asarray(A, order='C')
    TA = np.asarray(TA, order='C')
    B = np.asarray(B, order='C')
    TB = np.asarray(TB, order='C')
    
    # Check dimensions and reshape as needed
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    elif A.ndim != 2:
        raise ValueError("A must be a 1D or 2D array")
    
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    elif B.ndim != 2:
        raise ValueError("B must be a 1D or 2D array")
    
    # Check consistency
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same number of dimensions, got {A.shape[1]} and {B.shape[1]}")
    
    if len(TA) != A.shape[0]:
        raise ValueError(f"TA must have the same length as A, got {len(TA)} and {A.shape[0]}")
    
    if len(TB) != B.shape[0]:
        raise ValueError(f"TB must have the same length as B, got {len(TB)} and {B.shape[0]}")
    
    # Check dtypes
    if A.dtype != B.dtype or A.dtype != TA.dtype or A.dtype != TB.dtype:
        common_dtype = np.result_type(A.dtype, B.dtype, TA.dtype, TB.dtype)
        if common_dtype == np.float64:
            A = A.astype(np.float64)
            B = B.astype(np.float64)
            TA = TA.astype(np.float64)
            TB = TB.astype(np.float64)
        else:
            A = A.astype(np.float32)
            B = B.astype(np.float32)
            TA = TA.astype(np.float32)
            TB = TB.astype(np.float32)
    
    return A, TA, B, TB


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
    # Prepare data
    A, TA, B, TB = _prepare_time_series(A, TA, B, TB)
    
    nA = A.shape[0]
    nB = B.shape[0]
    dim = A.shape[1]
    
    # Choose the appropriate function based on data type
    if A.dtype == np.float64:
        func = _lib.twed
        
        # Create C pointers
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        TA_ptr = TA.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        TB_ptr = TB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call the C function
        result = func(A_ptr, nA, TA_ptr, B_ptr, nB, TB_ptr, nu, lamb, degree, dim)
    else:
        func = _lib.twedf
        
        # Create C pointers
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        TA_ptr = TA.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        TB_ptr = TB.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        result = func(A_ptr, nA, TA_ptr, B_ptr, nB, TB_ptr, nu, lamb, degree, dim)
    
    if result < 0:
        raise RuntimeError(f"cuTWED computation failed with error code {result}")
    
    return result


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
    # Convert to numpy arrays
    AA = np.asarray(AA, order='C')
    TAA = np.asarray(TAA, order='C')
    BB = np.asarray(BB, order='C')
    TBB = np.asarray(TBB, order='C')
    
    # Reshape 2D inputs to 3D
    if AA.ndim == 2:
        AA = AA.reshape(AA.shape[0], AA.shape[1], 1)
    elif AA.ndim != 3:
        raise ValueError("AA must be a 2D or 3D array")
    
    if BB.ndim == 2:
        BB = BB.reshape(BB.shape[0], BB.shape[1], 1)
    elif BB.ndim != 3:
        raise ValueError("BB must be a 2D or 3D array")
    
    # Check dimensions
    nAA, nA, dim_A = AA.shape
    nBB, nB, dim_B = BB.shape
    
    if dim_A != dim_B:
        raise ValueError(f"AA and BB must have the same dimension, got {dim_A} and {dim_B}")
    
    if TAA.shape != (nAA, nA):
        raise ValueError(f"TAA must have shape ({nAA}, {nA}), got {TAA.shape}")
    
    if TBB.shape != (nBB, nB):
        raise ValueError(f"TBB must have shape ({nBB}, {nB}), got {TBB.shape}")
    
    # Check dtypes and convert if needed
    if (AA.dtype != BB.dtype or AA.dtype != TAA.dtype or AA.dtype != TBB.dtype):
        common_dtype = np.result_type(AA.dtype, BB.dtype, TAA.dtype, TBB.dtype)
        if common_dtype == np.float64:
            AA = AA.astype(np.float64)
            BB = BB.astype(np.float64)
            TAA = TAA.astype(np.float64)
            TBB = TBB.astype(np.float64)
        else:
            AA = AA.astype(np.float32)
            BB = BB.astype(np.float32)
            TAA = TAA.astype(np.float32)
            TBB = TBB.astype(np.float32)
    
    # Create result array
    result = np.zeros((nAA, nBB), dtype=AA.dtype)
    
    # Choose the appropriate function based on data type
    if AA.dtype == np.float64:
        func = _lib.twed_batch
        
        # Create C pointers
        AA_ptr = AA.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        TAA_ptr = TAA.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        BB_ptr = BB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        TBB_ptr = TBB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call the C function
        ret_code = func(AA_ptr, nA, TAA_ptr, BB_ptr, nB, TBB_ptr, 
                       nu, lamb, degree, dim_A, nAA, nBB, result_ptr, int(tri))
    else:
        func = _lib.twed_batchf
        
        # Create C pointers
        AA_ptr = AA.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        TAA_ptr = TAA.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        BB_ptr = BB.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        TBB_ptr = TBB.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        ret_code = func(AA_ptr, nA, TAA_ptr, BB_ptr, nB, TBB_ptr, 
                       nu, lamb, degree, dim_A, nAA, nBB, result_ptr, int(tri))
    
    if ret_code != 0:
        raise RuntimeError(f"cuTWED batch computation failed with error code {ret_code}")
    
    return result


# CuPy-accelerated implementations
if HAS_CUPY:
    # Load kernels for CuPy
    _lp_norm_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void lp_norm_kernel(const float* X, int n, int dim, int degree, float* result) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;
        
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float val = fabsf(X[tid * dim + d]);
            sum += powf(val, (float)degree);
        }
        
        result[tid] = powf(sum, 1.0f / (float)degree);
    }
    ''', 'lp_norm_kernel')
    
    _lp_norm_kernel_double = cp.RawKernel(r'''
    extern "C" __global__
    void lp_norm_kernel_double(const double* X, int n, int dim, int degree, double* result) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;
        
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double val = fabs(X[tid * dim + d]);
            sum += pow(val, (double)degree);
        }
        
        result[tid] = pow(sum, 1.0 / (double)degree);
    }
    ''', 'lp_norm_kernel_double')
    
    class TWEDCupy:
        """CuPy implementation of TWED algorithm."""
        
        @staticmethod
        def compute(A, TA, B, TB, nu, lamb, degree=2):
            """
            Compute TWED using CuPy.
            
            Parameters
            ----------
            Same as twed function
            
            Returns
            -------
            float
                The TWED distance
            """
            # Prepare data
            A, TA, B, TB = _prepare_time_series(A, TA, B, TB)
            
            # Convert to CuPy arrays
            if isinstance(A, np.ndarray):
                A_gpu = cp.asarray(A)
                TA_gpu = cp.asarray(TA)
                B_gpu = cp.asarray(B)
                TB_gpu = cp.asarray(TB)
            else:
                A_gpu = A
                TA_gpu = TA
                B_gpu = B
                TB_gpu = TB
            
            nA = A_gpu.shape[0]
            nB = B_gpu.shape[0]
            dim = A_gpu.shape[1]
            
            # Initialize DP matrices
            dp = cp.zeros((nA + 1, nB + 1), dtype=A_gpu.dtype)
            
            # Compute distances
            if A_gpu.dtype == cp.float64:
                # Use CUDA function through ctypes
                result = twed(A_gpu.get(), TA_gpu.get(), B_gpu.get(), TB_gpu.get(),
                            nu, lamb, degree)
            else:
                # Use CUDA function through ctypes
                result = twed(A_gpu.get(), TA_gpu.get(), B_gpu.get(), TB_gpu.get(),
                            nu, lamb, degree)
            
            return result
        
        @staticmethod
        def compute_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=TriangleOpt.NOPT):
            """
            Compute batch TWED using CuPy.
            
            Parameters
            ----------
            Same as twed_batch function
            
            Returns
            -------
            ndarray
                Distance matrix
            """
            # Convert to CuPy arrays if not already
            if isinstance(AA, np.ndarray):
                AA_gpu = cp.asarray(AA)
                TAA_gpu = cp.asarray(TAA)
                BB_gpu = cp.asarray(BB)
                TBB_gpu = cp.asarray(TBB)
            else:
                AA_gpu = AA
                TAA_gpu = TAA
                BB_gpu = BB
                TBB_gpu = TBB
            
            # Reshape 2D inputs to 3D
            if AA_gpu.ndim == 2:
                AA_gpu = AA_gpu.reshape(AA_gpu.shape[0], AA_gpu.shape[1], 1)
            
            if BB_gpu.ndim == 2:
                BB_gpu = BB_gpu.reshape(BB_gpu.shape[0], BB_gpu.shape[1], 1)
            
            # Get dimensions
            nAA, nA, dim_A = AA_gpu.shape
            nBB, nB, dim_B = BB_gpu.shape
            
            # Create result array
            result_gpu = cp.zeros((nAA, nBB), dtype=AA_gpu.dtype)
            
            # Convert back to numpy for C function
            if AA_gpu.dtype == cp.float64:
                # Use CUDA function through ctypes
                result = twed_batch(AA_gpu.get(), TAA_gpu.get(), BB_gpu.get(), TBB_gpu.get(),
                                  nu, lamb, degree, tri)
                return cp.asarray(result)
            else:
                # Use CUDA function through ctypes
                result = twed_batch(AA_gpu.get(), TAA_gpu.get(), BB_gpu.get(), TBB_gpu.get(),
                                  nu, lamb, degree, tri)
                return cp.asarray(result)
    
    
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
        return TWEDCupy.compute(A, TA, B, TB, nu, lamb, degree)
    
    
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
        return TWEDCupy.compute_batch(AA, TAA, BB, TBB, nu, lamb, degree, tri)