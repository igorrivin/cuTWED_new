"""Wrapper for the CUDA C++ implementation of the TWED algorithm."""

import os
import ctypes
import numpy as np
from enum import IntEnum

# Try to load the shared library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../lib/libcuTWED.so')

try:
    _lib = ctypes.CDLL(_lib_path)
except OSError:
    # Try to find the library in standard locations
    try:
        _lib = ctypes.CDLL('libcuTWED.so')
    except OSError as e:
        raise ImportError(f"Failed to load the CUDA cuTWED library. Make sure it is installed correctly. Error: {e}")

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


def _prepare_arrays(A, TA, B, TB):
    """Prepare input arrays for CUDA computation."""
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
    
    # Make sure data types match
    dtype = np.result_type(A.dtype, B.dtype, TA.dtype, TB.dtype)
    if dtype == np.float64:
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
    """Compute Time Warp Edit Distance using the CUDA implementation.
    
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
    # Prepare inputs
    A, TA, B, TB = _prepare_arrays(A, TA, B, TB)
    
    # Get dimensions
    nA = A.shape[0]
    nB = B.shape[0]
    dim = A.shape[1]
    
    # Call the appropriate C function based on data type
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
        raise RuntimeError(f"CUDA TWED computation failed with error code {result}")
    
    return result


def _prepare_batch_arrays(AA, TAA, BB, TBB):
    """Prepare batch input arrays for CUDA computation."""
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
    dtype = np.result_type(AA.dtype, BB.dtype, TAA.dtype, TBB.dtype)
    if dtype == np.float64:
        AA = AA.astype(np.float64)
        BB = BB.astype(np.float64)
        TAA = TAA.astype(np.float64)
        TBB = TBB.astype(np.float64)
    else:
        AA = AA.astype(np.float32)
        BB = BB.astype(np.float32)
        TAA = TAA.astype(np.float32)
        TBB = TBB.astype(np.float32)
    
    return AA, TAA, BB, TBB


def twed_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=0):
    """Compute batch Time Warp Edit Distances using the CUDA implementation.
    
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
    # Prepare inputs
    AA, TAA, BB, TBB = _prepare_batch_arrays(AA, TAA, BB, TBB)
    
    # Get dimensions
    nAA, nA, dim = AA.shape
    nBB, nB = BB.shape[:2]
    
    # Create result array
    result = np.zeros((nAA, nBB), dtype=AA.dtype)
    
    # Call the appropriate C function based on data type
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
                         nu, lamb, degree, dim, nAA, nBB, result_ptr, tri)
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
                         nu, lamb, degree, dim, nAA, nBB, result_ptr, tri)
    
    if ret_code != 0:
        raise RuntimeError(f"CUDA batch TWED computation failed with error code {ret_code}")
    
    return result