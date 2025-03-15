"""
cuTWED: Time Warp Edit Distance algorithm with multiple backends.

This package provides a unified interface to the Time Warp Edit Distance (TWED)
algorithm with multiple backend implementations:

- cuda: Original CUDA C++ implementation (fastest, requires NVIDIA GPU and CUDA)
- cupy: CuPy implementation (fast, requires NVIDIA GPU)
- jax: JAX implementation (works on CPU/GPU/TPU)
- pytorch: PyTorch implementation (works on CPU/GPU)
- numpy: NumPy implementation (works on CPU only)

The backend can be selected at runtime with the set_backend() function.
"""

import os
import numpy as np
from enum import IntEnum
from typing import Union, Optional, List, Tuple, Any, Dict

from .backends import (
    get_backend,
    get_backend_name,
    set_backend,
    get_available_backends,
)

__version__ = '3.0.0'


class TriangleOpt(IntEnum):
    """Triangle optimization options for batch processing."""
    TRIU = -2  # Upper triangular optimization
    TRIL = -1  # Lower triangular optimization
    NOPT = 0   # No optimization


def twed(A: np.ndarray, 
         TA: np.ndarray, 
         B: np.ndarray, 
         TB: np.ndarray, 
         nu: float, 
         lamb: float, 
         degree: int = 2,
         backend: Optional[str] = None) -> float:
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
    backend : str, optional
        Backend to use for computation. If None, uses the current backend.
        
    Returns
    -------
    float
        The TWED distance
    """
    # Use the specified backend or the current one
    if backend is not None:
        old_backend = get_backend_name()
        set_backend(backend)
        
    # Get the backend implementation
    backend_module = get_backend()
    
    # Compute the TWED distance
    result = backend_module.twed(A, TA, B, TB, nu, lamb, degree)
    
    # Restore the original backend if changed
    if backend is not None:
        set_backend(old_backend)
    
    return result


def twed_batch(AA: np.ndarray,
               TAA: np.ndarray,
               BB: np.ndarray,
               TBB: np.ndarray,
               nu: float,
               lamb: float,
               degree: int = 2,
               tri: Union[TriangleOpt, int] = TriangleOpt.NOPT,
               backend: Optional[str] = None) -> np.ndarray:
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
    tri : TriangleOpt or int, optional
        Triangle optimization option, default is NOPT
    backend : str, optional
        Backend to use for computation. If None, uses the current backend.
        
    Returns
    -------
    ndarray
        Distance matrix of shape (nAA, nBB)
    """
    # Use the specified backend or the current one
    if backend is not None:
        old_backend = get_backend_name()
        set_backend(backend)
    
    # Get the backend implementation
    backend_module = get_backend()
    
    # Convert TriangleOpt enum to int if needed
    if isinstance(tri, TriangleOpt):
        tri = int(tri)
    
    # Compute the batch distances
    result = backend_module.twed_batch(AA, TAA, BB, TBB, nu, lamb, degree, tri)
    
    # Restore the original backend if changed
    if backend is not None:
        set_backend(old_backend)
    
    return result