"""NumPy implementation of the TWED algorithm."""

import numpy as np
from typing import Tuple, Union, Optional, Any


def lpnorm(x: np.ndarray, p: int = 2, axis: int = -1) -> np.ndarray:
    """Compute the Lp norm along the specified axis.
    
    Args:
        x: Input array
        p: Norm degree
        axis: Axis along which to compute the norm
        
    Returns:
        Lp norm of x
    """
    return np.power(np.sum(np.power(np.abs(x), p), axis=axis), 1/p)


def twed(A: np.ndarray, 
         TA: np.ndarray, 
         B: np.ndarray, 
         TB: np.ndarray, 
         nu: float, 
         lamb: float, 
         degree: int = 2) -> float:
    """Compute Time Warp Edit Distance between two time series.
    
    This is a pure NumPy implementation that works on CPU.
    
    Args:
        A: First time series values, shape (nA, dim) or (nA,)
        TA: First time series timestamps, shape (nA,)
        B: Second time series values, shape (nB, dim) or (nB,)
        TB: Second time series timestamps, shape (nB,)
        nu: Elasticity parameter
        lamb: Stiffness parameter
        degree: Power used in the Lp norm, default is 2
        
    Returns:
        The TWED distance
    """
    # Ensure A and B are 2D
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    # Check dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dimension mismatch: A has shape {A.shape}, B has shape {B.shape}")
    
    # Get sizes
    nA, dim = A.shape
    nB = B.shape[0]
    
    # Initialize DP matrix
    dp = np.full((nA + 1, nB + 1), np.inf)
    dp[0, 0] = 0.0
    
    # Precompute local costs
    # Cost for A[i-1] to zero
    da = np.zeros(nA + 1)
    for i in range(1, nA + 1):
        if i == 1:
            da[i] = lpnorm(A[i-1])
        else:
            da[i] = lpnorm(A[i-1] - A[i-2])
    
    # Cost for B[j-1] to zero
    db = np.zeros(nB + 1)
    for j in range(1, nB + 1):
        if j == 1:
            db[j] = lpnorm(B[j-1])
        else:
            db[j] = lpnorm(B[j-1] - B[j-2])
    
    # Dynamic programming to fill the matrix
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            # Cost between A[i-1] and B[j-1]
            cost = lpnorm(A[i-1] - B[j-1])
            
            # Additional cost for alignment of previous elements
            if i > 1 and j > 1:
                cost += lpnorm(A[i-2] - B[j-2])
            
            # Time timestamp differences
            htrans = np.abs(TA[i-1] - TB[j-1])
            if i > 1 and j > 1:
                htrans += np.abs(TA[i-2] - TB[j-2])
            
            # Case 1: Match A[i-1] and B[j-1]
            c1 = dp[i-1, j-1] + cost + nu * htrans
            
            # Case 2: Delete A[i-1]
            if i > 1:
                htrans2 = TA[i-1] - TA[i-2]
            else:
                htrans2 = TA[i-1]
            c2 = dp[i-1, j] + da[i] + lamb + nu * htrans2
            
            # Case 3: Delete B[j-1]
            if j > 1:
                htrans3 = TB[j-1] - TB[j-2]
            else:
                htrans3 = TB[j-1]
            c3 = dp[i, j-1] + db[j] + lamb + nu * htrans3
            
            # Take the minimum of the three cases
            dp[i, j] = min(c1, c2, c3)
    
    return dp[nA, nB]


def twed_batch(AA: np.ndarray, 
               TAA: np.ndarray, 
               BB: np.ndarray, 
               TBB: np.ndarray, 
               nu: float, 
               lamb: float, 
               degree: int = 2, 
               tri: int = 0) -> np.ndarray:
    """Compute batch Time Warp Edit Distances between multiple time series.
    
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
    # Ensure AA and BB are 3D
    if AA.ndim == 2:
        AA = AA.reshape(AA.shape[0], AA.shape[1], 1)
    if BB.ndim == 2:
        BB = BB.reshape(BB.shape[0], BB.shape[1], 1)
    
    # Get sizes
    nAA, nA, dim = AA.shape
    nBB, nB = BB.shape[:2]
    
    # Check that BB has the same dimensionality
    if BB.shape[2] != dim:
        raise ValueError(f"Dimension mismatch: AA has {dim} dimensions, BB has {BB.shape[2]}")
    
    # Initialize result matrix
    result = np.zeros((nAA, nBB), dtype=AA.dtype)
    
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