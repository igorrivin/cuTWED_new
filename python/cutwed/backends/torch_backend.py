"""PyTorch implementation of the TWED algorithm."""

import numpy as np
try:
    import torch
    from torch import Tensor
except ImportError:
    raise ImportError("PyTorch backend requires PyTorch to be installed. Please install it with: pip install torch")


def lpnorm(x: Tensor, p: int = 2, dim: int = -1) -> Tensor:
    """Compute the Lp norm along the specified dimension."""
    return torch.norm(x, p=p, dim=dim)


def twed(A: Tensor, 
         TA: Tensor, 
         B: Tensor, 
         TB: Tensor, 
         nu: float, 
         lamb: float, 
         degree: int = 2) -> float:
    """Compute Time Warp Edit Distance between two time series using PyTorch.
    
    This implementation works on CPU or GPU depending on the input tensors.
    
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
    # Convert inputs to PyTorch tensors if they're not already
    if not isinstance(A, Tensor):
        A = torch.tensor(A)
    if not isinstance(TA, Tensor):
        TA = torch.tensor(TA)
    if not isinstance(B, Tensor):
        B = torch.tensor(B)
    if not isinstance(TB, Tensor):
        TB = torch.tensor(TB)
    
    # Ensure inputs have the same dtype and device
    dtype = A.dtype
    device = A.device
    TA = TA.to(dtype=dtype, device=device)
    B = B.to(dtype=dtype, device=device)
    TB = TB.to(dtype=dtype, device=device)
    
    # Ensure A and B are 2D
    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)
    
    # Get dimensions
    nA, dim = A.shape
    nB = B.shape[0]
    
    # Initialize distance matrix
    dp = torch.full((nA + 1, nB + 1), float('inf'), dtype=dtype, device=device)
    dp[0, 0] = 0.0
    
    # Precompute local costs
    # Cost for A[i-1] to zero
    da = torch.zeros(nA + 1, dtype=dtype, device=device)
    for i in range(1, nA + 1):
        if i == 1:
            da[i] = lpnorm(A[i-1], p=degree)
        else:
            da[i] = lpnorm(A[i-1] - A[i-2], p=degree)
    
    # Cost for B[j-1] to zero
    db = torch.zeros(nB + 1, dtype=dtype, device=device)
    for j in range(1, nB + 1):
        if j == 1:
            db[j] = lpnorm(B[j-1], p=degree)
        else:
            db[j] = lpnorm(B[j-1] - B[j-2], p=degree)
    
    # Fill the DP matrix
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            # Cost between A[i-1] and B[j-1]
            cost = lpnorm(A[i-1] - B[j-1], p=degree)
            
            # Additional cost for alignment of previous elements
            if i > 1 and j > 1:
                cost += lpnorm(A[i-2] - B[j-2], p=degree)
            
            # Time timestamp differences
            htrans = torch.abs(TA[i-1] - TB[j-1])
            if i > 1 and j > 1:
                htrans += torch.abs(TA[i-2] - TB[j-2])
            
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
            
            # Take the minimum
            dp[i, j] = torch.min(torch.min(c1, c2), c3)
    
    # Return the result as a Python scalar
    return dp[nA, nB].item()


def twed_batch(AA: Tensor,
               TAA: Tensor,
               BB: Tensor,
               TBB: Tensor,
               nu: float,
               lamb: float,
               degree: int = 2,
               tri: int = 0) -> Tensor:
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
    # Convert inputs to PyTorch tensors if they're not already
    if not isinstance(AA, Tensor):
        AA = torch.tensor(AA)
    if not isinstance(TAA, Tensor):
        TAA = torch.tensor(TAA)
    if not isinstance(BB, Tensor):
        BB = torch.tensor(BB)
    if not isinstance(TBB, Tensor):
        TBB = torch.tensor(TBB)
    
    # Ensure all inputs have the same dtype and device
    dtype = AA.dtype
    device = AA.device
    TAA = TAA.to(dtype=dtype, device=device)
    BB = BB.to(dtype=dtype, device=device)
    TBB = TBB.to(dtype=dtype, device=device)
    
    # Ensure AA and BB are 3D
    if AA.dim() == 2:
        AA = AA.unsqueeze(2)
    if BB.dim() == 2:
        BB = BB.unsqueeze(2)
    
    # Get dimensions
    nAA, nA, dim = AA.shape
    nBB, nB = BB.shape[:2]
    
    # Initialize result matrix
    result = torch.zeros((nAA, nBB), dtype=dtype, device=device)
    
    # Determine computation strategy based on triangular optimization
    pairs = []
    if tri == -1 and nAA == nBB:  # Lower triangular
        for i in range(nAA):
            for j in range(min(i+1, nBB)):
                pairs.append((i, j))
    elif tri == -2 and nAA == nBB:  # Upper triangular
        for i in range(nAA):
            for j in range(i, nBB):
                pairs.append((i, j))
    else:  # Full matrix
        for i in range(nAA):
            for j in range(nBB):
                pairs.append((i, j))
    
    # Compute distances
    for i, j in pairs:
        result[i, j] = twed(AA[i], TAA[i], BB[j], TBB[j], nu, lamb, degree)
        
        # For triangular optimization with symmetric inputs, fill the other half
        if tri != 0 and nAA == nBB and i != j:
            result[j, i] = result[i, j]
    
    return result
    

def batched_twed_dp(AA: Tensor,
                   TAA: Tensor,
                   BB: Tensor,
                   TBB: Tensor,
                   nu: float,
                   lamb: float,
                   degree: int = 2) -> Tensor:
    """More efficient batched TWED implementation for same-length time series.
    
    This implementation is faster for batches where all time series have the same length,
    but is less flexible than the general twed_batch function.
    
    Args:
        AA: Batch of first time series, shape (batch, time, dim)
        TAA: Batch of first timestamps, shape (batch, time)
        BB: Batch of second time series, shape (batch, time, dim)
        TBB: Batch of second timestamps, shape (batch, time)
        nu: Elasticity parameter
        lamb: Stiffness parameter
        degree: Power used in the Lp norm
        
    Returns:
        Distances tensor of shape (batch,)
    """
    # Ensure inputs are 3D
    if AA.dim() == 2:
        AA = AA.unsqueeze(2)
    if BB.dim() == 2:
        BB = BB.unsqueeze(2)
    
    # Get dimensions
    batch, time, dim = AA.shape
    
    # Initialize DP matrix: [batch, time+1, time+1]
    dp = torch.full((batch, time+1, time+1), float('inf'), 
                     dtype=AA.dtype, device=AA.device)
    dp[:, 0, 0] = 0.0
    
    # Compute pairwise distances in batch
    # This could be optimized further with custom CUDA kernels
    for i in range(1, time+1):
        for j in range(1, time+1):
            # Current points
            a_i = AA[:, i-1]  # [batch, dim]
            b_j = BB[:, j-1]  # [batch, dim]
            
            # Compute distance between current points
            diff = a_i - b_j  # [batch, dim]
            cost = torch.norm(diff, p=degree, dim=1)  # [batch]
            
            # Add distance between previous points if available
            if i > 1 and j > 1:
                a_prev = AA[:, i-2]  # [batch, dim]
                b_prev = BB[:, j-2]  # [batch, dim]
                prev_diff = a_prev - b_prev  # [batch, dim]
                cost += torch.norm(prev_diff, p=degree, dim=1)  # [batch]
            
            # Time differences
            time_diff = torch.abs(TAA[:, i-1] - TBB[:, j-1])  # [batch]
            if i > 1 and j > 1:
                time_diff += torch.abs(TAA[:, i-2] - TBB[:, j-2])  # [batch]
            
            # Case 1: Match A[i-1] and B[j-1]
            c1 = dp[:, i-1, j-1] + cost + nu * time_diff  # [batch]
            
            # Case 2: Delete A[i-1]
            if i > 1:
                time_diff2 = TAA[:, i-1] - TAA[:, i-2]  # [batch]
            else:
                time_diff2 = TAA[:, i-1]  # [batch]
                
            # Compute deletion cost for A
            if i == 1:
                da_i = torch.norm(AA[:, i-1], p=degree, dim=1)  # [batch]
            else:
                da_i = torch.norm(AA[:, i-1] - AA[:, i-2], p=degree, dim=1)  # [batch]
                
            c2 = dp[:, i-1, j] + da_i + lamb + nu * time_diff2  # [batch]
            
            # Case 3: Delete B[j-1]
            if j > 1:
                time_diff3 = TBB[:, j-1] - TBB[:, j-2]  # [batch]
            else:
                time_diff3 = TBB[:, j-1]  # [batch]
                
            # Compute deletion cost for B
            if j == 1:
                db_j = torch.norm(BB[:, j-1], p=degree, dim=1)  # [batch]
            else:
                db_j = torch.norm(BB[:, j-1] - BB[:, j-2], p=degree, dim=1)  # [batch]
                
            c3 = dp[:, i, j-1] + db_j + lamb + nu * time_diff3  # [batch]
            
            # Take the minimum of the three cases
            dp[:, i, j] = torch.min(torch.min(c1, c2), c3)  # [batch]
    
    # Return the final distances
    return dp[:, time, time]