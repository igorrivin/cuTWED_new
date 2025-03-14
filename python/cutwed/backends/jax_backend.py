"""JAX implementation of the TWED algorithm."""

import numpy as np
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
except ImportError:
    raise ImportError("JAX backend requires JAX to be installed. Please install it with: pip install jax")


def lpnorm(x, p=2, axis=-1):
    """Compute the Lp norm along the specified axis."""
    return jnp.power(jnp.sum(jnp.power(jnp.abs(x), p), axis=axis), 1/p)


@jit
def _twed_inner(A, TA, B, TB, nu, lamb, degree):
    """Inner function for computing TWED distance."""
    # Get dimensions
    nA = A.shape[0]
    nB = B.shape[0]
    
    # Initialize DP matrix
    dp = jnp.full((nA + 1, nB + 1), jnp.inf)
    dp = dp.at[0, 0].set(0.0)
    
    # Precompute local costs
    da = jnp.zeros(nA + 1)
    db = jnp.zeros(nB + 1)
    
    # Initialize da and db
    def init_da(i, da):
        """Initialize da[i]."""
        da_val = jnp.where(i == 1, 
                           lpnorm(A[i-1], p=degree),
                           lpnorm(A[i-1] - A[i-2], p=degree))
        return da.at[i].set(da_val)
    
    def init_db(j, db):
        """Initialize db[j]."""
        db_val = jnp.where(j == 1,
                           lpnorm(B[j-1], p=degree),
                           lpnorm(B[j-1] - B[j-2], p=degree))
        return db.at[j].set(db_val)
    
    # Use lax.fori_loop to initialize da and db
    da = lax.fori_loop(1, nA + 1, init_da, da)
    db = lax.fori_loop(1, nB + 1, init_db, db)
    
    # Define a function to update a single cell in the DP matrix
    def update_dp_cell(dp, i, j):
        """Update dp[i, j]."""
        # Cost between A[i-1] and B[j-1]
        cost = lpnorm(A[i-1] - B[j-1], p=degree)
        
        # Additional cost for alignment of previous elements
        cost = jnp.where((i > 1) & (j > 1),
                         cost + lpnorm(A[i-2] - B[j-2], p=degree),
                         cost)
        
        # Time timestamp differences
        htrans = jnp.abs(TA[i-1] - TB[j-1])
        htrans = jnp.where((i > 1) & (j > 1),
                           htrans + jnp.abs(TA[i-2] - TB[j-2]),
                           htrans)
        
        # Case 1: Match A[i-1] and B[j-1]
        c1 = dp[i-1, j-1] + cost + nu * htrans
        
        # Case 2: Delete A[i-1]
        htrans2 = jnp.where(i > 1, TA[i-1] - TA[i-2], TA[i-1])
        c2 = dp[i-1, j] + da[i] + lamb + nu * htrans2
        
        # Case 3: Delete B[j-1]
        htrans3 = jnp.where(j > 1, TB[j-1] - TB[j-2], TB[j-1])
        c3 = dp[i, j-1] + db[j] + lamb + nu * htrans3
        
        # Take the minimum
        return dp.at[i, j].set(jnp.minimum(jnp.minimum(c1, c2), c3))
    
    # Fill the DP matrix using nested loops
    def fill_row(j, dp_i):
        """Fill a row of the DP matrix."""
        return lax.fori_loop(1, nB + 1, lambda j, dp: update_dp_cell(dp, i, j), dp_i)
    
    def fill_dp(i, dp):
        """Fill the DP matrix one row at a time."""
        return lax.fori_loop(1, nB + 1, lambda j, dp: update_dp_cell(dp, i, j), dp)
    
    # Use lax.fori_loop for the outer loop
    dp = lax.fori_loop(1, nA + 1, lambda i, dp: fill_dp(i, dp), dp)
    
    return dp[nA, nB]


def twed(A, TA, B, TB, nu, lamb, degree=2):
    """Compute Time Warp Edit Distance between two time series using JAX.
    
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
    # Convert inputs to JAX arrays
    A = jnp.asarray(A)
    TA = jnp.asarray(TA)
    B = jnp.asarray(B)
    TB = jnp.asarray(TB)
    
    # Ensure A and B are 2D
    if A.ndim == 1:
        A = A[:, jnp.newaxis]
    if B.ndim == 1:
        B = B[:, jnp.newaxis]
    
    # Compute TWED distance
    result = _twed_inner(A, TA, B, TB, nu, lamb, degree)
    
    # Convert to a Python scalar
    return float(result)


# Vectorized version of TWED to compute batch distances
_twed_vmapped = jit(vmap(_twed_inner, in_axes=(None, None, 0, 0, None, None, None)))


def twed_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=0):
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
    # Convert inputs to JAX arrays
    AA = jnp.asarray(AA)
    TAA = jnp.asarray(TAA)
    BB = jnp.asarray(BB)
    TBB = jnp.asarray(TBB)
    
    # Ensure AA and BB are 3D
    if AA.ndim == 2:
        AA = AA[:, :, jnp.newaxis]
    if BB.ndim == 2:
        BB = BB[:, :, jnp.newaxis]
    
    # Get dimensions
    nAA, nA, dim = AA.shape
    nBB, nB = BB.shape[:2]
    
    # Initialize result matrix
    result = jnp.zeros((nAA, nBB), dtype=AA.dtype)
    
    # Optimization for same length time series in batch
    if nA == nB and AA.shape[2] == BB.shape[2]:
        # Compute distances for each A against all Bs
        for i in range(nAA):
            # Extract single time series
            A_i = AA[i]
            TA_i = TAA[i]
            
            # Use vectorized version to compute distances to all Bs
            result = result.at[i].set(_twed_vmapped(A_i, TA_i, BB, TBB, nu, lamb, degree))
    else:
        # Fallback for different length time series
        for i in range(nAA):
            for j in range(nBB):
                # Check triangle optimization
                if (tri == -1 and i < j) or (tri == -2 and i > j):
                    continue
                    
                # Compute distance
                result = result.at[i, j].set(twed(AA[i], TAA[i], BB[j], TBB[j], nu, lamb, degree))
                
                # Fill symmetric part for triangle optimization
                if tri != 0 and nAA == nBB and i != j:
                    result = result.at[j, i].set(result[i, j])
    
    # Convert to numpy array
    return np.array(result)