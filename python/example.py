#!/usr/bin/env python
"""
Example of using cuTWED with NumPy and CuPy.
"""

import numpy as np
import time
import cutwed

# Check if CuPy is available
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
    print("CuPy not available, using NumPy only")

# Generate random time series
def generate_time_series(n, dim, seed=42):
    """Generate random time series."""
    np.random.seed(seed)
    A = np.random.randn(n, dim).astype(np.float32)
    TA = np.arange(n, dtype=np.float32)
    B = np.random.randn(n, dim).astype(np.float32)
    TB = np.arange(n, dtype=np.float32)
    return A, TA, B, TB

# Generate random time series for batch processing
def generate_batch_time_series(batch_size, n, dim, seed=42):
    """Generate random time series for batch processing."""
    np.random.seed(seed)
    AA = np.random.randn(batch_size, n, dim).astype(np.float32)
    TAA = np.tile(np.arange(n, dtype=np.float32), (batch_size, 1))
    BB = np.random.randn(batch_size, n, dim).astype(np.float32)
    TBB = np.tile(np.arange(n, dtype=np.float32), (batch_size, 1))
    return AA, TAA, BB, TBB

# Example 1: Basic TWED with NumPy
def example_basic_numpy():
    """Basic TWED with NumPy."""
    print("\nExample 1: Basic TWED with NumPy")
    A, TA, B, TB = generate_time_series(100, 3)
    
    # Parameters for TWED
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute TWED
    start_time = time.time()
    distance = cutwed.twed(A, TA, B, TB, nu, lamb, degree)
    elapsed = time.time() - start_time
    
    print(f"TWED distance: {distance:.6f}")
    print(f"Computation time: {elapsed:.6f} seconds")

# Example 2: Batch TWED with NumPy
def example_batch_numpy():
    """Batch TWED with NumPy."""
    print("\nExample 2: Batch TWED with NumPy")
    batch_size = 10
    AA, TAA, BB, TBB = generate_batch_time_series(batch_size, 100, 3)
    
    # Parameters for TWED
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute batch TWED
    start_time = time.time()
    distances = cutwed.twed_batch(AA, TAA, BB, TBB, nu, lamb, degree)
    elapsed = time.time() - start_time
    
    print(f"Batch TWED distances shape: {distances.shape}")
    print(f"First distance: {distances[0, 0]:.6f}")
    print(f"Computation time: {elapsed:.6f} seconds")

# Example 3: TWED with CuPy (if available)
def example_cupy():
    """TWED with CuPy."""
    if not has_cupy:
        print("\nExample 3: TWED with CuPy (skipped, CuPy not available)")
        return
    
    print("\nExample 3: TWED with CuPy")
    A_np, TA_np, B_np, TB_np = generate_time_series(100, 3)
    
    # Convert to CuPy arrays
    A = cp.asarray(A_np)
    TA = cp.asarray(TA_np)
    B = cp.asarray(B_np)
    TB = cp.asarray(TB_np)
    
    # Parameters for TWED
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute TWED using CuPy
    start_time = time.time()
    distance = cutwed.twed_cupy(A, TA, B, TB, nu, lamb, degree)
    elapsed = time.time() - start_time
    
    print(f"TWED distance (CuPy): {distance:.6f}")
    print(f"Computation time: {elapsed:.6f} seconds")
    
    # Compare with NumPy version
    numpy_distance = cutwed.twed(A_np, TA_np, B_np, TB_np, nu, lamb, degree)
    print(f"TWED distance (NumPy): {numpy_distance:.6f}")
    print(f"Difference: {abs(distance - numpy_distance):.6e}")

# Example 4: Batch TWED with CuPy (if available)
def example_batch_cupy():
    """Batch TWED with CuPy."""
    if not has_cupy:
        print("\nExample 4: Batch TWED with CuPy (skipped, CuPy not available)")
        return
    
    print("\nExample 4: Batch TWED with CuPy")
    batch_size = 10
    AA_np, TAA_np, BB_np, TBB_np = generate_batch_time_series(batch_size, 100, 3)
    
    # Convert to CuPy arrays
    AA = cp.asarray(AA_np)
    TAA = cp.asarray(TAA_np)
    BB = cp.asarray(BB_np)
    TBB = cp.asarray(TBB_np)
    
    # Parameters for TWED
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute batch TWED using CuPy
    start_time = time.time()
    distances = cutwed.twed_batch_cupy(AA, TAA, BB, TBB, nu, lamb, degree)
    elapsed = time.time() - start_time
    
    print(f"Batch TWED distances shape (CuPy): {distances.shape}")
    print(f"First distance (CuPy): {distances[0, 0]:.6f}")
    print(f"Computation time: {elapsed:.6f} seconds")
    
    # Compare with NumPy version
    start_time = time.time()
    numpy_distances = cutwed.twed_batch(AA_np, TAA_np, BB_np, TBB_np, nu, lamb, degree)
    numpy_elapsed = time.time() - start_time
    
    print(f"First distance (NumPy): {numpy_distances[0, 0]:.6f}")
    print(f"NumPy computation time: {numpy_elapsed:.6f} seconds")
    print(f"Speedup: {numpy_elapsed / elapsed:.2f}x")

# Example 5: Performance comparison with different sizes
def example_performance():
    """Performance comparison with different sizes."""
    print("\nExample 5: Performance comparison with different sizes")
    
    sizes = [100, 500, 1000]
    
    for n in sizes:
        print(f"\nTesting with time series length: {n}")
        A, TA, B, TB = generate_time_series(n, 3)
        
        # NumPy version
        start_time = time.time()
        numpy_distance = cutwed.twed(A, TA, B, TB, 1.0, 1.0, 2)
        numpy_elapsed = time.time() - start_time
        print(f"NumPy: {numpy_elapsed:.6f} seconds")
        
        # CuPy version (if available)
        if has_cupy:
            A_gpu = cp.asarray(A)
            TA_gpu = cp.asarray(TA)
            B_gpu = cp.asarray(B)
            TB_gpu = cp.asarray(TB)
            
            start_time = time.time()
            cupy_distance = cutwed.twed_cupy(A_gpu, TA_gpu, B_gpu, TB_gpu, 1.0, 1.0, 2)
            cupy_elapsed = time.time() - start_time
            print(f"CuPy: {cupy_elapsed:.6f} seconds")
            print(f"Speedup: {numpy_elapsed / cupy_elapsed:.2f}x")

if __name__ == "__main__":
    print("cuTWED Examples")
    example_basic_numpy()
    example_batch_numpy()
    example_cupy()
    example_batch_cupy()
    example_performance()