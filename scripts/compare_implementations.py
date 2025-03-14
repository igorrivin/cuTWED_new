#!/usr/bin/env python
"""
Compare the original cuTWED implementation with the refactored version.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add paths to original and refactored implementations
original_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
refactored_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python'))

sys.path.insert(0, original_path)
sys.path.insert(0, refactored_path)

# Try to import original implementation
try:
    from cuTWED import twed as original_twed
    from cuTWED import twed_batch as original_twed_batch
    has_original = True
except ImportError:
    print("Original cuTWED implementation not found")
    has_original = False

# Try to import refactored implementation
try:
    from cutwed import twed as new_twed
    from cutwed import twed_batch as new_twed_batch
    has_refactored = True
except ImportError:
    print("Refactored cuTWED implementation not found")
    has_refactored = False

# Try to import CuPy versions
try:
    import cupy as cp
    from cutwed import twed_cupy, twed_batch_cupy
    has_cupy = True
except ImportError:
    print("CuPy not found, skipping CuPy tests")
    has_cupy = False

# Check if we have both implementations
if not (has_original and has_refactored):
    print("Cannot compare implementations, at least one is missing")
    sys.exit(1)

# Generate random time series of different sizes
def generate_time_series(n, dim, dtype=np.float32):
    """Generate random time series."""
    np.random.seed(42)
    A = np.random.randn(n, dim).astype(dtype)
    TA = np.arange(n, dtype=dtype)
    B = np.random.randn(n, dim).astype(dtype)
    TB = np.arange(n, dtype=dtype)
    return A, TA, B, TB

# Generate batch data
def generate_batch_data(batch_size, n, dim, dtype=np.float32):
    """Generate random batch data."""
    np.random.seed(42)
    AA = np.random.randn(batch_size, n, dim).astype(dtype)
    TAA = np.tile(np.arange(n, dtype=dtype), (batch_size, 1))
    BB = np.random.randn(batch_size, n, dim).astype(dtype)
    TBB = np.tile(np.arange(n, dtype=dtype), (batch_size, 1))
    return AA, TAA, BB, TBB

# Parameters for TWED
nu = 1.0
lamb = 1.0
degree = 2

# Test different sizes
sizes = [10, 50, 100, 500, 1000]
dimensions = [1, 3, 10]
batch_sizes = [5, 10, 20]

# Results
results = {
    'single': {
        'original': {'time': [], 'distance': []},
        'refactored': {'time': [], 'distance': []},
        'cupy': {'time': [], 'distance': []}
    },
    'batch': {
        'original': {'time': [], 'distance': []},
        'refactored': {'time': [], 'distance': []},
        'cupy': {'time': [], 'distance': []}
    }
}

# Test single time series
print("Testing single time series...")
for size in sizes:
    for dim in dimensions:
        print(f"  Size: {size}, Dimension: {dim}")
        
        # Generate data
        A, TA, B, TB = generate_time_series(size, dim)
        
        # Test original implementation
        start_time = time.time()
        dist_original = original_twed(A, TA, B, TB, nu, lamb, degree)
        time_original = time.time() - start_time
        
        # Test refactored implementation
        start_time = time.time()
        dist_refactored = new_twed(A, TA, B, TB, nu, lamb, degree)
        time_refactored = time.time() - start_time
        
        print(f"    Original: {dist_original:.6f} in {time_original:.6f}s")
        print(f"    Refactored: {dist_refactored:.6f} in {time_refactored:.6f}s")
        print(f"    Difference: {abs(dist_original - dist_refactored):.6e}")
        
        results['single']['original']['time'].append(time_original)
        results['single']['original']['distance'].append(dist_original)
        results['single']['refactored']['time'].append(time_refactored)
        results['single']['refactored']['distance'].append(dist_refactored)
        
        # Test CuPy implementation if available
        if has_cupy:
            A_gpu = cp.asarray(A)
            TA_gpu = cp.asarray(TA)
            B_gpu = cp.asarray(B)
            TB_gpu = cp.asarray(TB)
            
            start_time = time.time()
            dist_cupy = twed_cupy(A_gpu, TA_gpu, B_gpu, TB_gpu, nu, lamb, degree)
            time_cupy = time.time() - start_time
            
            print(f"    CuPy: {dist_cupy:.6f} in {time_cupy:.6f}s")
            print(f"    Difference: {abs(dist_original - dist_cupy):.6e}")
            
            results['single']['cupy']['time'].append(time_cupy)
            results['single']['cupy']['distance'].append(dist_cupy)

# Test batch processing
print("\nTesting batch processing...")
for size in sizes[:3]:  # Use smaller sizes for batch processing
    for dim in dimensions[:2]:  # Use smaller dimensions for batch processing
        for batch_size in batch_sizes:
            print(f"  Size: {size}, Dimension: {dim}, Batch size: {batch_size}")
            
            # Generate batch data
            AA, TAA, BB, TBB = generate_batch_data(batch_size, size, dim)
            
            # Test original implementation
            start_time = time.time()
            dist_original = original_twed_batch(AA, TAA, BB, TBB, nu, lamb, degree)
            time_original = time.time() - start_time
            
            # Test refactored implementation
            start_time = time.time()
            dist_refactored = new_twed_batch(AA, TAA, BB, TBB, nu, lamb, degree)
            time_refactored = time.time() - start_time
            
            print(f"    Original: {dist_original[0, 0]:.6f} in {time_original:.6f}s")
            print(f"    Refactored: {dist_refactored[0, 0]:.6f} in {time_refactored:.6f}s")
            print(f"    Difference: {abs(dist_original[0, 0] - dist_refactored[0, 0]):.6e}")
            
            results['batch']['original']['time'].append(time_original)
            results['batch']['original']['distance'].append(dist_original[0, 0])
            results['batch']['refactored']['time'].append(time_refactored)
            results['batch']['refactored']['distance'].append(dist_refactored[0, 0])
            
            # Test CuPy implementation if available
            if has_cupy:
                AA_gpu = cp.asarray(AA)
                TAA_gpu = cp.asarray(TAA)
                BB_gpu = cp.asarray(BB)
                TBB_gpu = cp.asarray(TBB)
                
                start_time = time.time()
                dist_cupy = twed_batch_cupy(AA_gpu, TAA_gpu, BB_gpu, TBB_gpu, nu, lamb, degree)
                time_cupy = time.time() - start_time
                
                print(f"    CuPy: {dist_cupy[0, 0]:.6f} in {time_cupy:.6f}s")
                print(f"    Difference: {abs(dist_original[0, 0] - cp.asnumpy(dist_cupy)[0, 0]):.6e}")
                
                results['batch']['cupy']['time'].append(time_cupy)
                results['batch']['cupy']['distance'].append(cp.asnumpy(dist_cupy)[0, 0])

# Plot results
plt.figure(figsize=(12, 10))

# Plot single time series results
plt.subplot(2, 2, 1)
plt.title('Single Time Series - Computation Time')
plt.plot(range(len(results['single']['original']['time'])), results['single']['original']['time'], 'b-', label='Original')
plt.plot(range(len(results['single']['refactored']['time'])), results['single']['refactored']['time'], 'r-', label='Refactored')
if has_cupy:
    plt.plot(range(len(results['single']['cupy']['time'])), results['single']['cupy']['time'], 'g-', label='CuPy')
plt.xlabel('Test Case')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)

# Plot single time series accuracy
plt.subplot(2, 2, 2)
plt.title('Single Time Series - Distance')
plt.plot(range(len(results['single']['original']['distance'])), results['single']['original']['distance'], 'b-', label='Original')
plt.plot(range(len(results['single']['refactored']['distance'])), results['single']['refactored']['distance'], 'r-', label='Refactored')
if has_cupy:
    plt.plot(range(len(results['single']['cupy']['distance'])), results['single']['cupy']['distance'], 'g-', label='CuPy')
plt.xlabel('Test Case')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)

# Plot batch processing results
plt.subplot(2, 2, 3)
plt.title('Batch Processing - Computation Time')
plt.plot(range(len(results['batch']['original']['time'])), results['batch']['original']['time'], 'b-', label='Original')
plt.plot(range(len(results['batch']['refactored']['time'])), results['batch']['refactored']['time'], 'r-', label='Refactored')
if has_cupy:
    plt.plot(range(len(results['batch']['cupy']['time'])), results['batch']['cupy']['time'], 'g-', label='CuPy')
plt.xlabel('Test Case')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)

# Plot batch processing accuracy
plt.subplot(2, 2, 4)
plt.title('Batch Processing - Distance')
plt.plot(range(len(results['batch']['original']['distance'])), results['batch']['original']['distance'], 'b-', label='Original')
plt.plot(range(len(results['batch']['refactored']['distance'])), results['batch']['refactored']['distance'], 'r-', label='Refactored')
if has_cupy:
    plt.plot(range(len(results['batch']['cupy']['distance'])), results['batch']['cupy']['distance'], 'g-', label='CuPy')
plt.xlabel('Test Case')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_results.png')
plt.show()

# Print summary
print("\nSummary:")
print("  Single Time Series:")
print(f"    Original: {np.mean(results['single']['original']['time']):.6f}s")
print(f"    Refactored: {np.mean(results['single']['refactored']['time']):.6f}s")
if has_cupy:
    print(f"    CuPy: {np.mean(results['single']['cupy']['time']):.6f}s")
    
print("  Batch Processing:")
print(f"    Original: {np.mean(results['batch']['original']['time']):.6f}s")
print(f"    Refactored: {np.mean(results['batch']['refactored']['time']):.6f}s")
if has_cupy:
    print(f"    CuPy: {np.mean(results['batch']['cupy']['time']):.6f}s")

# Calculate speedups
print("\nSpeedups:")
print("  Single Time Series:")
print(f"    Refactored vs Original: {np.mean(results['single']['original']['time']) / np.mean(results['single']['refactored']['time']):.2f}x")
if has_cupy:
    print(f"    CuPy vs Original: {np.mean(results['single']['original']['time']) / np.mean(results['single']['cupy']['time']):.2f}x")
    
print("  Batch Processing:")
print(f"    Refactored vs Original: {np.mean(results['batch']['original']['time']) / np.mean(results['batch']['refactored']['time']):.2f}x")
if has_cupy:
    print(f"    CuPy vs Original: {np.mean(results['batch']['original']['time']) / np.mean(results['batch']['cupy']['time']):.2f}x")