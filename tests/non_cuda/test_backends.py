#!/usr/bin/env python
"""
Test script for cuTWED backends that don't require CUDA (NumPy, PyTorch, JAX).
This script tests both correctness and performance.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

# Add the Python package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

# Import backends directly to avoid compilation requirements
from cutwed.backends import numpy_backend

# Try to import optional backends
BACKENDS = {"numpy": numpy_backend}

try:
    from cutwed.backends import torch_backend
    BACKENDS["pytorch"] = torch_backend
    import torch
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available, skipping PyTorch tests")
    HAS_TORCH = False

try:
    from cutwed.backends import jax_backend
    BACKENDS["jax"] = jax_backend
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    print("JAX not available, skipping JAX tests")
    HAS_JAX = False


def generate_time_series(n: int, dim: int = 1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random time series for testing."""
    np.random.seed(seed)
    A = np.random.randn(n, dim).astype(np.float32)
    TA = np.arange(n, dtype=np.float32)
    B = np.random.randn(n, dim).astype(np.float32)
    TB = np.arange(n, dtype=np.float32)
    return A, TA, B, TB


def generate_sine_series(n: int, dim: int = 1, freq: float = 1.0, phase_shift: float = 0.0, 
                        noise_level: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sine wave time series with noise."""
    np.random.seed(seed)
    t = np.arange(n, dtype=np.float32)
    
    if dim == 1:
        values = np.sin(freq * 2 * np.pi * t / n + phase_shift)
        values = values + np.random.randn(n) * noise_level
        values = values.reshape(-1, 1)
    else:
        values = np.zeros((n, dim), dtype=np.float32)
        for d in range(dim):
            # Different frequency for each dimension
            f = freq * (1.0 + 0.1 * d)
            p = phase_shift * (1.0 + 0.1 * d)
            values[:, d] = np.sin(f * 2 * np.pi * t / n + p)
        
        values = values + np.random.randn(n, dim) * noise_level
    
    return values.astype(np.float32), t.astype(np.float32)


def convert_to_backend_format(A: np.ndarray, TA: np.ndarray, B: np.ndarray, TB: np.ndarray, 
                             backend: str) -> Tuple[Any, Any, Any, Any]:
    """Convert numpy arrays to the appropriate format for each backend."""
    if backend == "numpy":
        return A, TA, B, TB
    elif backend == "pytorch" and HAS_TORCH:
        return (torch.tensor(A), torch.tensor(TA), 
                torch.tensor(B), torch.tensor(TB))
    elif backend == "jax" and HAS_JAX:
        return (jnp.array(A), jnp.array(TA), 
                jnp.array(B), jnp.array(TB))
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")


def test_correctness():
    """Test that all backends produce comparable results."""
    print("\n=== Testing Correctness ===")
    
    # Generate a simple time series
    A, TA, B, TB = generate_time_series(100, dim=3)
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute distances with different backends
    results = {}
    for name, backend in BACKENDS.items():
        print(f"Testing {name} backend...")
        A_b, TA_b, B_b, TB_b = convert_to_backend_format(A, TA, B, TB, name)
        
        # Compute distance
        distance = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
        results[name] = distance
        print(f"  Distance: {distance:.6f}")
    
    # Compare results
    if len(results) > 1:
        print("\nComparison:")
        reference = results["numpy"]
        for name, value in results.items():
            if name != "numpy":
                rel_diff = abs(value - reference) / (abs(reference) + 1e-6)
                print(f"  {name} vs numpy: relative difference = {rel_diff:.6e}")
                assert rel_diff < 1e-5, f"Large difference between {name} and numpy"
    
    print("All backends produce consistent results!")


def test_correctness_batch():
    """Test batch computation correctness."""
    print("\n=== Testing Batch Correctness ===")
    
    # Generate batch data
    batch_size = 5
    length = 30
    dim = 2
    
    # Create time series with different frequencies
    AA = np.zeros((batch_size, length, dim), dtype=np.float32)
    TAA = np.tile(np.arange(length, dtype=np.float32), (batch_size, 1))
    BB = np.zeros((batch_size, length, dim), dtype=np.float32)
    TBB = np.tile(np.arange(length, dtype=np.float32), (batch_size, 1))
    
    for i in range(batch_size):
        AA[i], _ = generate_sine_series(length, dim, freq=1.0 + 0.2 * i, seed=42 + i)
        BB[i], _ = generate_sine_series(length, dim, freq=1.0 + 0.2 * i, phase_shift=0.5, seed=100 + i)
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute batch distances with different backends
    batch_results = {}
    for name, backend in BACKENDS.items():
        print(f"Testing {name} batch computation...")
        
        if name == "numpy":
            AA_b, TAA_b, BB_b, TBB_b = AA, TAA, BB, TBB
        elif name == "pytorch" and HAS_TORCH:
            AA_b = torch.tensor(AA)
            TAA_b = torch.tensor(TAA)
            BB_b = torch.tensor(BB)
            TBB_b = torch.tensor(TBB)
        elif name == "jax" and HAS_JAX:
            AA_b = jnp.array(AA)
            TAA_b = jnp.array(TAA)
            BB_b = jnp.array(BB)
            TBB_b = jnp.array(TBB)
        else:
            continue
        
        # Compute batch distances
        distances = backend.twed_batch(AA_b, TAA_b, BB_b, TBB_b, nu, lamb, degree)
        
        if name == "pytorch" and HAS_TORCH:
            distances = distances.numpy()
        elif name == "jax" and HAS_JAX:
            distances = np.array(distances)
            
        batch_results[name] = distances
        print(f"  First distance: {distances[0, 0]:.6f}")
    
    # Compare results
    if len(batch_results) > 1:
        print("\nBatch comparison:")
        reference = batch_results["numpy"]
        for name, values in batch_results.items():
            if name != "numpy":
                # Calculate max absolute relative difference
                rel_diff = np.max(np.abs(values - reference) / (np.abs(reference) + 1e-6))
                print(f"  {name} vs numpy: max relative difference = {rel_diff:.6e}")
                assert rel_diff < 1e-5, f"Large difference between {name} and numpy"
    
    print("All backends produce consistent batch results!")


def test_performance():
    """Test performance of different backends with varying time series sizes."""
    print("\n=== Testing Performance ===")
    
    # Series of lengths to test
    lengths = [10, 50, 100, 200, 500, 1000]
    dimensions = [1, 3, 10]
    
    # Store results for plotting
    results = {backend: {dim: [] for dim in dimensions} for backend in BACKENDS.keys()}
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    for dim in dimensions:
        print(f"\nTesting with dimension: {dim}")
        
        for length in lengths:
            print(f"  Testing length: {length}")
            
            # Generate time series
            A, TA, B, TB = generate_time_series(length, dim=dim)
            
            # Test each backend
            for name, backend in BACKENDS.items():
                A_b, TA_b, B_b, TB_b = convert_to_backend_format(A, TA, B, TB, name)
                
                # Warm-up run
                _ = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
                
                # Timed runs
                times = []
                runs = 3  # Number of runs to average
                for _ in range(runs):
                    start_time = time.time()
                    _ = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                results[name][dim].append(avg_time)
                print(f"    {name}: {avg_time:.6f} seconds")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['o', 's', '^', 'D', 'x']
    
    for d_idx, dim in enumerate(dimensions):
        plt.subplot(1, len(dimensions), d_idx + 1)
        for b_idx, (backend, dim_results) in enumerate(results.items()):
            plt.plot(lengths, dim_results[dim], 
                    marker=markers[b_idx % len(markers)],
                    color=colors[b_idx % len(colors)],
                    label=backend)
        
        plt.xlabel('Time Series Length')
        plt.ylabel('Computation Time (s)')
        plt.title(f'Performance with {dim} dimension(s)')
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    
    print(f"Performance plot saved to 'performance_comparison.png'")


def test_batch_performance():
    """Test batch computation performance."""
    print("\n=== Testing Batch Performance ===")
    
    # Batch sizes to test
    batch_sizes = [2, 5, 10, 20]
    length = 100
    dim = 3
    
    # Store results for plotting
    results = {backend: [] for backend in BACKENDS.keys()}
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size: {batch_size}")
        
        # Generate batch data
        AA = np.zeros((batch_size, length, dim), dtype=np.float32)
        TAA = np.tile(np.arange(length, dtype=np.float32), (batch_size, 1))
        BB = np.zeros((batch_size, length, dim), dtype=np.float32)
        TBB = np.tile(np.arange(length, dtype=np.float32), (batch_size, 1))
        
        for i in range(batch_size):
            AA[i], _ = generate_sine_series(length, dim, freq=1.0 + 0.2 * i, seed=42 + i)
            BB[i], _ = generate_sine_series(length, dim, freq=1.0 + 0.2 * i, phase_shift=0.5, seed=100 + i)
        
        # Test each backend
        for name, backend in BACKENDS.items():
            if name == "numpy":
                AA_b, TAA_b, BB_b, TBB_b = AA, TAA, BB, TBB
            elif name == "pytorch" and HAS_TORCH:
                AA_b = torch.tensor(AA)
                TAA_b = torch.tensor(TAA)
                BB_b = torch.tensor(BB)
                TBB_b = torch.tensor(TBB)
            elif name == "jax" and HAS_JAX:
                AA_b = jnp.array(AA)
                TAA_b = jnp.array(TAA)
                BB_b = jnp.array(BB)
                TBB_b = jnp.array(TBB)
            else:
                continue
            
            # Warm-up
            _ = backend.twed_batch(AA_b, TAA_b, BB_b, TBB_b, nu, lamb, degree)
            
            # Timed runs
            times = []
            runs = 3  # Number of runs to average
            for _ in range(runs):
                start_time = time.time()
                _ = backend.twed_batch(AA_b, TAA_b, BB_b, TBB_b, nu, lamb, degree)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            results[name].append(avg_time)
            print(f"    {name}: {avg_time:.6f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for b_idx, (backend, times) in enumerate(results.items()):
        plt.plot(batch_sizes, times, 
                marker=markers[b_idx % len(markers)],
                color=colors[b_idx % len(colors)],
                label=backend)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Computation Time (s)')
    plt.title('Batch Computation Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig('batch_performance_comparison.png')
    plt.close()
    
    print(f"Batch performance plot saved to 'batch_performance_comparison.png'")


def run_benchmarks():
    """Run a series of benchmarks comparing different backends."""
    print("\n=== Running Benchmarks ===")
    
    print(f"Available backends: {list(BACKENDS.keys())}")
    
    test_correctness()
    test_correctness_batch()
    test_performance()
    test_batch_performance()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_benchmarks()