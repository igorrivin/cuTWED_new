#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import backends
from cutwed.backends import numpy_backend

# Try to import optional backends
backends = {"numpy": numpy_backend}

try:
    from cutwed.backends import torch_backend
    backends["pytorch"] = torch_backend
    import torch
    has_torch = True
    print("PyTorch backend available")
except ImportError:
    print("PyTorch not available, skipping PyTorch tests")
    has_torch = False

try:
    from cutwed.backends import jax_backend
    backends["jax"] = jax_backend
    import jax
    import jax.numpy as jnp
    has_jax = True
    print("JAX backend available")
except ImportError:
    print("JAX not available, skipping JAX tests")
    has_jax = False

def generate_time_series(n: int, dim: int = 1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random time series for testing."""
    np.random.seed(seed)
    A = np.random.randn(n, dim).astype(np.float32)
    TA = np.arange(n, dtype=np.float32)
    B = np.random.randn(n, dim).astype(np.float32)
    TB = np.arange(n, dtype=np.float32)
    return A, TA, B, TB

def convert_to_backend_format(A: np.ndarray, TA: np.ndarray, B: np.ndarray, TB: np.ndarray, 
                             backend: str) -> Tuple[Any, Any, Any, Any]:
    """Convert numpy arrays to the appropriate format for each backend."""
    if backend == "numpy":
        return A, TA, B, TB
    elif backend == "pytorch" and has_torch:
        return (torch.tensor(A), torch.tensor(TA), 
                torch.tensor(B), torch.tensor(TB))
    elif backend == "jax" and has_jax:
        return (jnp.array(A), jnp.array(TA), 
                jnp.array(B), jnp.array(TB))
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")

def test_simple():
    """Test a simple comparison with all available backends."""
    print("\n=== Simple Backend Comparison ===")
    
    # Generate a simple time series
    A, TA, B, TB = generate_time_series(100, dim=2)
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    results = {}
    
    for name, backend in backends.items():
        print(f"Testing {name} backend...")
        A_b, TA_b, B_b, TB_b = convert_to_backend_format(A, TA, B, TB, name)
        
        start_time = time.time()
        distance = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
        end_time = time.time()
        
        # Handle tensor to scalar conversion
        if name == "pytorch" and has_torch and torch.is_tensor(distance):
            distance = distance.item()  # Convert from tensor to scalar
        elif name == "jax" and has_jax and hasattr(distance, "item"):
            distance = float(distance)  # Convert from DeviceArray to scalar
        
        print(f"  Distance: {distance:.6f}")
        print(f"  Time: {(end_time - start_time)*1000:.2f} ms")
        
        results[name] = {"distance": distance, "time": (end_time - start_time) * 1000}
    
    # Print summary table
    if results:
        print("\nSummary:")
        
        ref_name = list(results.keys())[0]
        ref_dist = results[ref_name]["distance"]
        
        for name, res in results.items():
            rel_diff = abs(res["distance"] - ref_dist) / (abs(ref_dist) + 1e-6)
            print(f"  {name:8} | Distance: {res['distance']:.6f} | Time: {res['time']:.2f} ms | Rel Diff: {rel_diff:.6e}")

def performance_test():
    """Test performance across different time series lengths."""
    print("\n=== Performance Test ===")
    
    lengths = [50, 100, 200, 500]
    dimensions = [1, 3]
    
    results = {}
    
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    for dim in dimensions:
        print(f"\nDimension: {dim}")
        for length in lengths:
            print(f"  Length: {length}")
            
            A, TA, B, TB = generate_time_series(length, dim=dim)
            
            for name, backend in backends.items():
                A_b, TA_b, B_b, TB_b = convert_to_backend_format(A, TA, B, TB, name)
                
                # Warm-up
                _ = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
                
                # Timed runs
                runs = 3
                times = []
                
                for _ in range(runs):
                    start = time.time()
                    _ = backend.twed(A_b, TA_b, B_b, TB_b, nu, lamb, degree)
                    times.append(time.time() - start)
                
                avg_time = sum(times) / len(times)
                key = (name, dim, length)
                results[key] = avg_time * 1000  # Convert to ms
                
                print(f"    {name:8} | {avg_time*1000:.2f} ms")
    
    # Create a DataFrame for easier analysis
    data = []
    for (backend, dim, length), time_ms in results.items():
        data.append({
            "Backend": backend,
            "Dimensions": dim,
            "Length": length,
            "Time (ms)": time_ms
        })
    
    df = pd.DataFrame(data)
    
    # Print summary by backend
    print("\nPerformance Summary by Backend:")
    for backend in df["Backend"].unique():
        subset = df[df["Backend"] == backend]
        print(f"  {backend:8} | Avg: {subset['Time (ms)'].mean():.2f} ms | Min: {subset['Time (ms)'].min():.2f} ms | Max: {subset['Time (ms)'].max():.2f} ms")
    
    # Simple bar plot for different lengths, 1D
    plt.figure(figsize=(10, 6))
    for backend in df["Backend"].unique():
        subset = df[(df["Backend"] == backend) & (df["Dimensions"] == 1)]
        if not subset.empty:
            plt.plot(subset["Length"], subset["Time (ms)"], marker='o', label=backend)
    
    plt.xlabel("Time Series Length")
    plt.ylabel("Time (ms)")
    plt.title("Performance Comparison (1D)")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_1d.png")
    print("\nSaved plot to performance_1d.png")

if __name__ == "__main__":
    test_simple()
    performance_test()