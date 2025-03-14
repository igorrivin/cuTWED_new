#!/usr/bin/env python
"""
Test script for cuTWED on a realistic time series classification task.
This demonstrates the practical application of the algorithm with different backends.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Add the Python package directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

# Import backends directly
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


def generate_synthetic_dataset(n_samples: int = 150, length: int = 100, n_classes: int = 3, 
                              noise_level: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset with different pattern classes."""
    np.random.seed(seed)
    time = np.arange(length)
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    # Class 0: Sine wave
    for i in range(samples_per_class):
        noise = np.random.normal(0, noise_level, length)
        freq = 1.0 + 0.1 * (i % 5)  # Slight variation in frequency
        signal = np.sin(2 * np.pi * freq * time / length) + noise
        X.append(signal)
        y.append(0)
    
    # Class 1: Square wave
    for i in range(samples_per_class):
        noise = np.random.normal(0, noise_level, length)
        square = np.zeros(length)
        offset = i % 10  # Slight variation in phase
        square[(length//4 + offset):(3*length//4 + offset)] = 1
        signal = square + noise
        X.append(signal)
        y.append(1)
    
    # Class 2: Triangle wave
    for i in range(samples_per_class):
        noise = np.random.normal(0, noise_level, length)
        triangle = np.zeros(length)
        scaling = 0.8 + 0.4 * (i % 5) / 5  # Slight variation in amplitude
        for j in range(length):
            if j < length/2:
                triangle[j] = scaling * 2 * j / length
            else:
                triangle[j] = scaling * (2 - 2 * j / length)
        signal = triangle + noise
        X.append(signal)
        y.append(2)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    return X, y


def plot_examples(X: np.ndarray, y: np.ndarray, length: int = 100):
    """Plot examples from each class."""
    plt.figure(figsize=(15, 5))
    
    n_classes = len(np.unique(y))
    time = np.arange(length)
    
    for i in range(n_classes):
        plt.subplot(1, n_classes, i+1)
        for j in range(3):  # Plot 3 examples from each class
            idx = np.where(y == i)[0][j]
            plt.plot(time, X[idx], alpha=0.7)
        
        class_names = ["Sine", "Square", "Triangle"]
        plt.title(f"Class {i}: {class_names[i]}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dataset_examples.png')
    plt.close()


def twed_distance_matrix(X_train: np.ndarray, X_test: np.ndarray, 
                         backend_name: str, backend_module) -> np.ndarray:
    """Compute the TWED distance matrix between training and test sets."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    length = X_train.shape[1]
    
    # Create time vectors
    time = np.arange(length, dtype=np.float32)
    
    # Convert inputs to the appropriate format
    if backend_name == "numpy":
        X_train_b = X_train
        X_test_b = X_test
        time_b = time
    elif backend_name == "pytorch" and HAS_TORCH:
        X_train_b = torch.tensor(X_train)
        X_test_b = torch.tensor(X_test)
        time_b = torch.tensor(time)
    elif backend_name == "jax" and HAS_JAX:
        X_train_b = jnp.array(X_train)
        X_test_b = jnp.array(X_test)
        time_b = jnp.array(time)
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend_name}")
    
    # Parameters for TWED
    nu = 1.0
    lamb = 1.0
    degree = 2
    
    # Compute distances
    print(f"Computing distances using {backend_name} backend...")
    start_time = time.time()
    
    # Reshape for batch processing
    AA = X_train_b.reshape(n_train, length, 1)
    TAA = np.tile(time, (n_train, 1)) if backend_name == "numpy" else (
        torch.tile(time_b, (n_train, 1)) if backend_name == "pytorch" else
        jnp.tile(time_b, (n_train, 1))
    )
    
    BB = X_test_b.reshape(n_test, length, 1)
    TBB = np.tile(time, (n_test, 1)) if backend_name == "numpy" else (
        torch.tile(time_b, (n_test, 1)) if backend_name == "pytorch" else
        jnp.tile(time_b, (n_test, 1))
    )
    
    # Compute distance matrix
    distances = backend_module.twed_batch(AA, TAA, BB, TBB, nu, lamb, degree)
    
    # Convert to numpy if needed
    if backend_name == "pytorch" and HAS_TORCH:
        distances = distances.cpu().numpy()
    elif backend_name == "jax" and HAS_JAX:
        distances = np.array(distances)
    
    elapsed = time.time() - start_time
    print(f"  Computed {n_train}x{n_test} distances in {elapsed:.2f} seconds")
    
    return distances


def run_knn_classification(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray,
                         distance_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """Run k-nearest neighbors classification using precomputed distances."""
    # Create and fit the classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    knn.fit(distance_matrix.T, y_test)  # Note: transposed because KNN expects (n_test, n_train)
    
    # Make predictions
    y_pred = knn.predict(distance_matrix.T)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, conf_matrix


def run_realistic_test():
    """Run a realistic time series classification test."""
    print("\n=== Running Realistic Classification Test ===")
    
    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples=300, length=100, noise_level=0.2)
    
    # Plot examples
    plot_examples(X, y)
    print("Dataset examples plotted to 'dataset_examples.png'")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test each backend
    results = {}
    
    for name, backend in BACKENDS.items():
        print(f"\nTesting {name} backend...")
        
        # Compute distance matrix
        distance_matrix = twed_distance_matrix(X_train, X_test, name, backend)
        
        # Run classification
        accuracy, conf_matrix = run_knn_classification(X_train, y_train, X_test, y_test, distance_matrix)
        
        print(f"  Classification accuracy: {accuracy:.4f}")
        print("  Confusion matrix:")
        print(conf_matrix)
        
        results[name] = {
            "accuracy": accuracy,
            "conf_matrix": conf_matrix
        }
    
    # Print summary
    print("\nSummary:")
    for name, result in results.items():
        print(f"  {name} backend: {result['accuracy']:.4f} accuracy")
    
    return results


if __name__ == "__main__":
    results = run_realistic_test()