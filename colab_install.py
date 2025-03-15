#!/usr/bin/env python
"""
Special installation script for Google Colab
"""

import os
import sys
import subprocess
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Current directory: {os.getcwd()}")

# Check if running in Colab
try:
    import google.colab
    is_colab = True
    print("Running in Google Colab environment")
except ImportError:
    is_colab = False
    print("Not running in Google Colab")

# Check for GPU
try:
    print("Checking for NVIDIA GPU...")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"GPU detected: {result.stdout.splitlines()[0]}")
        has_gpu = True
    else:
        print("No GPU detected (nvidia-smi failed)")
        has_gpu = False
except Exception as e:
    print(f"Error checking for GPU: {e}")
    has_gpu = False

# Install required packages
print("\nInstalling required packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'cmake'])

# Clone the repository (if not already in it)
repo_dir = os.path.join(os.getcwd(), 'cuTWED_new')
if not os.path.exists(repo_dir):
    print("\nCloning repository...")
    subprocess.check_call(['git', 'clone', 'https://github.com/igorrivin/cuTWED_new.git'])
    os.chdir(repo_dir)
else:
    print(f"\nRepository directory already exists: {repo_dir}")
    os.chdir(repo_dir)

# Build and install (CPU-only mode for simplicity)
print("\nBuilding with CMake...")
build_dir = os.path.join(os.getcwd(), 'build')
os.makedirs(build_dir, exist_ok=True)
os.chdir(build_dir)

# Configure with CMake
print("Running CMake configure...")
cmake_cmd = ['cmake', '..',
             '-DCUTWED_USE_CUDA=OFF',
             '-DCUTWED_BUILD_PYTHON=ON']
subprocess.check_call(cmake_cmd)

# Build
print("\nRunning CMake build...")
subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'])

# Install Python package
print("\nInstalling non-CUDA Python package...")
os.chdir('..')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])

print("\nVerifying installation...")
try:
    import cutwed
    print(f"Available backends: {cutwed.get_available_backends()}")
    print(f"Current backend: {cutwed.get_backend_name()}")
    print("\nInstallation successful!")
except ImportError as e:
    print(f"Error importing cutwed: {e}")
    print("Installation failed!")

print("\nTesting with simple example...")
try:
    import numpy as np
    import cutwed
    
    # Create simple time series
    A = np.random.randn(10, 2).astype(np.float32)
    TA = np.arange(10, dtype=np.float32)
    B = np.random.randn(10, 2).astype(np.float32)
    TB = np.arange(10, dtype=np.float32)
    
    # Compute distance
    distance = cutwed.twed(A, TA, B, TB, 1.0, 1.0, 2)
    print(f"TWED distance: {distance}")
    print("Test successful!")
except Exception as e:
    print(f"Test failed: {e}")