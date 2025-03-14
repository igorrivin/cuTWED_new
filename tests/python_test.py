import unittest
import numpy as np
import sys
import os

# Add the Python package to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

try:
    from cutwed import twed, twed_batch, TriangleOpt
    has_cutwed = True
except ImportError:
    has_cutwed = False

try:
    import cupy as cp
    from cutwed import twed_cupy, twed_batch_cupy
    has_cupy = True
except ImportError:
    has_cupy = False


@unittest.skipIf(not has_cutwed, "cutwed not installed")
class TestCuTWED(unittest.TestCase):
    def setUp(self):
        # Generate random time series
        np.random.seed(42)
        self.A = np.random.randn(100, 3).astype(np.float32)
        self.TA = np.arange(100, dtype=np.float32)
        self.B = np.random.randn(80, 3).astype(np.float32)
        self.TB = np.arange(80, dtype=np.float32)
        
        # Parameters for TWED
        self.nu = 1.0
        self.lamb = 1.0
        self.degree = 2
        
        # Generate batch data
        self.AA = np.random.randn(10, 50, 3).astype(np.float32)
        self.TAA = np.tile(np.arange(50, dtype=np.float32), (10, 1))
        self.BB = np.random.randn(8, 40, 3).astype(np.float32)
        self.TBB = np.tile(np.arange(40, dtype=np.float32), (8, 1))
    
    def test_twed_float(self):
        """Test basic TWED with float32 precision."""
        distance = twed(self.A, self.TA, self.B, self.TB, self.nu, self.lamb, self.degree)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
    
    def test_twed_double(self):
        """Test basic TWED with float64 precision."""
        A_double = self.A.astype(np.float64)
        TA_double = self.TA.astype(np.float64)
        B_double = self.B.astype(np.float64)
        TB_double = self.TB.astype(np.float64)
        
        distance = twed(A_double, TA_double, B_double, TB_double, self.nu, self.lamb, self.degree)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
    
    def test_twed_batch(self):
        """Test batch TWED."""
        distances = twed_batch(self.AA, self.TAA, self.BB, self.TBB, self.nu, self.lamb, self.degree)
        self.assertEqual(distances.shape, (10, 8))
        self.assertGreater(np.min(distances), 0)
    
    def test_triangle_opt(self):
        """Test triangle optimization."""
        # Create square batch
        AA = np.random.randn(5, 30, 3).astype(np.float32)
        TAA = np.tile(np.arange(30, dtype=np.float32), (5, 1))
        
        # Compute full distance matrix
        full = twed_batch(AA, TAA, AA, TAA, self.nu, self.lamb, self.degree)
        
        # Compute upper triangular
        upper = twed_batch(AA, TAA, AA, TAA, self.nu, self.lamb, self.degree, TriangleOpt.TRIU)
        
        # Compute lower triangular
        lower = twed_batch(AA, TAA, AA, TAA, self.nu, self.lamb, self.degree, TriangleOpt.TRIL)
        
        # Check that matrices are equal
        np.testing.assert_allclose(full, upper, rtol=1e-5)
        np.testing.assert_allclose(full, lower, rtol=1e-5)
        
        # Check diagonal
        np.testing.assert_allclose(np.diag(full), np.zeros(5), atol=1e-5)


@unittest.skipIf(not has_cupy, "cupy not installed")
class TestCuTWEDCupy(unittest.TestCase):
    def setUp(self):
        # Generate random time series
        np.random.seed(42)
        self.A_np = np.random.randn(100, 3).astype(np.float32)
        self.TA_np = np.arange(100, dtype=np.float32)
        self.B_np = np.random.randn(80, 3).astype(np.float32)
        self.TB_np = np.arange(80, dtype=np.float32)
        
        # Convert to CuPy arrays
        self.A = cp.asarray(self.A_np)
        self.TA = cp.asarray(self.TA_np)
        self.B = cp.asarray(self.B_np)
        self.TB = cp.asarray(self.TB_np)
        
        # Parameters for TWED
        self.nu = 1.0
        self.lamb = 1.0
        self.degree = 2
        
        # Generate batch data
        self.AA_np = np.random.randn(10, 50, 3).astype(np.float32)
        self.TAA_np = np.tile(np.arange(50, dtype=np.float32), (10, 1))
        self.BB_np = np.random.randn(8, 40, 3).astype(np.float32)
        self.TBB_np = np.tile(np.arange(40, dtype=np.float32), (8, 1))
        
        # Convert to CuPy arrays
        self.AA = cp.asarray(self.AA_np)
        self.TAA = cp.asarray(self.TAA_np)
        self.BB = cp.asarray(self.BB_np)
        self.TBB = cp.asarray(self.TBB_np)
    
    def test_twed_cupy(self):
        """Test CuPy version of TWED."""
        distance_cupy = twed_cupy(self.A, self.TA, self.B, self.TB, self.nu, self.lamb, self.degree)
        distance_numpy = twed(self.A_np, self.TA_np, self.B_np, self.TB_np, self.nu, self.lamb, self.degree)
        
        self.assertIsInstance(distance_cupy, float)
        self.assertGreater(distance_cupy, 0)
        
        # Check that results are close
        self.assertAlmostEqual(distance_cupy, distance_numpy, delta=1e-5)
    
    def test_twed_batch_cupy(self):
        """Test CuPy version of batch TWED."""
        distances_cupy = twed_batch_cupy(self.AA, self.TAA, self.BB, self.TBB, self.nu, self.lamb, self.degree)
        distances_numpy = twed_batch(self.AA_np, self.TAA_np, self.BB_np, self.TBB_np, self.nu, self.lamb, self.degree)
        
        self.assertEqual(distances_cupy.shape, (10, 8))
        self.assertGreater(cp.min(distances_cupy).item(), 0)
        
        # Check that results are close
        np.testing.assert_allclose(cp.asnumpy(distances_cupy), distances_numpy, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()