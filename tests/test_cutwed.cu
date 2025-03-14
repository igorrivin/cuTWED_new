#include "cuTWED.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <chrono>

using namespace cutwed;

// Helper function to generate random data
template<typename T>
void generate_random_data(std::vector<T>& A, std::vector<T>& TA, 
                         std::vector<T>& B, std::vector<T>& TB,
                         int nA, int nB, int dim, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    
    A.resize(nA * dim);
    TA.resize(nA);
    B.resize(nB * dim);
    TB.resize(nB);
    
    // Fill with random data
    for (int i = 0; i < nA * dim; ++i) {
        A[i] = dist(gen);
    }
    
    for (int i = 0; i < nB * dim; ++i) {
        B[i] = dist(gen);
    }
    
    // Fill time with sequential values
    for (int i = 0; i < nA; ++i) {
        TA[i] = static_cast<T>(i);
    }
    
    for (int i = 0; i < nB; ++i) {
        TB[i] = static_cast<T>(i);
    }
}

// Helper function to generate batch data
template<typename T>
void generate_batch_data(std::vector<T>& AA, std::vector<T>& TAA, 
                        std::vector<T>& BB, std::vector<T>& TBB,
                        int nAA, int nA, int nBB, int nB, int dim, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    
    AA.resize(nAA * nA * dim);
    TAA.resize(nAA * nA);
    BB.resize(nBB * nB * dim);
    TBB.resize(nBB * nB);
    
    // Fill with random data
    for (int i = 0; i < nAA * nA * dim; ++i) {
        AA[i] = dist(gen);
    }
    
    for (int i = 0; i < nBB * nB * dim; ++i) {
        BB[i] = dist(gen);
    }
    
    // Fill time with sequential values
    for (int a = 0; a < nAA; ++a) {
        for (int i = 0; i < nA; ++i) {
            TAA[a * nA + i] = static_cast<T>(i);
        }
    }
    
    for (int b = 0; b < nBB; ++b) {
        for (int i = 0; i < nB; ++i) {
            TBB[b * nB + i] = static_cast<T>(i);
        }
    }
}

// Test basic TWED functionality
template<typename T>
bool test_basic() {
    std::cout << "Testing basic TWED functionality with " 
              << (std::is_same<T, float>::value ? "float" : "double") 
              << " precision..." << std::endl;
    
    // Generate random data
    std::vector<T> A, TA, B, TB;
    int nA = 100;
    int nB = 80;
    int dim = 3;
    
    generate_random_data<T>(A, TA, B, TB, nA, nB, dim);
    
    T nu = 1.0;
    T lambda = 1.0;
    int degree = 2;
    
    // Test host computation
    try {
        auto start = std::chrono::high_resolution_clock::now();
        T dist = TWED<T>::compute(A.data(), nA, TA.data(), B.data(), nB, TB.data(), nu, lambda, degree, dim);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "TWED distance: " << dist << std::endl;
        std::cout << "Computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // Allocate device memory
        T *A_dev, *TA_dev, *B_dev, *TB_dev;
        TWED<T>::malloc_dev(nA, &A_dev, &TA_dev, nB, &B_dev, &TB_dev, dim);
        
        // Copy data to device
        TWED<T>::copy_to_dev(nA, A.data(), A_dev, TA.data(), TA_dev, 
                           nB, B.data(), B_dev, TB.data(), TB_dev, dim);
        
        // Test device computation
        start = std::chrono::high_resolution_clock::now();
        T dist_dev = TWED<T>::compute_dev(A_dev, nA, TA_dev, B_dev, nB, TB_dev, nu, lambda, degree, dim);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        std::cout << "TWED device distance: " << dist_dev << std::endl;
        std::cout << "Device computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // Free device memory
        TWED<T>::free_dev(A_dev, TA_dev, B_dev, TB_dev);
        
        // Compare results
        T rel_diff = std::abs(dist - dist_dev) / (std::abs(dist) + 1e-6);
        std::cout << "Relative difference: " << rel_diff << std::endl;
        
        return rel_diff < 1e-5;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

// Test batch TWED functionality
template<typename T>
bool test_batch() {
    std::cout << "Testing batch TWED functionality with " 
              << (std::is_same<T, float>::value ? "float" : "double") 
              << " precision..." << std::endl;
    
    // Generate random batch data
    std::vector<T> AA, TAA, BB, TBB;
    int nAA = 10;
    int nBB = 8;
    int nA = 50;
    int nB = 40;
    int dim = 3;
    
    generate_batch_data<T>(AA, TAA, BB, TBB, nAA, nA, nBB, nB, dim);
    
    T nu = 1.0;
    T lambda = 1.0;
    int degree = 2;
    
    // Allocate result matrix
    std::vector<T> result(nAA * nBB);
    
    // Test batch computation
    try {
        auto start = std::chrono::high_resolution_clock::now();
        int ret = TWED<T>::batch_compute(
            AA.data(), nA, TAA.data(),
            BB.data(), nB, TBB.data(),
            nu, lambda, degree, dim,
            nAA, nBB, result.data()
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "Batch computation returned: " << ret << std::endl;
        std::cout << "First distance: " << result[0] << std::endl;
        std::cout << "Computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // Allocate device memory
        T *AA_dev, *TAA_dev, *BB_dev, *TBB_dev;
        TWED<T>::malloc_dev(nA, &AA_dev, &TAA_dev, nB, &BB_dev, &TBB_dev, dim, nAA, nBB);
        
        // Copy data to device
        TWED<T>::copy_to_dev(nA, AA.data(), AA_dev, TAA.data(), TAA_dev, 
                           nB, BB.data(), BB_dev, TBB.data(), TB_dev, dim, nAA, nBB);
        
        // Allocate device result matrix
        std::vector<T> result_dev(nAA * nBB);
        
        // Test batch device computation
        start = std::chrono::high_resolution_clock::now();
        ret = TWED<T>::batch_compute_dev(
            AA_dev, nA, TAA_dev,
            BB_dev, nB, TBB_dev,
            nu, lambda, degree, dim,
            nAA, nBB, result_dev.data()
        );
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        std::cout << "Batch device computation returned: " << ret << std::endl;
        std::cout << "First device distance: " << result_dev[0] << std::endl;
        std::cout << "Device computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // Free device memory
        TWED<T>::free_dev(AA_dev, TAA_dev, BB_dev, TBB_dev);
        
        // Compare results
        T max_diff = 0;
        for (int i = 0; i < nAA * nBB; ++i) {
            T rel_diff = std::abs(result[i] - result_dev[i]) / (std::abs(result[i]) + 1e-6);
            max_diff = std::max(max_diff, rel_diff);
        }
        std::cout << "Maximum relative difference: " << max_diff << std::endl;
        
        return max_diff < 1e-5;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    bool run_basic = false;
    bool run_batch = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--basic") run_basic = true;
        else if (arg == "--batch") run_batch = true;
        else if (arg == "--all") {
            run_basic = true;
            run_batch = true;
        }
    }
    
    // If no test specified, run all
    if (!run_basic && !run_batch) {
        run_basic = true;
        run_batch = true;
    }
    
    bool all_pass = true;
    
    // Run basic tests
    if (run_basic) {
        bool test_double = test_basic<double>();
        bool test_float = test_basic<float>();
        all_pass &= test_double && test_float;
        
        std::cout << "Basic tests " 
                  << (test_double && test_float ? "PASSED" : "FAILED") 
                  << std::endl;
    }
    
    // Run batch tests
    if (run_batch) {
        bool test_double = test_batch<double>();
        bool test_float = test_batch<float>();
        all_pass &= test_double && test_float;
        
        std::cout << "Batch tests " 
                  << (test_double && test_float ? "PASSED" : "FAILED") 
                  << std::endl;
    }
    
    std::cout << "All tests " << (all_pass ? "PASSED" : "FAILED") << std::endl;
    
    return all_pass ? 0 : 1;
}