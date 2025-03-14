#ifndef CUTWED_HPP
#define CUTWED_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cutwed {

/**
 * @brief Triangle optimization options for batch processing
 */
enum class TriangleOpt {
    TRIU = -2,  ///< Upper triangular optimization
    TRIL = -1,  ///< Lower triangular optimization
    NOPT = 0    ///< No optimization
};

/**
 * @brief Error handling class for cuTWED
 */
class CuTWEDError : public std::runtime_error {
public:
    explicit CuTWEDError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Main cuTWED algorithm implementation
 * 
 * @tparam T Precision type (float or double)
 */
template<typename T>
class TWED {
public:
    /**
     * @brief Compute TWED distance between two time series
     * 
     * @param A First time series values (nA x dim)
     * @param nA Length of first time series
     * @param TA First time series timestamps (nA)
     * @param B Second time series values (nB x dim)
     * @param nB Length of second time series
     * @param TB Second time series timestamps (nB)
     * @param nu Elasticity parameter
     * @param lambda Stiffness parameter
     * @param degree LP-norm degree (default=2)
     * @param dim Dimensionality of time series points
     * @return T The TWED distance
     */
    static T compute(
        const T* A, int nA, const T* TA,
        const T* B, int nB, const T* TB,
        T nu, T lambda, int degree, int dim
    );

    /**
     * @brief Compute TWED distance between two time series (device memory)
     * 
     * @param A_dev First time series values (device memory)
     * @param nA Length of first time series
     * @param TA_dev First time series timestamps (device memory)
     * @param B_dev Second time series values (device memory)
     * @param nB Length of second time series
     * @param TB_dev Second time series timestamps (device memory)
     * @param nu Elasticity parameter
     * @param lambda Stiffness parameter
     * @param degree LP-norm degree (default=2)
     * @param dim Dimensionality of time series points
     * @return T The TWED distance
     */
    static T compute_dev(
        const T* A_dev, int nA, const T* TA_dev,
        const T* B_dev, int nB, const T* TB_dev,
        T nu, T lambda, int degree, int dim
    );

    /**
     * @brief Compute batch TWED distances
     * 
     * @param AA Batch of first time series (nAA x nA x dim)
     * @param nA Length of first time series
     * @param TAA Batch of first timestamps (nAA x nA)
     * @param BB Batch of second time series (nBB x nB x dim)
     * @param nB Length of second time series
     * @param TBB Batch of second timestamps (nBB x nB)
     * @param nu Elasticity parameter
     * @param lambda Stiffness parameter
     * @param degree LP-norm degree (default=2)
     * @param dim Dimensionality of time series points
     * @param nAA Number of time series in first batch
     * @param nBB Number of time series in second batch
     * @param RRes Result matrix (nAA x nBB) - must be pre-allocated
     * @param tri Triangle optimization option
     * @return int Error code (0 for success)
     */
    static int batch_compute(
        const T* AA, int nA, const T* TAA,
        const T* BB, int nB, const T* TBB,
        T nu, T lambda, int degree, int dim,
        int nAA, int nBB, T* RRes, TriangleOpt tri = TriangleOpt::NOPT
    );

    /**
     * @brief Compute batch TWED distances (device memory)
     * 
     * @param AA_dev Batch of first time series (device memory)
     * @param nA Length of first time series
     * @param TAA_dev Batch of first timestamps (device memory)
     * @param BB_dev Batch of second time series (device memory)
     * @param nB Length of second time series
     * @param TBB_dev Batch of second timestamps (device memory)
     * @param nu Elasticity parameter
     * @param lambda Stiffness parameter
     * @param degree LP-norm degree (default=2)
     * @param dim Dimensionality of time series points
     * @param nAA Number of time series in first batch
     * @param nBB Number of time series in second batch
     * @param RRes Result matrix (nAA x nBB) - must be pre-allocated
     * @param tri Triangle optimization option
     * @return int Error code (0 for success)
     */
    static int batch_compute_dev(
        const T* AA_dev, int nA, const T* TAA_dev,
        const T* BB_dev, int nB, const T* TBB_dev,
        T nu, T lambda, int degree, int dim,
        int nAA, int nBB, T* RRes, TriangleOpt tri = TriangleOpt::NOPT
    );

    /**
     * @brief Allocate device memory for TWED computation
     * 
     * @param nA Length of first time series
     * @param A_dev Pointer to device memory for first time series
     * @param TA_dev Pointer to device memory for first timestamps
     * @param nB Length of second time series
     * @param B_dev Pointer to device memory for second time series
     * @param TB_dev Pointer to device memory for second timestamps
     * @param dim Dimensionality of time series points
     * @param nAA Number of time series in first batch
     * @param nBB Number of time series in second batch
     */
    static void malloc_dev(
        int nA, T** A_dev, T** TA_dev,
        int nB, T** B_dev, T** TB_dev,
        int dim, int nAA = 1, int nBB = 1
    );

    /**
     * @brief Free device memory allocated by malloc_dev
     * 
     * @param A_dev Device memory for first time series
     * @param TA_dev Device memory for first timestamps
     * @param B_dev Device memory for second time series
     * @param TB_dev Device memory for second timestamps
     */
    static void free_dev(
        T* A_dev, T* TA_dev,
        T* B_dev, T* TB_dev
    );

    /**
     * @brief Copy host data to device memory
     * 
     * @param nA Length of first time series
     * @param A Host memory for first time series
     * @param A_dev Device memory for first time series
     * @param TA Host memory for first timestamps
     * @param TA_dev Device memory for first timestamps
     * @param nB Length of second time series
     * @param B Host memory for second time series
     * @param B_dev Device memory for second time series
     * @param TB Host memory for second timestamps
     * @param TB_dev Device memory for second timestamps
     * @param dim Dimensionality of time series points
     * @param nAA Number of time series in first batch
     * @param nBB Number of time series in second batch
     */
    static void copy_to_dev(
        int nA, const T* A, T* A_dev, const T* TA, T* TA_dev,
        int nB, const T* B, T* B_dev, const T* TB, T* TB_dev,
        int dim, int nAA = 1, int nBB = 1
    );

private:
    // Dimension and batch limits
    static constexpr int DIMENSION_LIMIT = 64; // Increased from original 32
    static constexpr int BATCH_LIMIT = 131070; // Increased from original 65535
    
    // Private implementation functions (defined in .cu file)
    static T evalZ(
        const T* A, const T* DA, int nA, const T* TA,
        const T* B, const T* DB, int nB, const T* TB,
        T nu, int degree, T lambda, int dim
    );
    
    static void grid_evalZ(
        const T* A, const T* DA, int nA, const T* TA,
        const T* BB, const T* DBB, int nB, const T* TBB,
        T nu, int degree, T lambda, int dim,
        T* Res_dev, int nAA, int nBB, int tril
    );
};

// C-style interface for backward compatibility
extern "C" {
    // Double precision functions
    double twed(double A[], int nA, double TA[],
                double B[], int nB, double TB[],
                double nu, double lambda, int degree, int dim);
                
    double twed_dev(double A_dev[], int nA, double TA_dev[],
                    double B_dev[], int nB, double TB_dev[],
                    double nu, double lambda, int degree, int dim);
                    
    int twed_batch(double AA[], int nA, double TAA[],
                   double BB[], int nB, double TBB[],
                   double nu, double lambda, int degree, int dim,
                   int nAA, int nBB, double* RRes, int tri);
                   
    int twed_batch_dev(double AA_dev[], int nA, double TAA_dev[],
                       double BB_dev[], int nB, double TBB_dev[],
                       double nu, double lambda, int degree, int dim,
                       int nAA, int nBB, double* RRes, int tri);
                       
    void twed_malloc_dev(int nA, double **A_dev, double **TA_dev,
                         int nB, double **B_dev, double **TB_dev,
                         int dim, int nAA, int nBB);
                         
    void twed_free_dev(double *A_dev, double *TA_dev,
                       double *B_dev, double *TB_dev);
                       
    void twed_copy_to_dev(int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                          int nB, double B[], double B_dev[], double TB[], double TB_dev[],
                          int dim, int nAA, int nBB);
    
    // Single precision functions
    float twedf(float A[], int nA, float TA[],
                float B[], int nB, float TB[],
                float nu, float lambda, int degree, int dim);
                
    float twed_devf(float A_dev[], int nA, float TA_dev[],
                    float B_dev[], int nB, float TB_dev[],
                    float nu, float lambda, int degree, int dim);
                    
    int twed_batchf(float AA[], int nA, float TAA[],
                    float BB[], int nB, float TBB[],
                    float nu, float lambda, int degree, int dim,
                    int nAA, int nBB, float* RRes, int tri);
                    
    int twed_batch_devf(float AA_dev[], int nA, float TAA_dev[],
                        float BB_dev[], int nB, float TBB_dev[],
                        float nu, float lambda, int degree, int dim,
                        int nAA, int nBB, float* RRes, int tri);
                        
    void twed_malloc_devf(int nA, float **A_dev, float **TA_dev,
                          int nB, float **B_dev, float **TB_dev,
                          int dim, int nAA, int nBB);
                          
    void twed_free_devf(float *A_dev, float *TA_dev,
                        float *B_dev, float *TB_dev);
                        
    void twed_copy_to_devf(int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                           int nB, float B[], float B_dev[], float TB[], float TB_dev[],
                           int dim, int nAA, int nBB);
}

} // namespace cutwed

#endif // CUTWED_HPP