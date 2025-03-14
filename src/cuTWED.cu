#include "cuTWED.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace cutwed {

// Helper structures for diagonal index mapping
template<typename T>
struct RCIdx {
    int row;
    int col;
};

template<typename T>
struct DiagIdx {
    int orth_diag;  // the "left" diagonals
    int idx;       // index along the diag
};

// Helper functions for diagonal index mapping
template<typename T>
__device__ __host__ inline RCIdx<T> map_diag_to_rc(int orth_diag, int idx) {
    return {orth_diag - idx, idx};
}

template<typename T>
__device__ __host__ inline DiagIdx<T> map_rc_to_diag(int row, int col) {
    return {row + col, col};
}

// Kernel implementations for computing LP norms and vector subtraction
template<typename T>
__device__ T lpnorm(const int p, const int dim, const T* P) {
    const T pf = static_cast<T>(p);
    T s = 0;
    
    #pragma unroll
    for (int d = 0; d < dim; d++) {
        s += pow(fabs(P[d]), pf);
    }
    
    return pow(s, static_cast<T>(1.0) / pf);
}

template<typename T>
__device__ void vsub(const int dim, const T* P1, const T* P2, T* P3) {
    #pragma unroll
    for (int d = 0; d < dim; d++) {
        P3[d] = P1[d] - P2[d];
    }
}

// Kernel for computing local distances
template<typename T>
__global__ void local_distance_kernel(
    const T* __restrict__ X, 
    const int n, 
    const int degree,
    const int dim,
    T* __restrict__ D, 
    const int nBatch
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int bat = blockIdx.y * blockDim.y + threadIdx.y;
    const int gtid = (n + 1) * bat + tid;
    
    // Check bounds
    if (tid > n || bat >= nBatch) return;
    
    T d;
    
    if (tid == 0) {
        d = 0;
    } else if (tid == 1) {
        d = lpnorm<T>(degree, dim, &X[bat * n * dim + (tid - 1) * dim]);
    } else {
        // Use shared memory for temporary storage
        T tmp[64]; // Using a fixed size for now, should be <= DIMENSION_LIMIT
        
        vsub<T>(dim, 
             &X[bat * n * dim + (tid - 1) * dim], 
             &X[bat * n * dim + (tid - 2) * dim], 
             tmp);
             
        d = lpnorm<T>(degree, dim, tmp);
    }
    
    D[gtid] = d;
}

// Kernel for dynamic programming computation
template<typename T>
__global__ void evalZ_kernel(
    int diagIdx,
    T* DP_diag_lag_2,
    T* DP_diag_lag,
    T* DP_diag,
    const T* __restrict__ A, 
    const T* __restrict__ DA, 
    int nA, 
    const T* __restrict__ TA,
    const T* __restrict__ B, 
    const T* __restrict__ DB, 
    int nB, 
    const T* __restrict__ TB,
    T nu, 
    int degree, 
    T lambda, 
    int dim, 
    int nBB
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int Bid = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (Bid >= nBB || tid > diagIdx) return;
    
    // Map from diagonal index to row/col
    const auto id = map_diag_to_rc<T>(diagIdx, tid);
    const int row = id.row;
    const int col = id.col;
    const int n = (nA + 1) + (nB + 1) - 1;
    
    // Check bounds again
    if (row > nA || col > nB) return;
    
    // Compute dynamic programming indexes
    const size_t tidDrm1 = Bid * n + map_rc_to_diag<T>(row - 1, col).idx;
    const size_t tidDcm1 = Bid * n + map_rc_to_diag<T>(row, col - 1).idx;
    const size_t tidDrm1cm1 = Bid * n + map_rc_to_diag<T>(row - 1, col - 1).idx;
    
    const T* Bptr = &B[Bid * nB * dim];
    const T* TBptr = &TB[Bid * nB];
    const T* DBpter = &DB[Bid * (nB + 1)];
    
    // Initialize DP distance
    T d = 0;
    T d2 = 0;
    const T recip = static_cast<T>(1.0) / degree;
    
    if (row == 0 && col == 0) {
        d = 0;
    } else if (row == 0 || col == 0) {
        d = INFINITY;
    } else {
        // Compute distance between points
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            d += pow(fabs(A[(row - 1) * dim + i] - Bptr[(col - 1) * dim + i]), degree);
            if (row > 1 && col > 1) {
                d2 += pow(fabs(A[(row - 2) * dim + i] - Bptr[(col - 2) * dim + i]), degree);
            }
        }
        d = pow(d, recip) + pow(d2, recip);
    }
    
    if (row < 1 || col < 1) {
        DP_diag[Bid * n + tid] = d;
        return;
    }
    
    // Compute dynamic programming updates
    T htrans;
    T dmin;
    T dist;
    
    // Case 1: Keep Both
    htrans = fabs((T)(TA[row - 1] - TBptr[col - 1]));
    if (col > 1 && row > 1) {
        htrans += fabs((T)(TA[row - 2] - TBptr[col - 2]));
    }
    dmin = DP_diag_lag_2[tidDrm1cm1] + d + nu * htrans;
    
    // Case 2: Delete point in A
    if (row > 1)
        htrans = ((T)(TA[row - 1] - TA[row - 2]));
    else 
        htrans = (T)TA[row - 1];
        
    dist = DA[row] + DP_diag_lag[tidDrm1] + lambda + nu * htrans;
    dmin = fmin(dmin, dist);
    
    // Case 3: Delete Point in B
    if (col > 1)
        htrans = ((T)(TBptr[col - 1] - TBptr[col - 2]));
    else 
        htrans = (T)TBptr[col - 1];
        
    dist = DBpter[col] + DP_diag_lag[tidDcm1] + lambda + nu * htrans;
    dmin = fmin(dmin, dist);
    
    // Assign minimal result
    DP_diag[Bid * n + tid] = dmin;
}

// Kernel for extracting results
template<typename T>
__global__ void result_agg_kernel(
    T* __restrict__ res,
    const T* __restrict__ DPP,
    int nBB,
    int res_offset,
    int stride
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= nBB) return;
    
    res[tid] = DPP[tid * stride + res_offset];
}

// Implementation of evalZ function (single time series pair)
template<typename T>
T evalZ(
    const T* A, const T* DA, int nA, const T* TA,
    const T* B, const T* DB, int nB, const T* TB,
    T nu, int degree, T lambda, int dim
) {
    const int n = (nA + 1) + (nB + 1) - 1;
    
    // Allocate device memory for diagonals with RAII
    DeviceMemory<T> DP_diag(n);
    DeviceMemory<T> DP_diag_lag(n);
    DeviceMemory<T> DP_diag_lag_2(n);
    
    // Create a CUDA stream
    CudaStream stream;
    
    // Process diagonals
    for (int diagIdx = 0; diagIdx < n; diagIdx++) {
        // Cycle the diagonals
        T* tmp_ptr = DP_diag_lag_2.get();
        DP_diag_lag_2 = DeviceMemory<T>(std::move(DP_diag_lag));
        DP_diag_lag = DeviceMemory<T>(std::move(DP_diag));
        DP_diag = DeviceMemory<T>(n);
        DP_diag_lag_2.get() = tmp_ptr;
        
        // Launch kernel
        dim3 block_dim(32);
        dim3 grid_dim((diagIdx + block_dim.x) / block_dim.x);
        
        evalZ_kernel<T><<<grid_dim, block_dim, 0, stream.get()>>>(
            diagIdx, DP_diag_lag_2.get(), DP_diag_lag.get(), DP_diag.get(),
            A, DA, nA, TA, B, DB, nB, TB, nu, degree, lambda, dim, 1
        );
        
        // Check for errors
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Get the final result
    T result;
    CUDA_CHECK(cudaMemcpy(&result, 
                        &DP_diag.get()[map_rc_to_diag<T>(nA, nB).idx], 
                        sizeof(T), 
                        cudaMemcpyDeviceToHost));
    
    return result;
}

// Implementation of grid_evalZ function (batch processing)
template<typename T>
void grid_evalZ(
    const T* A, const T* DA, int nA, const T* TA,
    const T* BB, const T* DBB, int nB, const T* TBB,
    T nu, int degree, T lambda, int dim,
    T* Res_dev, int nAA, int nBB, int tril
) {
    const int n = (nA + 1) + (nB + 1) - 1;
    
    // Allocate device memory for diagonals with RAII
    DeviceMemory<T> DP_diag(nBB * n);
    DeviceMemory<T> DP_diag_lag(nBB * n);
    DeviceMemory<T> DP_diag_lag_2(nBB * n);
    
    // Create a CUDA stream
    CudaStream stream;
    
    // Determine actual number of B time series to process
    int effective_nBB = tril != -1 ? tril : nBB;
    
    // Process diagonals
    for (int diagIdx = 0; diagIdx < n; diagIdx++) {
        // Cycle the diagonals
        T* tmp_ptr = DP_diag_lag_2.get();
        DP_diag_lag_2 = DeviceMemory<T>(std::move(DP_diag_lag));
        DP_diag_lag = DeviceMemory<T>(std::move(DP_diag));
        DP_diag = DeviceMemory<T>(nBB * n);
        DP_diag_lag_2.get() = tmp_ptr;
        
        // Launch kernel with 2D grid
        dim3 block_dim(32, 32);
        dim3 grid_dim(
            (diagIdx + block_dim.x) / block_dim.x,
            (effective_nBB + block_dim.y) / block_dim.y
        );
        
        evalZ_kernel<T><<<grid_dim, block_dim, 0, stream.get()>>>(
            diagIdx, DP_diag_lag_2.get(), DP_diag_lag.get(), DP_diag.get(),
            A, DA, nA, TA, BB, DBB, nB, TBB, nu, degree, lambda, dim, effective_nBB
        );
        
        // Check for errors
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Extract results
    dim3 block_dim(256);
    dim3 grid_dim((effective_nBB + block_dim.x - 1) / block_dim.x);
    
    result_agg_kernel<T><<<grid_dim, block_dim, 0, stream.get()>>>(
        Res_dev, DP_diag.get(), effective_nBB, map_rc_to_diag<T>(nA, nB).idx, n
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize
    stream.synchronize();
}

// Template implementations for TWED class methods

// Compute method
template<typename T>
T TWED<T>::compute(
    const T* A, int nA, const T* TA,
    const T* B, int nB, const T* TB,
    T nu, T lambda, int degree, int dim
) {
    // Validate inputs
    if (dim > DIMENSION_LIMIT) {
        throw CuTWEDError("Dimension exceeds limit: " + std::to_string(dim) +
                          " > " + std::to_string(DIMENSION_LIMIT));
    }
    
    if (degree <= 0) {
        throw CuTWEDError("Degree must be positive");
    }
    
    // Allocate device memory for time series
    DeviceMemory<T> A_dev(nA * dim);
    DeviceMemory<T> TA_dev(nA);
    DeviceMemory<T> B_dev(nB * dim);
    DeviceMemory<T> TB_dev(nB);
    
    // Copy data to device
    A_dev.copyFromHost(A, nA * dim);
    TA_dev.copyFromHost(TA, nA);
    B_dev.copyFromHost(B, nB * dim);
    TB_dev.copyFromHost(TB, nB);
    
    // Compute TWED on device
    T result = compute_dev(
        A_dev.get(), nA, TA_dev.get(),
        B_dev.get(), nB, TB_dev.get(),
        nu, lambda, degree, dim
    );
    
    return result;
}

// Compute_dev method
template<typename T>
T TWED<T>::compute_dev(
    const T* A_dev, int nA, const T* TA_dev,
    const T* B_dev, int nB, const T* TB_dev,
    T nu, T lambda, int degree, int dim
) {
    // Validate inputs
    if (dim > DIMENSION_LIMIT) {
        throw CuTWEDError("Dimension exceeds limit: " + std::to_string(dim) +
                          " > " + std::to_string(DIMENSION_LIMIT));
    }
    
    if (degree <= 0) {
        throw CuTWEDError("Degree must be positive");
    }
    
    // Create CUDA streams
    std::vector<CudaStream> streams(2);
    
    // Allocate device memory for distance arrays
    DeviceMemory<T> DA_dev(nA + 1);
    DeviceMemory<T> DB_dev(nB + 1);
    
    // Compute local distances for A
    dim3 block_dim_A(256);
    dim3 grid_dim_A((nA + block_dim_A.x - 1) / block_dim_A.x);
    
    local_distance_kernel<T><<<grid_dim_A, block_dim_A, 0, streams[0].get()>>>(
        A_dev, nA, degree, dim, DA_dev.get(), 1
    );
    
    // Compute local distances for B
    dim3 block_dim_B(256);
    dim3 grid_dim_B((nB + block_dim_B.x - 1) / block_dim_B.x);
    
    local_distance_kernel<T><<<grid_dim_B, block_dim_B, 0, streams[1].get()>>>(
        B_dev, nB, degree, dim, DB_dev.get(), 1
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize streams
    for (auto& stream : streams) {
        stream.synchronize();
    }
    
    // Compute TWED distance
    T result = evalZ<T>(
        A_dev, DA_dev.get(), nA, TA_dev,
        B_dev, DB_dev.get(), nB, TB_dev,
        nu, degree, lambda, dim
    );
    
    return result;
}

// Batch_compute method
template<typename T>
int TWED<T>::batch_compute(
    const T* AA, int nA, const T* TAA,
    const T* BB, int nB, const T* TBB,
    T nu, T lambda, int degree, int dim,
    int nAA, int nBB, T* RRes, TriangleOpt tri
) {
    // Validate inputs
    if (dim > DIMENSION_LIMIT) {
        throw CuTWEDError("Dimension exceeds limit: " + std::to_string(dim) +
                          " > " + std::to_string(DIMENSION_LIMIT));
    }
    
    if (nAA > BATCH_LIMIT || nBB > BATCH_LIMIT) {
        throw CuTWEDError("Batch size exceeds limit: nAA=" + std::to_string(nAA) +
                          " nBB=" + std::to_string(nBB) + 
                          " limit=" + std::to_string(BATCH_LIMIT));
    }
    
    if (degree <= 0) {
        throw CuTWEDError("Degree must be positive");
    }
    
    // Allocate device memory for time series
    DeviceMemory<T> AA_dev(nAA * nA * dim);
    DeviceMemory<T> TAA_dev(nAA * nA);
    DeviceMemory<T> BB_dev(nBB * nB * dim);
    DeviceMemory<T> TBB_dev(nBB * nB);
    
    // Copy data to device
    AA_dev.copyFromHost(AA, nAA * nA * dim);
    TAA_dev.copyFromHost(TAA, nAA * nA);
    BB_dev.copyFromHost(BB, nBB * nB * dim);
    TBB_dev.copyFromHost(TBB, nBB * nB);
    
    // Compute TWED on device
    int result = batch_compute_dev(
        AA_dev.get(), nA, TAA_dev.get(),
        BB_dev.get(), nB, TBB_dev.get(),
        nu, lambda, degree, dim,
        nAA, nBB, RRes, tri
    );
    
    return result;
}

// Batch_compute_dev method
template<typename T>
int TWED<T>::batch_compute_dev(
    const T* AA_dev, int nA, const T* TAA_dev,
    const T* BB_dev, int nB, const T* TBB_dev,
    T nu, T lambda, int degree, int dim,
    int nAA, int nBB, T* RRes, TriangleOpt tri
) {
    // Validate inputs
    if (dim > DIMENSION_LIMIT) {
        throw CuTWEDError("Dimension exceeds limit: " + std::to_string(dim) +
                          " > " + std::to_string(DIMENSION_LIMIT));
    }
    
    if (nAA > BATCH_LIMIT || nBB > BATCH_LIMIT) {
        throw CuTWEDError("Batch size exceeds limit: nAA=" + std::to_string(nAA) +
                          " nBB=" + std::to_string(nBB) + 
                          " limit=" + std::to_string(BATCH_LIMIT));
    }
    
    if (degree <= 0) {
        throw CuTWEDError("Degree must be positive");
    }
    
    // Check triangle optimization parameters
    int tril = -1;  // Default to no triangle optimization
    
    if (tri == TriangleOpt::TRIL || tri == TriangleOpt::TRIU) {
        if (nAA != nBB) {
            throw CuTWEDError("Triangle optimization requires symmetric batch (nAA == nBB)");
        }
        tril = 0;  // Enable triangle optimization
    }
    
    // Create CUDA streams
    std::vector<CudaStream> streams(2);
    
    // Allocate device memory
    DeviceMemory<T> DA_dev(nA + 1);
    DeviceMemory<T> DBB_dev(nBB * (nB + 1));
    DeviceMemory<T> Res_dev_write(nBB);
    DeviceMemory<T> Res_dev_read(nBB);
    
    // Compute local distances for B time series
    dim3 block_dim_B(32, 32);
    dim3 grid_dim_B(
        (nB + block_dim_B.x - 1) / block_dim_B.x,
        (nBB + block_dim_B.y - 1) / block_dim_B.y
    );
    
    local_distance_kernel<T><<<grid_dim_B, block_dim_B, 0, streams[0].get()>>>(
        BB_dev, nB, degree, dim, DBB_dev.get(), nBB
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    
    // Allocate results array
    DeviceMemory<T> results_dev(nAA * nBB);
    
    // Process each A time series
    for (int a = 0; a < nAA; a++) {
        // Compute local distances for current A time series
        dim3 block_dim_A(256);
        dim3 grid_dim_A((nA + block_dim_A.x - 1) / block_dim_A.x);
        
        local_distance_kernel<T><<<grid_dim_A, block_dim_A, 0, streams[1].get()>>>(
            &AA_dev[a * nA * dim], nA, degree, dim, DA_dev.get(), 1
        );
        
        // Check for errors
        CUDA_CHECK(cudaGetLastError());
        
        // Synchronize streams
        for (auto& stream : streams) {
            stream.synchronize();
        }
        
        // Update triangle optimization parameter if needed
        if (tril != -1) {
            tril = a;
        }
        
        // Compute TWED distances
        grid_evalZ<T>(
            &AA_dev[a * nA * dim], DA_dev.get(), nA, &TAA_dev[a * nA],
            BB_dev, DBB_dev.get(), nB, TBB_dev,
            nu, degree, lambda, dim,
            Res_dev_write.get(), nAA, nBB, tril
        );
        
        // Swap read/write buffers
        T* tmp_ptr = Res_dev_read.get();
        Res_dev_read = DeviceMemory<T>(std::move(Res_dev_write));
        Res_dev_write = DeviceMemory<T>(nBB);
        Res_dev_write.get() = tmp_ptr;
        
        // Copy results to host
        Res_dev_read.copyToHost(&RRes[a * nBB], nBB);
    }
    
    // Handle triangle optimization if needed
    if (tri == TriangleOpt::TRIU) {
        // Copy results to device for transpose
        DeviceMemory<T> matrix_dev(nAA * nBB);
        matrix_dev.copyFromHost(RRes, nAA * nBB);
        
        // Create cuBLAS handle
        CublasHandle handle;
        
        // Set up transpose parameters
        T alpha = 1.0;
        T beta = 0.0;
        
        // Perform transpose
        CUBLAS_CHECK(cublasSetStream(handle.get(), streams[0].get()));
        
        if constexpr (std::is_same<T, float>::value) {
            CUBLAS_CHECK(cublasSgeam(
                handle.get(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                nBB, nAA,
                &alpha,
                matrix_dev.get(), nAA,
                &beta,
                nullptr, nBB,
                results_dev.get(), nBB
            ));
        } else {
            CUBLAS_CHECK(cublasDgeam(
                handle.get(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                nBB, nAA,
                &alpha,
                matrix_dev.get(), nAA,
                &beta,
                nullptr, nBB,
                results_dev.get(), nBB
            ));
        }
        
        // Copy transposed results back to host
        results_dev.copyToHost(RRes, nAA * nBB);
    }
    
    return 0;
}

// Malloc_dev method
template<typename T>
void TWED<T>::malloc_dev(
    int nA, T** A_dev, T** TA_dev,
    int nB, T** B_dev, T** TB_dev,
    int dim, int nAA, int nBB
) {
    // Allocate device memory
    size_t size_A = nAA * nA * dim * sizeof(T);
    size_t size_TA = nAA * nA * sizeof(T);
    size_t size_B = nBB * nB * dim * sizeof(T);
    size_t size_TB = nBB * nB * sizeof(T);
    
    CUDA_CHECK(cudaMalloc(A_dev, size_A));
    CUDA_CHECK(cudaMalloc(TA_dev, size_TA));
    CUDA_CHECK(cudaMalloc(B_dev, size_B));
    CUDA_CHECK(cudaMalloc(TB_dev, size_TB));
}

// Free_dev method
template<typename T>
void TWED<T>::free_dev(
    T* A_dev, T* TA_dev,
    T* B_dev, T* TB_dev
) {
    // Free device memory
    CUDA_CHECK(cudaFree(A_dev));
    CUDA_CHECK(cudaFree(TA_dev));
    CUDA_CHECK(cudaFree(B_dev));
    CUDA_CHECK(cudaFree(TB_dev));
}

// Copy_to_dev method
template<typename T>
void TWED<T>::copy_to_dev(
    int nA, const T* A, T* A_dev, const T* TA, T* TA_dev,
    int nB, const T* B, T* B_dev, const T* TB, T* TB_dev,
    int dim, int nAA, int nBB
) {
    // Copy data to device
    size_t size_A = nAA * nA * dim * sizeof(T);
    size_t size_TA = nAA * nA * sizeof(T);
    size_t size_B = nBB * nB * dim * sizeof(T);
    size_t size_TB = nBB * nB * sizeof(T);
    
    CUDA_CHECK(cudaMemcpy(A_dev, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(TA_dev, TA, size_TA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_dev, B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(TB_dev, TB, size_TB, cudaMemcpyHostToDevice));
}

// Explicit template instantiations
template class TWED<float>;
template class TWED<double>;

// C-style interface implementations for backward compatibility
extern "C" {

// Double precision functions
double twed(double A[], int nA, double TA[],
            double B[], int nB, double TB[],
            double nu, double lambda, int degree, int dim) {
    try {
        return TWED<double>::compute(A, nA, TA, B, nB, TB, nu, lambda, degree, dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1.0;
    }
}

double twed_dev(double A_dev[], int nA, double TA_dev[],
                double B_dev[], int nB, double TB_dev[],
                double nu, double lambda, int degree, int dim) {
    try {
        return TWED<double>::compute_dev(A_dev, nA, TA_dev, B_dev, nB, TB_dev, nu, lambda, degree, dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1.0;
    }
}

int twed_batch(double AA[], int nA, double TAA[],
               double BB[], int nB, double TBB[],
               double nu, double lambda, int degree, int dim,
               int nAA, int nBB, double* RRes, int tri) {
    try {
        TriangleOpt tri_opt = static_cast<TriangleOpt>(tri);
        return TWED<double>::batch_compute(AA, nA, TAA, BB, nB, TBB, nu, lambda, degree, dim, nAA, nBB, RRes, tri_opt);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1;
    }
}

int twed_batch_dev(double AA_dev[], int nA, double TAA_dev[],
                   double BB_dev[], int nB, double TBB_dev[],
                   double nu, double lambda, int degree, int dim,
                   int nAA, int nBB, double* RRes, int tri) {
    try {
        TriangleOpt tri_opt = static_cast<TriangleOpt>(tri);
        return TWED<double>::batch_compute_dev(AA_dev, nA, TAA_dev, BB_dev, nB, TBB_dev, nu, lambda, degree, dim, nAA, nBB, RRes, tri_opt);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1;
    }
}

void twed_malloc_dev(int nA, double **A_dev, double **TA_dev,
                     int nB, double **B_dev, double **TB_dev,
                     int dim, int nAA, int nBB) {
    try {
        TWED<double>::malloc_dev(nA, A_dev, TA_dev, nB, B_dev, TB_dev, dim, nAA, nBB);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

void twed_free_dev(double *A_dev, double *TA_dev,
                   double *B_dev, double *TB_dev) {
    try {
        TWED<double>::free_dev(A_dev, TA_dev, B_dev, TB_dev);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

void twed_copy_to_dev(int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                      int nB, double B[], double B_dev[], double TB[], double TB_dev[],
                      int dim, int nAA, int nBB) {
    try {
        TWED<double>::copy_to_dev(nA, A, A_dev, TA, TA_dev, nB, B, B_dev, TB, TB_dev, dim, nAA, nBB);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

// Single precision functions
float twedf(float A[], int nA, float TA[],
            float B[], int nB, float TB[],
            float nu, float lambda, int degree, int dim) {
    try {
        return TWED<float>::compute(A, nA, TA, B, nB, TB, nu, lambda, degree, dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1.0f;
    }
}

float twed_devf(float A_dev[], int nA, float TA_dev[],
                float B_dev[], int nB, float TB_dev[],
                float nu, float lambda, int degree, int dim) {
    try {
        return TWED<float>::compute_dev(A_dev, nA, TA_dev, B_dev, nB, TB_dev, nu, lambda, degree, dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1.0f;
    }
}

int twed_batchf(float AA[], int nA, float TAA[],
                float BB[], int nB, float TBB[],
                float nu, float lambda, int degree, int dim,
                int nAA, int nBB, float* RRes, int tri) {
    try {
        TriangleOpt tri_opt = static_cast<TriangleOpt>(tri);
        return TWED<float>::batch_compute(AA, nA, TAA, BB, nB, TBB, nu, lambda, degree, dim, nAA, nBB, RRes, tri_opt);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1;
    }
}

int twed_batch_devf(float AA_dev[], int nA, float TAA_dev[],
                    float BB_dev[], int nB, float TBB_dev[],
                    float nu, float lambda, int degree, int dim,
                    int nAA, int nBB, float* RRes, int tri) {
    try {
        TriangleOpt tri_opt = static_cast<TriangleOpt>(tri);
        return TWED<float>::batch_compute_dev(AA_dev, nA, TAA_dev, BB_dev, nB, TBB_dev, nu, lambda, degree, dim, nAA, nBB, RRes, tri_opt);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
        return -1;
    }
}

void twed_malloc_devf(int nA, float **A_dev, float **TA_dev,
                      int nB, float **B_dev, float **TB_dev,
                      int dim, int nAA, int nBB) {
    try {
        TWED<float>::malloc_dev(nA, A_dev, TA_dev, nB, B_dev, TB_dev, dim, nAA, nBB);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

void twed_free_devf(float *A_dev, float *TA_dev,
                    float *B_dev, float *TB_dev) {
    try {
        TWED<float>::free_dev(A_dev, TA_dev, B_dev, TB_dev);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

void twed_copy_to_devf(int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                       int nB, float B[], float B_dev[], float TB[], float TB_dev[],
                       int dim, int nAA, int nBB) {
    try {
        TWED<float>::copy_to_dev(nA, A, A_dev, TA, TA_dev, nB, B, B_dev, TB, TB_dev, dim, nAA, nBB);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuTWED error: %s\n", e.what());
    }
}

} // extern "C"

} // namespace cutwed