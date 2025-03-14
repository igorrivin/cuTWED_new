#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cutwed {

/**
 * @brief Exception class for CUDA errors
 */
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& message, cudaError_t code = cudaSuccess) 
        : std::runtime_error(message + ": " + cudaGetErrorString(code)), error_code(code) {}
    
    cudaError_t error_code;
};

/**
 * @brief Exception class for cuBLAS errors
 */
class CublasError : public std::runtime_error {
public:
    explicit CublasError(const std::string& message, cublasStatus_t code = CUBLAS_STATUS_SUCCESS) 
        : std::runtime_error(message + ": " + getCublasErrorString(code)), error_code(code) {}
    
    cublasStatus_t error_code;
    
    static const char* getCublasErrorString(cublasStatus_t error) {
        switch (error) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "Unknown cuBLAS error";
        }
    }
};

/**
 * @brief Class for CUDA stream management with RAII
 */
class CudaStream {
public:
    CudaStream() {
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            throw CudaError("Failed to create CUDA stream", error);
        }
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Get the stream
    cudaStream_t get() const { return stream_; }
    
    // Synchronize the stream
    void synchronize() const {
        cudaError_t error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) {
            throw CudaError("Failed to synchronize CUDA stream", error);
        }
    }
    
    // Disable copying
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
private:
    cudaStream_t stream_;
};

/**
 * @brief Class for cuBLAS handle management with RAII
 */
class CublasHandle {
public:
    CublasHandle() {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw CublasError("Failed to create cuBLAS handle", status);
        }
    }
    
    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }
    
    // Get the handle
    cublasHandle_t get() const { return handle_; }
    
    // Set the stream
    void setStream(cudaStream_t stream) {
        cublasStatus_t status = cublasSetStream(handle_, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw CublasError("Failed to set cuBLAS stream", status);
        }
    }
    
    // Disable copying
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    
private:
    cublasHandle_t handle_;
};

/**
 * @brief Template class for CUDA device memory management with RAII
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : data_(nullptr), size_(0) {}
    
    explicit DeviceMemory(size_t size) : size_(size) {
        if (size_ > 0) {
            cudaError_t error = cudaMalloc(&data_, size_ * sizeof(T));
            if (error != cudaSuccess) {
                throw CudaError("Failed to allocate device memory", error);
            }
        }
    }
    
    ~DeviceMemory() {
        free();
    }
    
    // Allocate memory
    void allocate(size_t size) {
        free();
        size_ = size;
        if (size_ > 0) {
            cudaError_t error = cudaMalloc(&data_, size_ * sizeof(T));
            if (error != cudaSuccess) {
                throw CudaError("Failed to allocate device memory", error);
            }
        }
    }
    
    // Free memory
    void free() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }
    
    // Copy from host to device
    void copyFromHost(const T* host_data, size_t count, cudaStream_t stream = 0) {
        if (count > size_) {
            throw std::runtime_error("Copy count exceeds allocated size");
        }
        cudaError_t error = cudaMemcpyAsync(data_, host_data, count * sizeof(T), 
                                           cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            throw CudaError("Failed to copy data from host to device", error);
        }
    }
    
    // Copy from device to host
    void copyToHost(T* host_data, size_t count, cudaStream_t stream = 0) const {
        if (count > size_) {
            throw std::runtime_error("Copy count exceeds allocated size");
        }
        cudaError_t error = cudaMemcpyAsync(host_data, data_, count * sizeof(T), 
                                           cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) {
            throw CudaError("Failed to copy data from device to host", error);
        }
    }
    
    // Get the device pointer
    T* get() const { return data_; }
    
    // Get the size
    size_t size() const { return size_; }
    
    // Check if memory is allocated
    bool isAllocated() const { return data_ != nullptr; }
    
    // Disable copying
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Allow moving
    DeviceMemory(DeviceMemory&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
private:
    T* data_;
    size_t size_;
};

// Helper function to check CUDA errors
inline void checkCudaErrors(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaError(std::string(file) + ":" + std::to_string(line), error);
    }
}

// Helper function to check cuBLAS errors
inline void checkCublasErrors(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CublasError(std::string(file) + ":" + std::to_string(line), status);
    }
}

// Macro to simplify error checking
#define CUDA_CHECK(call) checkCudaErrors(call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkCublasErrors(call, __FILE__, __LINE__)

} // namespace cutwed

#endif // CUDA_UTILS_HPP