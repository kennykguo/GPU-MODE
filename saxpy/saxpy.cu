#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** failed - aborting\n"); \
            exit(1); \
        } \
    } while (0)

// basic sequential saxpy for reference
void saxpy_cpu(int n, float a, const float* x, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// method 1: cublas saxpy
void saxpy_cublas(int n, float a, const float* x, float* y) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    
    // copy data to device
    cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
    cublasSetVector(n, sizeof(float), y, 1, d_y, 1);
    
    // perform saxpy on device
    cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);
    
    // copy result back to host
    cublasGetVector(n, sizeof(float), d_y, 1, y, 1);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}

// method 2: openacc saxpy
#ifdef _OPENACC
void saxpy_openacc(int n, float a, const float* restrict x, float* restrict y) {
    #pragma acc kernels copyin(x[0:n]) copy(y[0:n])
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
#endif

// method 3: cuda c++ kernel
__global__ void saxpy_kernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_cuda(int n, float a, const float* x, float* y) {
    float *d_x, *d_y;
    size_t size = n * sizeof(float);
    
    // allocate device memory
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    
    // copy data to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    // launch kernel with 256 threads per block
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);
    
    cudaCheckErrors("kernel launch");
    
    // copy result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(d_x);
    cudaFree(d_y);
}

// method 4: thrust saxpy using transform
void saxpy_thrust(int n, float a, const float* x, float* y) {
    // create host vectors from raw pointers
    thrust::host_vector<float> h_x(x, x + n);
    thrust::host_vector<float> h_y(y, y + n);
    
    // copy to device
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    
    // perform saxpy using transform with lambda
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                     [a] __device__ (float x_val, float y_val) {
                         return a * x_val + y_val;
                     });
    
    // copy result back to host
    h_y = d_y;
    
    // copy back to output array
    thrust::copy(h_y.begin(), h_y.end(), y);
}

// timing utility
template<typename Func>
double time_function(Func func, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    std::cout << name << ": " << time_ms << " ms" << std::endl;
    return time_ms;
}

// verification function
bool verify_result(const float* result, const float* reference, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(result[i] - reference[i]) > tolerance) {
            std::cout << "mismatch at index " << i << ": " 
                      << result[i] << " vs " << reference[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 1 << 20;  // 1m elements
    const float a = 2.0f;
    
    // allocate host memory
    std::vector<float> x(n), y_ref(n);
    
    // initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
        y_ref[i] = dis(gen);
    }
    
    // create reference result
    std::vector<float> y_cpu = y_ref;
    saxpy_cpu(n, a, x.data(), y_cpu.data());
    
    std::cout << "testing saxpy with " << n << " elements" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // test cublas
    {
        std::vector<float> y_test = y_ref;
        time_function([&]() { saxpy_cublas(n, a, x.data(), y_test.data()); }, "cublas");
        std::cout << "cublas correct: " << verify_result(y_test.data(), y_cpu.data(), n) << std::endl;
    }
    
    // test cuda c++
    {
        std::vector<float> y_test = y_ref;
        time_function([&]() { saxpy_cuda(n, a, x.data(), y_test.data()); }, "cuda c++");
        std::cout << "cuda c++ correct: " << verify_result(y_test.data(), y_cpu.data(), n) << std::endl;
    }
    
    // test thrust
    {
        std::vector<float> y_test = y_ref;
        time_function([&]() { saxpy_thrust(n, a, x.data(), y_test.data()); }, "thrust");
        std::cout << "thrust correct: " << verify_result(y_test.data(), y_cpu.data(), n) << std::endl;
    }
    
    #ifdef _OPENACC
    // test openacc
    {
        std::vector<float> y_test = y_ref;
        time_function([&]() { saxpy_openacc(n, a, x.data(), y_test.data()); }, "openacc");
        std::cout << "openacc correct: " << verify_result(y_test.data(), y_cpu.data(), n) << std::endl;
    }
    #endif
    
    return 0;
}

/*
compilation commands:

# basic cuda version
nvcc -o saxpy_cuda saxpy.cu -lcublas

# with thrust (included in cuda)
nvcc -o saxpy_thrust saxpy.cu -lcublas --std=c++14

# with openacc (using pgi/nvidia hpc sdk)
nvc++ -acc -ta=tesla -Minfo=accel -o saxpy_openacc saxpy.cpp -lcublas

# full version with all features
nvcc -o saxpy_full saxpy.cu -lcublas --std=c++14 -O3
*/