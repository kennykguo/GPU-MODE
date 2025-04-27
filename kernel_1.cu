#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>


#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, float *A, const float *B,
                            float beta, float *C){
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the thread is in bounds to begin with, for our matrix

  if (x < M && y < N){
    float tmp = 0.0;

    for (int i = 0; i< K; ++i){
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void init_matrix(float *mat, int rows, int cols){
  for (int i = 0; i < rows * cols; ++i){
    mat[i] = (float)rand() / RAND_MAX;
  }
}


int main(){
  int M = 1024;
  int N = 1024;
  int K = 1024;



  printf("Benchmarking SGEMM");
  float *h_A = (float*)malloc(M * K * sizeof(float));
  float *h_B = (float*)malloc(K * N * sizeof(float));
  float *h_C = (float*)malloc(M * N * sizeof(float));
  float *h_C_ref = (float *)malloc(M * N * sizeof(float));
  srand(42);

  init_matrix(h_A, M, K);
  init_matrix(h_B, M, N);
  init_matrix(h_C, M, K);
}




// create as many blocks to map to all of the matrix
dim3 gridDim(CEIL_DIV(M,32), CEIL_DIV(N,32), 1);


// 32 x 32 = 1024 threads per block

dim3 blockDim(32, 32, 1);

// launch the asynchronous execution of the kernel on the device
// the function call returns immediately on the host
