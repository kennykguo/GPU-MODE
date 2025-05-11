#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


// REMEMBER THIS KERNEL IS INITIALIZED AS:
// dim3 gridDim((N + 31) / 32, (M + 31) / 32); // Ceiling division to cover entire matrix
// dim3 blockDim(4, 4); // 4×4 threads per block, NOT 32×32

// launched like dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
// kernel can be made faster through reducing warp divergence. unrolling loops for ILP, checking if thread count needs to be higher due to saturation of arithmetic intensity
__global__ void sgemm_smem(int M, int N, int K, float alpha, float *A, const float *B, float beta, float *C){
  // moving onto 1d arrays
  // a: 1d array with indexing a[row * k + col]
  // b: 1d array with indexing b[row * n + col]
  // c: 1d array with indexing c[row * n + col]
  // grid divides c into blocks of size blocksize × blocksize (2d)
  // block dimensions should match blocksize for efficient thread mapping (32, 32)

  // tile dimensions
  const int BM = 64; 
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8; // TM = TN

  const uint block_row_idx = blockIdx.y;
  const uint block_col_idx = blockIdx.x;

  const int total_results = BM * BN; // results per block to calculate
  const uint threads_per_block = total_results / TM; // # of threads per block
  assert(threads_per_block == blockDim.x); // total # of threads should be consistent

  const int thread_row = threadIdx.x / BN;
  const int thread_col = threadIdx.x % BN;

  __shared__ float As[BM * BK]; // floats to store mat mul answers
  __shared__ float Bs[BK * BN]; // block matrix sizes

  A += block_row_idx * BM * K; // Move block's threads to corresponding tiles of memory
  B += block_col_idx * BN;
  C += (block_row_idx * BM) * N + (block_col_idx * BN);
  
  // for loading into smem
  const int inner_row_A = threadIdx.x / BK;
  const int inner_col_A = threadIdx.x % BK;
  const int inner_row_B = threadIdx.x / BN;
  const int inner_col_B = threadIdx.x % BN;

  // registers
  float tmp_results[TM] = {0.0f};


  for (uint block_idx = 0; block_idx < K; block_idx += BK){
    // load into smem first
    // in every block, there are 64 x 8 = 512 threads
    // each thread loads exactly one element, in a block
    // hence assert(threads_per_block == blockDim.x);
    // inner_row_A ranges from 0 to 63 (64 rows)
    // inner_col_A ranges from 0 to 7 (8 columns)
    // inner_row_B ranges from 0 to 7 (8 rows)
    // inner_col_B ranges from 0 to 63 (64 columns)
    As[inner_row_A * BK + inner_col_A] = A[inner_row_A * K + inner_col_A];
    Bs[inner_row_B * BN + inner_col_B] = B[inner_row_B * N + inner_col_B];
    __syncthreads();
    
    A += BK;
    B += BK * N;

    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx){ //   
      const float B_tmp = Bs[(dot_idx) * BN + inner_col_B]; // move to a register
      for (uint res_idx = 0; res_idx < TM; ++res_idx){ //loop over entire tile

        // (thread_row * TM + res_idx) iterates row by row. dot_idx we iterate across BK correctly
        // in As, calculate down, with B_tmp
        tmp_results[res_idx] += As[(thread_row * TM + res_idx) * BK + dot_idx] * B_tmp;
      }
      // ensure all computations done before next tile
    }
    __syncthreads();
  }

  // store into C
  // each thread in each block calculated a column of 8 elements
  for (uint res_idx = 0; res_idx < TM; ++res_idx){ //loop over entire tile
    C[(thread_row * TM + res_idx)* N + thread_col]  = alpha * tmp_results[res_idx] + beta * C[(thread_row * TM + res_idx)* N + thread_col];
  }
}   


void sgemm_cpu(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  // Compute C = alpha * A * B + beta * C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}



// Function to verify the results between CPU and GPU
int verify_results(float *cpu_C, float *gpu_C, int M, int N) {
  const float epsilon = 1e-2; // Tolerance for floating point comparison
  int errors = 0;
  
  for (int i = 0; i < M * N; i++) {
    float diff = fabsf(cpu_C[i] - gpu_C[i]);
    if (diff > epsilon) {
      errors++;
      if (errors <= 10) { // Print only the first 10 errors
        printf("Error at index %d: GPU = %f, CPU = %f (diff = %f)\n", 
               i, gpu_C[i], cpu_C[i], diff);
      }
    }
  }
  
  if (errors > 0) {
    printf("Verification FAILED: %d errors found out of %d elements\n", errors, M * N);
    return 0;
  } else {
    printf("Verification PASSED: All values match within epsilon %e\n", epsilon);
    return 1;
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
  // initialize matrix dimensions
  int M = 1024;
  int N = 1024;
  int K = 1024;


  printf("benchmarking SGEMM\n");

  // allocate host device pointers
  float *h_A = (float*)malloc(M * K * sizeof(float));
  float *h_B = (float*)malloc(K * N * sizeof(float));
  float *h_C = (float*)malloc(M * N * sizeof(float));
  float *h_C_ref = (float *)malloc(M * N * sizeof(float));
  
  srand(42); // initialize matrices
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);
  init_matrix(h_C, M, N);

  memcpy(h_C_ref, h_C, M * N * sizeof(float)); // make a copy to the reference pointer
  float *d_A, *d_B, *d_C;
  // allocate to the device
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
  // copy to the device from host
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  
  float alpha = 1.1f;
  float beta = 1.2f;

  // initialize timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // grid and block dimensions - 2d and 2d
  dim3 gridDim((N + 64 - 1) / 64, (M + 64 - 1) / 64, 1);
  dim3 blockDim(512, 1, 1);
  const int num_iterations = 10;
  
  // benchmark loop  
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i){
    // Reset C to original values before each iteration
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    sgemm_smem<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  // calculate the statistics and print the statistics
  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  float avg_time_ms = milliseconds / num_iterations;
  double gflops = (2.0 * M * N * K) / (avg_time_ms * 1e6);
  printf("average kernel execution time: %.3f ms\n", avg_time_ms);
  printf("performance: %.2f GFLOPS\n", gflops);

  CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  
  // verify results
  sgemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_C_ref);
  verify_results(h_C_ref, h_C, M, N);
  
  // clean up variables
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
}