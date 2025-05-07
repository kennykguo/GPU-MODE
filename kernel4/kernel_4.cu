#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>


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
  // 
  const int BLOCKSIZE = 32; // actual thread block is 4 x 4 threads!
  const int TM = 8;  // elements in tile computed per thread in m dimension
  const int TN = 8;  // elements in tile computed per thread in n dimension

  // thread organization within the block
  const int threadrows = BLOCKSIZE / TM;  // 4 - threads in y-direction (4)
  const int threadcols = BLOCKSIZE / TN;  // 4 - threads in x-direction (4)

  // actual block dimensions used for computation
  const int BM = threadrows * TM;  // block size in m dimension (32)
  const int BN = threadcols * TN;  // block size in n dimension (32)
  const int BK = 16;               // tiling size in k dimension

  // thread identifiers within the block
  const int threadRow = threadIdx.y;
  const int threadCol = threadIdx.x; //
  
  // local thread id (linearized)
  const int tid = threadRow * blockDim.x + threadCol;

  // starting point for this thread's computation in the global matrix C
  // row + sub-row
  // col + sub-col
  const int rowOffset = blockIdx.y * BM + (tid / threadcols) * TM;
  const int colOffset = blockIdx.x * BN + (tid % threadcols) * TN;

  // allocate shared memory for tiles
  __shared__ float As[BM][BK]; // 32 x 32
  __shared__ float Bs[BK][BN]; // 32 x 32

  // per-thread accumulation registers
  float threadResults[TM][TN] = {0.0f}; // 8 x 8

  // iterate through k dimension in blocks of size bk
  // REMEMBER: each thread must calculate its corresponding 8x8 results in the C matrix
  // looping through A, and B in the K dim, in increments of BK
  // bkIdx is the corresponding global gridded index for for col in A, aswell as the row in B
  // by adding internal col, we get the actual column to load, and vice versa for row
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // load As
    // starting from the global thread index - REMEMBER 4 x 4 block of threads

    // caching 32 x 16

    // loading into As, from A
    // loading into BM X BK from left to right
    // if BK != threadrows * threadcols, some threads need to write more than once to smem, or be idle
    // total threads running per block - threadrows * threadcols)
    // loop uses tid as a way to create internal var row and col, to populate shared memory
    for (int i = tid; i < BM * BK; i += threadrows * threadcols) {
        // left to right allows coalescing
        // internal values for As
        int row = i / BK; // O to BM  -1
        int col = i % BK; // 0 to BK - 1

        // bounds check (BM * BK not always a multiple of threadrows * threadcols)
        if (blockIdx.y * BM + row < M && bkIdx + col < K) {
            // block row index multiplifed by block size in M dim + sub- row
            As[row][col] = A[(blockIdx.y * BM + row) * K + (bkIdx + col)];
        } 
        // else {
        //     As[row][col] = 0.0f;
        // }
    }
    
    for (int i = tid; i < BK * BN; i += threadrows * threadcols) {

        int row = i / BN;
        int col = i % BN;
        
        // left to right allows coalescing
        if (bkIdx + row < K && blockIdx.x * BN + col < N) {
            Bs[row][col] = B[(bkIdx + row) * N + (blockIdx.x * BN + col)];
        } 
        // else {
        //     Bs[row][col] = 0.0f;
        // }
    }
    
    // wait for all threads to finish loading into shared memory
    __syncthreads();

    // compute the dot product for this thread's assigned elements (TM X TN) -> 8 x 8
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // each thread handles a tm×tn block
        for (int i = 0; i < TM; ++i) {
            int arow = (tid / threadcols) * TM + i;
            // bounds check -> incase arow is not a multiple of BM
            if (arow < BM) {
              float aval = As[arow][dotIdx]; // left to right
              for (int j = 0; j < TN; ++j) {
                int bcol = (tid % threadcols) * TN + j;
                if (bcol < BN) { // another boundary check
                    threadResults[i][j] += aval * Bs[dotIdx][bcol];
                }
              }
            }
        }
    }
      
    // wait for all threads to finish using the shared memory before loading next tile
    __syncthreads();
  }

  // write results back to global memory with alpha and beta scaling
  for (int i = 0; i < TM; ++i) {
      int globalRow = rowOffset + i;

      if (globalRow < M) {
          for (int j = 0; j < TN; ++j) {
              int globalCol = colOffset + j;
              
              if (globalCol < N) {
                  int globalIdx = globalRow * N + globalCol;
                  C[globalIdx] = alpha * threadResults[i][j] + beta * C[globalIdx];
              }
          }
      }
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
  dim3 gridDim((N + 31) / 32, (M + 31) / 32, 1); 
  dim3 blockDim(4, 4);
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