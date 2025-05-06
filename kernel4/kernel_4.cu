#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>


#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_smem(int M, int N, int K, float alpha, float *A, const float *B, float beta, float *C){
    // moving onto 1d arrays
    // A: 1D array with indexing A[row * K + col]
    // B: 1D array with indexing B[row * N + col]
    // C: 1D array with indexing C[row * N + col]
    // Grid divides C into blocks of size BLOCKSIZE × BLOCKSIZE (2D)
    // Block dimensions should match BLOCKSIZE for efficient thread mapping (32, 32)

    // TILE DIMENSIONS
    const int BLOCKSIZE = 32;
    const int TM = 8;  // elements in tile computed per thread in M dimension
    const int TN = 8;  // elements in tile computed per thread in N dimension

    // row and col index in the context of each tile
    const int threadrows = BLOCKSIZE / TM;  // threads in y-direction
    const int threadcols = BLOCKSIZE / TN;  // threads in x-direction

    // work through K one blocktile at a time
    // need to write it like this since TM and TN are not always divisors of BLOCKSIZE

    // updated size of block size in M and N dimensions, to ensure that we have a clean multiple of TM and TN respectively
    const int BM = threadrows * TM;  // block size (simplified)
    // technically is block size in M dim
    const int BN = threadcols * TN;  // block size (simplified)
    // block size in N dimension
    const int BK = 16;  // tiling size in K dimension

    // thread to block (vars local to block)
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    // map thread to output in C
    // block in vertical dir
    // sub-block in vertical block
    const int rowOffset = blockIdx.y * BM + threadRow * TM;
    // rowOffset is global. BM is just block_size don't forget. the col dimension is
    // sub-block is indexed this way, because we are calculating TM entries, per thread
    const int colOffset = blockIdx.x * BN + threadCol * TN;

    // focusing on C matrix, segment each block into BM x BN blocks. need to have a valid thread checking mechanism
    // out of bounds of c matrix
    // if (row >= M || col >= N){
    // return;
    // }

    // caching the current BM x BN block
    // each block of threads gets an equal share of shared memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];


    // per-thread accumulation
    float threadResults[TM][TN] = {0.0};

    // block index increments here by bk. this loop computes the dot product per thread
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[threadRow % BM][threadCol %BK] = A[(bkIdx + threadRow) * K + threadCol];
        Bs[threadRow % BK][threadCol % BN] = B[(bkIdx + threadRow) * N + blockIdx.x * BLOCKSIZE + threadCol];
    
        // wait for all threads to load their data into shared memory
        __syncthreads();

        // compute dot product for this tile (each thread computes)
        // loops over BK (dot product)
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float b_temp = Bs[dotIdx][threadCol % BN];
            for (uint resIdx = 0; resIdx < TM; ++resIdx){
                threadResults[resIdx][threadCol % TN] += As[threadRow % BM][resIdx] * b_temp;
            }
            
        }

    // wait for all threads to finish using the shared memory before loading next tile
    __syncthreads();
    }

    // write result to global memory
    // thread level -> all threads literally have computed their dot product for that entry
    // C = alpha * (A × B) + beta * C
    C[(rowOffset + TM) * N + (colOffset + TN)] = threadResults[threadRow][threadCol];
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
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32, 32);
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