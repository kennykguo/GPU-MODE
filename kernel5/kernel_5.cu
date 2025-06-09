#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


// kernel optimizations: reduce warp divergence, unroll loops for ilp, 
// check if thread count needs increase due to arithmetic intensity saturation
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  // === BLOCK TILE DIMENSIONS ===
  // each block processes a 128x128 tile of the output matrix
  const int block_tile_rows = 128;     // BM - rows per block tile
  const int block_tile_cols = 128;     // BN - columns per block tile  
  const int block_tile_k_dim = 8;      // BK - k-dimension per block tile iteration
  
  // === THREAD TILE DIMENSIONS ===
  // each thread processes an 8x8 sub-tile within the block tile
  const int thread_tile_rows = 8;      // TM - rows per thread
  const int thread_tile_cols = 8;      // TN - columns per thread

  // === BLOCK POSITION IN GRID ===
  // which 128x128 block this thread block is responsible for
  const int block_row_idx = blockIdx.y;  // which block row in grid (was cRow)
  const int block_col_idx = blockIdx.x;  // which block column in grid (was cCol)

  // === THREAD POSITION WITHIN BLOCK ===
  // 256 threads arranged as 16x16 grid, each handling 8x8 output elements
  // total threads = (block_tile_rows * block_tile_cols) / (thread_tile_rows * thread_tile_cols)
  //               = (128 * 128) / (8 * 8) = 256 threads per block
  const int threads_per_col = block_tile_cols / thread_tile_cols; // 128/8 = 16
  const int thread_col_idx = threadIdx.x % threads_per_col;       // 0-15: which thread column
  const int thread_row_idx = threadIdx.x / threads_per_col;       // 0-15: which thread row

  // === SHARED MEMORY ALLOCATION ===
  // shared memory for current block tiles from A and B matrices
  __shared__ float shared_a_tile[block_tile_rows * block_tile_k_dim]; // 128 x 8 = 1024 elements
  __shared__ float shared_b_tile[block_tile_k_dim * block_tile_cols]; // 8 x 128 = 1024 elements

  // === ADVANCE GLOBAL MEMORY POINTERS ===
  // move pointers to start of current block's data
  A += block_row_idx * block_tile_rows * K;                    // advance to block row start
  B += block_col_idx * block_tile_cols;                        // advance to block column start  
  C += block_row_idx * block_tile_rows * N + block_col_idx * block_tile_cols; // advance to output block

  // === SHARED MEMORY LOADING INDICES ===
  // calculate which elements each thread loads into shared memory
  
  // A matrix loading (128x8 elements, 256 threads, 4 elements per thread)
  const int a_load_row = threadIdx.x / block_tile_k_dim;        // which row in A tile (0-31)
  const int a_load_col = threadIdx.x % block_tile_k_dim;        // which col in A tile (0-7)
  const int a_load_stride = blockDim.x / block_tile_k_dim;      // 256/8 = 32 rows per iteration
  
  // B matrix loading (8x128 elements, 256 threads, 4 elements per thread)  
  const int b_load_row = threadIdx.x / block_tile_cols;         // which row in B tile (0-1)
  const int b_load_col = threadIdx.x % block_tile_cols;         // which col in B tile (0-127)
  const int b_load_stride = blockDim.x / block_tile_cols;       // 256/128 = 2 rows per iteration

  // === THREAD-LOCAL STORAGE ===
  // each thread accumulates results for its 8x8 output sub-tile
  float thread_results[thread_tile_rows * thread_tile_cols] = {0.0}; // 64 results per thread
  float reg_a_values[thread_tile_rows] = {0.0};               // cache for A values (8 elements)
  float reg_b_values[thread_tile_cols] = {0.0};               // cache for B values (8 elements)

  // === MAIN K-DIMENSION LOOP ===
  // iterate through k-dimension in chunks of block_tile_k_dim (8)
  for (int k_block_start = 0; k_block_start < K; k_block_start += block_tile_k_dim) {
    
    // === LOAD A BLOCK TILE INTO SHARED MEMORY ===
    // each thread loads 4 elements: block_tile_rows/a_load_stride = 128/32 = 4
    // loading pattern: 8-thread groups load consecutive columns of 32-row-spaced rows
    for (int load_offset = 0; load_offset < block_tile_rows; load_offset += a_load_stride) {
      shared_a_tile[(a_load_row + load_offset) * block_tile_k_dim + a_load_col] = 
          A[(a_load_row + load_offset) * K + a_load_col];
    }
    // threads 0-7: load cols 0-7 of rows {0,32,64,96}
    // threads 8-15: load cols 0-7 of rows {1,33,65,97}
    // pattern continues for all 256 threads, achieving coalesced access

    // === LOAD B BLOCK TILE INTO SHARED MEMORY ===
    // each thread loads 4 elements: block_tile_k_dim/b_load_stride = 8/2 = 4  
    // loading pattern: 128-thread groups load consecutive columns of 2-row-spaced rows
    for (int load_offset = 0; load_offset < block_tile_k_dim; load_offset += b_load_stride) {
      shared_b_tile[(b_load_row + load_offset) * block_tile_cols + b_load_col] = 
          B[(b_load_row + load_offset) * N + b_load_col];
    }
    // threads 0-127: load cols 0-127 of rows {0,2,4,6}
    // threads 128-255: load cols 0-127 of rows {1,3,5,7}
    // achieves coalesced access with 128 consecutive threads per row
    
    __syncthreads(); // ensure all threads finish loading before computation

    // === ADVANCE TO NEXT BLOCK TILES ===
    A += block_tile_k_dim;           // move 8 columns right in A
    B += block_tile_k_dim * N;       // move 8 rows down in B

    // === COMPUTE DOT PRODUCTS FOR THIS BLOCK TILE ===
    // iterate through k-dimension of current block tile (8 iterations)
    for (int dot_product_idx = 0; dot_product_idx < block_tile_k_dim; ++dot_product_idx) {
      
      // === LOAD A VALUES INTO REGISTERS (VERTICAL SLICE) ===
      // load 8 consecutive A elements from same column, different rows
      for (int i = 0; i < thread_tile_rows; ++i) {
        reg_a_values[i] = shared_a_tile[(thread_row_idx * thread_tile_rows + i) * block_tile_k_dim + dot_product_idx];
      }
      // thread 0: loads A elements from rows 0-7, column dot_product_idx
      // thread 16: loads A elements from rows 8-15, column dot_product_idx
      
      // === LOAD B VALUES INTO REGISTERS (HORIZONTAL SLICE) ===
      // load 8 consecutive B elements from same row, different columns
      for (int i = 0; i < thread_tile_cols; ++i) {
        reg_b_values[i] = shared_b_tile[dot_product_idx * block_tile_cols + thread_col_idx * thread_tile_cols + i];
      }
      // thread 0: loads B elements from row dot_product_idx, columns 0-7
      // thread 1: loads B elements from row dot_product_idx, columns 8-15
      
      // === COMPUTE OUTER PRODUCT AND ACCUMULATE ===
      // compute 8x8 outer product of reg_a_values and reg_b_values
      // each iteration adds one outer product to accumulated results
      for (int result_row = 0; result_row < thread_tile_rows; ++result_row) {
        for (int result_col = 0; result_col < thread_tile_cols; ++result_col) {
          thread_results[result_row * thread_tile_cols + result_col] += 
              reg_a_values[result_row] * reg_b_values[result_col];
        }
      }
      // produces 64 multiply-accumulate operations per thread per dot_product_idx
      // total: 64 ops × 8 iterations × 256 threads = 131,072 ops per block tile
    }
    
    __syncthreads(); // ensure all threads finish computation before next iteration
  }

  // === WRITE RESULTS TO GLOBAL MEMORY ===
  // each thread writes its 8x8 results to corresponding positions in output matrix
  for (int result_row = 0; result_row < thread_tile_rows; ++result_row) {
    for (int result_col = 0; result_col < thread_tile_cols; ++result_col) {
      // calculate global position: block position + thread position + element position
      int global_row = thread_row_idx * thread_tile_rows + result_row;
      int global_col = thread_col_idx * thread_tile_cols + result_col;
      
      C[global_row * N + global_col] = 
          alpha * thread_results[result_row * thread_tile_cols + result_col] +
          beta * C[global_row * N + global_col];
    }
  }
  // thread 0: writes to rows 0-7, columns 0-7 of current block
  // thread 1: writes to rows 0-7, columns 8-15 of current block
  // pattern continues across all 256 threads in block
}

// === CORRECTED KERNEL LAUNCH CONFIGURATION ===
// grid dimensions: divide matrix by block tile size
// dim3 grid_dim((N + 127) / 128, (M + 127) / 128, 1);  // use block_tile_cols=128, block_tile_rows=128
// dim3 block_dim(256, 1, 1);                            // 256 threads per block


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
  dim3 gridDim((N + 127) / 128, (M + 127) / 128, 1);  // use BN=128, BM=128
  dim3 blockDim(256, 1, 1);                            // 256 threads per block
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