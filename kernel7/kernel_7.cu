#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// this kernel builds on the vectorized kernel by adding bank conflict resolution
// for shared memory B matrix access while maintaining all vectorization benefits
// coordinate transformation that maps natural storage coordinates
// to a "linearized" layout that eliminates bank conflicts during computation
// - shared memory has 32 banks, each 4 bytes wide
// - bank = (address / 4) % 32
// - conflicts occur when multiple threads access same bank, different addresses
// - conflicts can reduce bandwidth by up to 32x
// - change both storage pattern AND access pattern via coordinate transformation
// - ensure consecutive thread_col_idx values access consecutive addresses
// - consecutive addresses map to consecutive banks → zero conflicts
__global__ void sgemm_resolve_bank_conflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  // === BLOCK TILE DIMENSIONS ===
  // BM=128, BN=128, BK=8: each block processes 128x128 output tile
  // TM=8, TN=8: each thread processes 8x8 sub-tile
  // total threads per block: (BM*BN)/(TM*TN) = (128*128)/(8*8) = 256

  const int BM = 128; // BM - rows per block tile
  const int BN = 128; // BN - columns per block tile
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;
  
  // THREAD TILE DIMENSIONS
  // each thread processes an 8x8 sub-tile within the block tile
  const int thread_tile_rows = 8; // TM - rows per thread
  const int thread_tile_cols = 8; // TN - columns per thread

  // === BLOCK POSITION IN GRID ===
  const int block_row_idx = blockIdx.y;  // which 128x128 block row in grid
  const int block_col_idx = blockIdx.x;  // which 128x128 block column in grid

  // === THREAD POSITION WITHIN BLOCK ===
  // 256 threads arranged as 16x16 logical grid
  const int threads_per_col = BN / TN; // 128/8 = 16 threads span column
  const int thread_col_idx = threadIdx.x % threads_per_col; // 0-15: which thread column
  const int thread_row_idx = threadIdx.x / threads_per_col; // 0-15: which thread row

  // === SHARED MEMORY ALLOCATION ===
  // As: transposed layout from previous kernel (enables vectorized access)
  // Bs: new "linearized" layout (eliminates bank conflicts)
  __shared__ float shared_a_tile_transposed[BM * BK]; // 128 x 8 = 1024 elements
  __shared__ float shared_b_tile_linearized[BK * BN]; // 8 x 128 = 1024 elements

  // === ADVANCE GLOBAL MEMORY POINTERS ===
  A += block_row_idx * BM * K; // move to current block's A data
  B += block_col_idx * BN; // move to current block's B data
  C += block_row_idx * BM * N + block_col_idx * BN; // move to current block's C data

  // === VECTORIZED LOADING INDICES ===
  // each thread loads 4 elements (128-bit / 32-bit = 4) using float4
  
  // A matrix loading indices (maintains transpose from vectorized kernel)
  const int a_load_row = threadIdx.x / (BK / 4);         // which row to load (0-127)
  const int a_load_col_group = threadIdx.x % (BK / 4);   // which group of 4 columns (0-1)
  
  // B matrix loading indices (for linearized storage)
  const int b_load_row = threadIdx.x / (BN / 4);         // which row to load (0-7)
  const int b_load_col_group = threadIdx.x % (BN / 4);   // which group of 4 columns (0-31)

  // === THREAD-LOCAL STORAGE ===
  float thread_output_tile[TM * TN] = {0.0};    // 8x8 output tile per thread
  float reg_a_values[TM] = {0.0};               // register cache for A values
  float reg_b_values[TN] = {0.0};               // register cache for B values

  // === MAIN K-DIMENSION LOOP ===
  for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
    
    // === VECTORIZED A MATRIX LOADING WITH TRANSPOSE ===
    // identical to vectorized kernel: transpose A during loading for vectorized access
    float4 a_vector = reinterpret_cast<float4 *>(&A[a_load_row * K + a_load_col_group * 4])[0];
    
    // store in transposed layout: As[col][row] instead of As[row][col]
    shared_a_tile_transposed[(a_load_col_group * 4 + 0) * BM + a_load_row] = a_vector.x;
    shared_a_tile_transposed[(a_load_col_group * 4 + 1) * BM + a_load_row] = a_vector.y;
    shared_a_tile_transposed[(a_load_col_group * 4 + 2) * BM + a_load_row] = a_vector.z;
    shared_a_tile_transposed[(a_load_col_group * 4 + 3) * BM + a_load_row] = a_vector.w;

    // === VECTORIZED B MATRIX LOADING WITH "LINEARIZATION" ===
    // NEW: complex storage pattern designed to eliminate bank conflicts
    float4 b_vector = reinterpret_cast<float4 *>(&B[b_load_row * N + b_load_col_group * 4])[0];
    
    // COORDINATE TRANSFORMATION FORMULA:
    // maps loading coordinates (b_load_row, b_load_col_group, offset) 
    // to storage coordinates that enable conflict-free access during computation

    // FORMULA BREAKDOWN:
    // - b_load_col_group % 2: creates even/odd groups (0,1,0,1,...)
    // - b_load_col_group / 2: pairs threads (0,0,1,1,2,2,...)  
    // - b_load_row * 8: separates data from different rows
    // - offset: position within float4 (0,1,2,3)
    // - * 16: provides stride spacing for consecutive access during computation

    // RESULT: when computation accesses Bs[base + thread_col_idx], consecutive
    // thread_col_idx values hit consecutive addresses → consecutive banks → no conflicts
    
    shared_b_tile_linearized[((b_load_col_group % 2) * 4 + b_load_row * 8 + 0) * 16 + b_load_col_group / 2] = b_vector.x;
    shared_b_tile_linearized[((b_load_col_group % 2) * 4 + b_load_row * 8 + 1) * 16 + b_load_col_group / 2] = b_vector.y;
    shared_b_tile_linearized[((b_load_col_group % 2) * 4 + b_load_row * 8 + 2) * 16 + b_load_col_group / 2] = b_vector.z;
    shared_b_tile_linearized[((b_load_col_group % 2) * 4 + b_load_row * 8 + 3) * 16 + b_load_col_group / 2] = b_vector.w;
    
    __syncthreads(); // ensure all threads finish loading before computation

    // === ADVANCE TO NEXT BLOCK TILES ===
    A += BK;           // move BK columns right in A
    B += BK * N;       // move BK rows down in B

    // === COMPUTE DOT PRODUCTS FOR THIS BLOCK TILE ===
    for (int dot_product_idx = 0; dot_product_idx < BK; ++dot_product_idx) {
      
      // === LOAD A VALUES INTO REGISTERS (VECTORIZED ACCESS) ===
      // transposed layout enables stride-1 access → vectorized LDS.128 instructions
      for (int i = 0; i < TM; ++i) {
        reg_a_values[i] = shared_a_tile_transposed[dot_product_idx * BM + thread_row_idx * TM + i];
      }
      
      // === LOAD B VALUES INTO REGISTERS (CONFLICT-FREE ACCESS) ===
      // NEW ACCESS PATTERN: designed to work with linearized storage
      // ensures consecutive thread_col_idx values access consecutive addresses

      // CONFLICT ANALYSIS:
      // when all threads load reg_b_values[i] simultaneously:
      // - thread 0: accesses Bs[(dot_product_idx*8 + i)*16 + 0]
      // - thread 1: accesses Bs[(dot_product_idx*8 + i)*16 + 1]  
      // - thread 2: accesses Bs[(dot_product_idx*8 + i)*16 + 2]
      // - thread 15: accesses Bs[(dot_product_idx*8 + i)*16 + 15]

      // ADDRESSES: base, base+1, base+2, ..., base+15 (consecutive!)
      // BANKS: consecutive addresses → consecutive banks → zero conflicts!
      
      for (int i = 0; i < TN; ++i) {
        reg_b_values[i] = shared_b_tile_linearized[(dot_product_idx * 8 + i) * 16 + thread_col_idx];
      }
      
      // === COMPUTE OUTER PRODUCT AND ACCUMULATE ===
      // identical to previous kernels: 8x8 outer product per thread
      for (int result_row = 0; result_row < TM; ++result_row) {
        for (int result_col = 0; result_col < TN; ++result_col) {
          thread_output_tile[result_row * TN + result_col] +=
              reg_a_values[result_row] * reg_b_values[result_col];
        }
      }
    }
    
    __syncthreads(); // ensure all threads finish computation before next iteration
  }

  // === VECTORIZED RESULTS WRITE TO GLOBAL MEMORY ===
  // identical to vectorized kernel: use float4 for 128-bit stores
  for (int result_row = 0; result_row < TM; result_row += 1) {
    for (int result_col = 0; result_col < TN; result_col += 4) {
      
      // vectorized read-modify-write using float4
      int global_row = thread_row_idx * TM + result_row;
      int global_col = thread_col_idx * TN + result_col;
      
      float4 c_vector = reinterpret_cast<float4 *>(&C[global_row * N + global_col])[0];
      
      // perform GEMM update: C = alpha * A*B + beta * C
      c_vector.x = alpha * thread_output_tile[result_row * TN + result_col + 0] + beta * c_vector.x;
      c_vector.y = alpha * thread_output_tile[result_row * TN + result_col + 1] + beta * c_vector.y;
      c_vector.z = alpha * thread_output_tile[result_row * TN + result_col + 2] + beta * c_vector.z;
      c_vector.w = alpha * thread_output_tile[result_row * TN + result_col + 3] + beta * c_vector.w;
      
      // vectorized write back using 128-bit store
      reinterpret_cast<float4 *>(&C[global_row * N + global_col])[0] = c_vector;
    }
  }
}

// this kernel combines multiple optimizations:
// 1. vectorized GMEM access (float4) for 4x bandwidth improvement
// 2. transposed A matrix storage for vectorized SMEM access  
// 3. "linearized" B matrix storage for bank conflict elimination
// 4. coordinate transformation that maps natural access patterns to conflict-free patterns
// - maintains all vectorization benefits from previous kernel
// - eliminates bank conflicts that could reduce bandwidth by 32x
// - expected additional speedup: several hundred GFLOPS
// - total performance: approaching cuBLAS levels
// the complex storage formula is the price paid to make computation access simple and efficient
// coordinate transformation enables conflict-free access while preserving mathematical correctness






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
        printf("Error at index %d - GPU = %f, CPU = %f (diff = %f)\n", 
               i, gpu_C[i], cpu_C[i], diff);
      }
    }
  }
  
  if (errors > 0) {
    printf("Verification FAILED - %d errors found out of %d elements\n", errors, M * N);
    return 0;
  } else {
    printf("Verification PASSED - All values match within epsilon %e\n", epsilon);
    return 1;
  }
}

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error - %s at %s -%d\n", \
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
    sgemm_resolve_bank_conflicts<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  // calculate the statistics and print the statistics
  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  float avg_time_ms = milliseconds / num_iterations;
  double gflops = (2.0 * M * N * K) / (avg_time_ms * 1e6);
  printf("average kernel execution time - %.3f ms\n", avg_time_ms);
  printf("performance - %.2f GFLOPS\n", gflops);

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