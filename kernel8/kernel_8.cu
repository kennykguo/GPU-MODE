#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// this kernel builds directly on the vectorized kernel by adding a simple padding strategy
// to resolve shared memory bank conflicts while maintaining all vectorization benefits
// KEY INNOVATION: instead of complex coordinate transformations, simply add 5 extra columns
// to the B matrix shared memory layout to break systematic bank conflict pattern
// - shared memory has 32 banks, each 4 bytes wide
// - bank number = (address ÷ 4) % 32
// - conflicts occur when multiple threads simultaneously access same bank, different addresses
// - conflicts can reduce memory bandwidth by up to 32x
// in the vectorized kernel, the B matrix had stride of exactly 128 elements between rows
// since 128 = 4 × 32, this created systematic repetition in bank access patterns
// when all threads simultaneously load regN values, they hit predictable bank conflicts
// add 5 padding columns per row, changing stride from 128 to 133 elements
// since 133 = 4 × 32 + 5, this breaks the systematic pattern and shifts bank access
// cost: only 5 × 8 × 4 = 160 bytes additional shared memory per block
// trades tiny amount of memory for significant performance improvement
// much simpler than complex addressing schemes while achieving similar results


__global__ void sgemm_resolve_bank_conflicts_with_padding(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  // === BLOCK TILE DIMENSIONS ===
  // BM=128, BN=128, BK=8: each block processes 128×128 output tile
  // TM=8, TN=8: each thread processes 8×8 sub-tile within block tile
  // total threads per block: (BM×BN)/(TM×TN) = (128×128)/(8×8) = 256 threads
  
  const int BM = 128; // BM - rows per block tile
  const int BN = 128; // BN - columns per block tile
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  // === BLOCK POSITION IN GRID ===
  const int block_row_idx = blockIdx.y;  // which 128×128 block row in grid
  const int block_col_idx = blockIdx.x;  // which 128×128 block column in grid

  // === THREAD POSITION WITHIN BLOCK ===
  // 256 threads arranged as 16×16 logical grid, each handling 8×8 output elements
  const int threads_per_col = BN / TN;                       // 128/8 = 16 threads span column
  const int thread_col_idx = threadIdx.x % threads_per_col;  // 0-15: which thread column  
  const int thread_row_idx = threadIdx.x / threads_per_col;  // 0-15: which thread row

  // === SHARED MEMORY ALLOCATION ===
  // A matrix: maintains transposed layout from vectorized kernel for efficient access
  // B matrix: NEW - padded layout to resolve bank conflicts
  __shared__ float shared_a_tile_transposed[BM * BK];     // 128 × 8 = 1024 elements (unchanged)
  
  const int padding_cols = 5;  // magic number chosen to break bank conflict patterns
  __shared__ float shared_b_tile_padded[BK * (BN + padding_cols)];  // 8 × 133 = 1064 elements

  // === ADVANCE GLOBAL MEMORY POINTERS ===
  // move pointers to start of current block's data
  A += block_row_idx * BM * K;                    // advance to block row start
  B += block_col_idx * BN;                        // advance to block column start  
  C += block_row_idx * BM * N + block_col_idx * BN; // advance to output block position

  // === VECTORIZED LOADING INDICES ===
  // each thread loads 4 elements (128-bit / 32-bit = 4) using float4
  // maintains vectorization benefits from previous kernel
  
  // A matrix loading indices (unchanged from vectorized kernel)
  const int a_load_row = threadIdx.x / (BK / 4);         // which row to load in A (0-127)
  const int a_load_col_group = threadIdx.x % (BK / 4);   // which group of 4 columns (0-1)
  
  // B matrix loading indices (same calculation, different storage due to padding)
  const int b_load_row = threadIdx.x / (BN / 4);         // which row to load in B (0-7)
  const int b_load_col_group = threadIdx.x % (BN / 4);   // which group of 4 columns (0-31)

  // === THREAD-LOCAL STORAGE ===
  // each thread accumulates results for its 8×8 output sub-tile
  float thread_output_tile[TM * TN] = {0.0};      // 64 results per thread (8×8 tile)
  float reg_a_values[TM] = {0.0};                 // register cache for A column slice (8 elements)
  float reg_b_values[TN] = {0.0};                 // register cache for B row slice (8 elements)

  // === MAIN K-DIMENSION LOOP ===
  // iterate through k-dimension in chunks of BK=8 elements
  for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
    
    // === VECTORIZED A MATRIX LOADING WITH TRANSPOSE ===
    // identical to vectorized kernel: transpose A during loading for efficient register access
    float4 a_vector = reinterpret_cast<float4 *>(&A[a_load_row * K + a_load_col_group * 4])[0];
    
    // store in transposed layout: As[col][row] instead of As[row][col]
    // enables stride-1 access during computation → vectorized LDS.128 instructions
    shared_a_tile_transposed[(a_load_col_group * 4 + 0) * BM + a_load_row] = a_vector.x;
    shared_a_tile_transposed[(a_load_col_group * 4 + 1) * BM + a_load_row] = a_vector.y;
    shared_a_tile_transposed[(a_load_col_group * 4 + 2) * BM + a_load_row] = a_vector.z;
    shared_a_tile_transposed[(a_load_col_group * 4 + 3) * BM + a_load_row] = a_vector.w;

    // === VECTORIZED B MATRIX LOADING WITH PADDING ===
    // NEW: simple row-major storage but with 133-element stride instead of 128
    // the extra 5 columns per row are never accessed but change memory layout
    float4 b_vector = reinterpret_cast<float4 *>(&B[b_load_row * N + b_load_col_group * 4])[0];
    
    // PADDING LAYOUT EXPLANATION:
    // each row occupies (BN + padding_cols) = 133 memory locations
    // only first BN=128 locations contain actual data, last 5 are unused padding
    // this breaks the systematic bank access patterns that caused conflicts
    //
    // MEMORY LAYOUT:
    // Row 0: [B₀₀ B₀₁ ... B₀₁₂₇ | X X X X X] ← 5 unused padding elements
    // Row 1: [B₁₀ B₁₁ ... B₁₁₂₇ | X X X X X] ← 5 unused padding elements
    // ...
    // stride between rows: 133 elements instead of 128
    
    shared_b_tile_padded[b_load_row * (BN + padding_cols) + b_load_col_group * 4 + 0] = b_vector.x;
    shared_b_tile_padded[b_load_row * (BN + padding_cols) + b_load_col_group * 4 + 1] = b_vector.y;
    shared_b_tile_padded[b_load_row * (BN + padding_cols) + b_load_col_group * 4 + 2] = b_vector.z;
    shared_b_tile_padded[b_load_row * (BN + padding_cols) + b_load_col_group * 4 + 3] = b_vector.w;
    
    __syncthreads(); // ensure all threads finish loading before computation

    // === ADVANCE TO NEXT BLOCK TILES ===
    A += BK;           // move BK columns right in A
    B += BK * N;       // move BK rows down in B

    // === COMPUTE DOT PRODUCTS FOR THIS BLOCK TILE ===
    // each thread computes its 8×8 output tile through accumulation of partial outer products
    for (int dot_product_idx = 0; dot_product_idx < BK; ++dot_product_idx) {
      
      // === LOAD A VALUES INTO REGISTERS (COLUMN SLICE) ===
      // load 8 consecutive A elements from transposed shared memory
      // represents one column slice from original A matrix needed for this thread's computation
      for (int i = 0; i < TM; ++i) {
        reg_a_values[i] = shared_a_tile_transposed[dot_product_idx * BM + thread_row_idx * TM + i];
      }
      // transposed layout enables stride-1 access → vectorized LDS.128 instructions
      
      // === LOAD B VALUES INTO REGISTERS (ROW SLICE) ===
      // NEW ACCESS PATTERN: uses padded stride of 133 instead of 128
      // represents one row slice from original B matrix needed for this thread's computation
      for (int i = 0; i < TN; ++i) {
        reg_b_values[i] = shared_b_tile_padded[dot_product_idx * (BN + padding_cols) + thread_col_idx * TN + i];
      }
      
      // === BANK CONFLICT RESOLUTION IN ACTION ===
      // when all threads simultaneously load reg_b_values[0], they access:
      // thread 0: Bs[dot_product_idx * 133 + 0]
      // thread 1: Bs[dot_product_idx * 133 + 8]  
      // thread 2: Bs[dot_product_idx * 133 + 16]
      // ...
      // the 133-element stride (vs 128 in vectorized kernel) shifts bank access patterns
      // between different dot_product_idx iterations, reducing systematic conflicts
      //
      // KEY INSIGHT: 133 = 4×32 + 5, so bank pattern shifts by 5 positions each iteration
      // this breaks the repetitive pattern that occurred with 128 = 4×32 exactly
      
      // === COMPUTE OUTER PRODUCT AND ACCUMULATE ===
      // compute 8×8 outer product of reg_a_values (column) and reg_b_values (row)
      // identical computation to vectorized kernel, just with different memory access pattern
      for (int result_row = 0; result_row < TM; ++result_row) {
        for (int result_col = 0; result_col < TN; ++result_col) {
          thread_output_tile[result_row * TN + result_col] +=
              reg_a_values[result_row] * reg_b_values[result_col];
        }
      }
      // each thread performs 64 multiply-accumulate operations per dot_product_idx
      // final result: complete 8×8 output tile after processing all K-dimension chunks
    }
    
    __syncthreads(); // ensure all threads finish computation before next iteration
  }

  // === VECTORIZED RESULTS WRITE TO GLOBAL MEMORY ===
  // identical to vectorized kernel: use float4 for 128-bit stores
  // maintains all output optimization benefits while adding bank conflict resolution
  for (int result_row = 0; result_row < TM; result_row += 1) {
    for (int result_col = 0; result_col < TN; result_col += 4) {
      
      // vectorized read-modify-write using float4 for optimal bandwidth
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

// - maintains all vectorization benefits (float4 loading, transposed A matrix, vectorized output)
// - adds simple padding strategy to resolve remaining bank conflicts
// - changes B matrix stride from 128 to 133 elements per row
// - breaks systematic bank conflict patterns without complex addressing logic
// - cost: only 160 bytes additional shared memory per block (tiny overhead)
// - mathematical: 133 = 4×32 + 5 breaks the exact multiple of 32 that caused conflicts
// - empirical: chosen through testing as minimal padding that effectively reduces conflicts
// - engineering: represents sweet spot between conflict reduction and memory overhead




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
    sgemm_resolve_bank_conflicts_with_padding<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
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