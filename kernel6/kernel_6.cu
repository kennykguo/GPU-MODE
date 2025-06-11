#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


// vectorized sgemm kernel with 128-bit memory access optimization
// optimizations - vectorized gmem loads/stores, transposed As for vectorized smem access
__global__ void sgemm_vectorized(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  // BLOCK TILE DIMENSIONS
  // each block processes a 128x128 tile of the output matrix
  const int block_tile_rows = 128;     // BM - rows per block tile
  const int block_tile_cols = 128;     // BN - columns per block tile
  const int block_tile_k_dim = 8;      // BK - k-dimension per block tile iteration
  
  // THREAD TILE DIMENSIONS
  // each thread processes an 8x8 sub-tile within the block tile
  const int thread_tile_rows = 8;      // TM - rows per thread
  const int thread_tile_cols = 8;      // TN - columns per thread
  
  // BLOCK POSITION IN GRID
  // which 128x128 block this thread block is responsible for
  const int block_row_idx = blockIdx.y;  // which block row in grid
  const int block_col_idx = blockIdx.x;  // which block column in grid

  // THREAD COUNT VERIFICATION
  const int total_results_per_block = block_tile_rows * block_tile_cols; // 128*128 = 16384
  const int num_threads_per_block = total_results_per_block / (thread_tile_rows * thread_tile_cols); // 16384/64 = 256
  assert(num_threads_per_block == blockDim.x); // ensure 256 threads per block

  // THREAD POSITION WITHIN BLOCK
  // 256 threads arranged as 16x16 grid, each handling 8x8 output elements
  const int threads_per_col = block_tile_cols / thread_tile_cols; // 128/8 = 16
  const int thread_col_idx = threadIdx.x % threads_per_col;       // 0-15 - which thread column
  const int thread_row_idx = threadIdx.x / threads_per_col;       // 0-15 - which thread row

  // SHARED MEMORY ALLOCATION
  // As is transposed - [BK * BM] instead of [BM * BK] for vectorized access
  __shared__ float shared_a_tile_transposed[block_tile_k_dim * block_tile_rows]; // 8 x 128 = 1024 elements
  __shared__ float shared_b_tile[block_tile_k_dim * block_tile_cols];            // 8 x 128 = 1024 elements

  // ADVANCE GLOBAL MEMORY POINTERS
  // move pointers to start of current block's data
  A += block_row_idx * block_tile_rows * K;                    // advance to block row start
  B += block_col_idx * block_tile_cols;                        // advance to block column start
  C += block_row_idx * block_tile_rows * N + block_col_idx * block_tile_cols; // advance to output block

  // VECTORIZED LOADING INDICES
  // calculate which elements each thread loads using float4 (4 elements per thread)
  
  // A matrix loading - 128x8 elements, 256 threads, each loads 4 elements
  // threads now load groups of 4 consecutive elements for vectorization
  const int vector_width = 4; // float4 loads 4 elements
  
  const int a_load_row = threadIdx.x / (block_tile_k_dim / vector_width);    // which row in A (0-127)
  const int a_load_col_group = threadIdx.x % (block_tile_k_dim / vector_width); // which group of 4 cols (0-1)
  const int a_row_stride = (num_threads_per_block * vector_width) / block_tile_k_dim; // 256*4/8 = 128 rows per iteration
  
  // B matrix loading - 8x128 elements, 256 threads, each loads 4 elements  
  const int b_load_row = threadIdx.x / (block_tile_cols / vector_width);     // which row in B (0-7) 
  const int b_load_col_group = threadIdx.x % (block_tile_cols / vector_width); // which group of 4 cols (0-31)
  const int b_row_stride = num_threads_per_block / (block_tile_cols / vector_width); // 256/32 = 8 rows per iteration

  // THREAD-LOCAL STORAGE
  // each thread accumulates results for its 8x8 output sub-tile
  float thread_results[thread_tile_rows * thread_tile_cols] = {0.0}; // 64 results per thread
  float reg_a_values[thread_tile_rows] = {0.0};                      // cache for A values (8 elements)
  float reg_b_values[thread_tile_cols] = {0.0};                      // cache for B values (8 elements)

  // MAIN K-DIMENSION LOOP
  // iterate through k-dimension in chunks of block_tile_k_dim (8)
  for (int k_block_start = 0; k_block_start < K; k_block_start += block_tile_k_dim) {
    
    // VECTORIZED A MATRIX LOADING WITH TRANSPOSE
    // load 4 consecutive floats from A using float4 for 128-bit bandwidth
    // transpose during GMEM-SMEM transfer for efficient regM loading later
    float4 a_vector = reinterpret_cast<float4 *>(&A[a_load_row * K + a_load_col_group * vector_width])[0];
    
    // store transposed - As[col][row] = As[col * block_tile_rows + row]
    // this enables stride-1 access when loading regM values during computation
    shared_a_tile_transposed[(a_load_col_group * vector_width + 0) * block_tile_rows + a_load_row] = a_vector.x;
    shared_a_tile_transposed[(a_load_col_group * vector_width + 1) * block_tile_rows + a_load_row] = a_vector.y;
    shared_a_tile_transposed[(a_load_col_group * vector_width + 3) * block_tile_rows + a_load_row] = a_vector.z;
    shared_a_tile_transposed[(a_load_col_group * vector_width + 3) * block_tile_rows + a_load_row] = a_vector.w;

    // VECTORIZED B MATRIX LOADING (NO TRANSPOSE)
    // direct vectorized copy using float4 for 128-bit bandwidth
    // B doesn't need transpose since regN loading was already stride-1 in original layout
    reinterpret_cast<float4 *>(&shared_b_tile[b_load_row * block_tile_cols + b_load_col_group * vector_width])[0] =
        reinterpret_cast<float4 *>(&B[b_load_row * N + b_load_col_group * vector_width])[0];
    
    __syncthreads(); // ensure all threads finish loading before computation

    // ADVANCE TO NEXT BLOCK TILES
    A += block_tile_k_dim; // move 8 columns right in A
    B += block_tile_k_dim * N; // move 8 rows down in B

    // COMPUTE DOT PRODUCTS FOR THIS BLOCK TILE
    // iterate through k-dimension of current block tile (8 iterations)
    for (int dot_product_idx = 0; dot_product_idx < block_tile_k_dim; ++dot_product_idx) {
      
      // LOAD A VALUES INTO REGISTERS
      // load 8 consecutive A elements from transposed shared memory
      // transposed layout enables stride-1 access - vectorized LDS.128 instructions
      // Now As is indexed the same as Bs
      for (int i = 0; i < thread_tile_rows; ++i) {
        reg_a_values[i] = shared_a_tile_transposed[dot_product_idx * block_tile_rows + thread_row_idx * thread_tile_rows + i];
      }
      // access pattern - consecutive elements with stride 1
      // hardware generates - 2x LDS.128 per SMEM (loads 4 elements each) instead of 8x LDS.32 (single val loaded)
      
      // LOAD B VALUES INTO REGISTERS
      // load 8 consecutive B elements (already stride-1 in original layout)
      for (int i = 0; i < thread_tile_cols; ++i) {
        reg_b_values[i] = shared_b_tile[dot_product_idx * block_tile_cols + thread_col_idx * thread_tile_cols + i];
      }
      
      // COMPUTE OUTER PRODUCT AND ACCUMULATE
      // compute 8x8 outer product of reg_a_values and reg_b_values
      // TWO for loops calculate partial dot product (looks like a permutation, because its for the entire 8x8 block)
      for (int result_row = 0; result_row < thread_tile_rows; ++result_row) {
        for (int result_col = 0; result_col < thread_tile_cols; ++result_col) {
          thread_results[result_row * thread_tile_cols + result_col] += 
              reg_a_values[result_row] * reg_b_values[result_col];
        }
      }
      // produces 64 multiply-accumulate operations per thread per dot_product_idx
    }
    
    __syncthreads(); // ensure all threads finish computation before next iteration
  }

  // VECTORIZED RESULTS WRITE TO GLOBAL MEMORY
  // write 8x8 results using vectorized stores for improved bandwidth
  // process 2 rows at a time, 4 elements per row using float4
  for (int result_row = 0; result_row < thread_tile_rows; result_row += 1) {
    for (int result_col = 0; result_col < thread_tile_cols; result_col += vector_width) {
      
      // VECTORIZED READ-MODIFY-WRITE
      // load 4 consecutive C values using 128-bit load
      int global_row = thread_row_idx * thread_tile_rows + result_row;
      int global_col = thread_col_idx * thread_tile_cols + result_col;
      
      float4 c_vector = reinterpret_cast<float4 *>(&C[global_row * N + global_col])[0];
      
      // perform GEMM update - C = alpha * A*B + beta * C
      c_vector.x = alpha * thread_results[result_row * thread_tile_cols + result_col + 0] + beta * c_vector.x;
      c_vector.y = alpha * thread_results[result_row * thread_tile_cols + result_col + 1] + beta * c_vector.y;
      c_vector.z = alpha * thread_results[result_row * thread_tile_cols + result_col + 2] + beta * c_vector.z;
      c_vector.w = alpha * thread_results[result_row * thread_tile_cols + result_col + 3] + beta * c_vector.w;
      
      // write back using 128-bit store
      reinterpret_cast<float4 *>(&C[global_row * N + global_col])[0] = c_vector;
    }
  }
  // each thread writes 8 rows × 2 vectorized stores = 16 total STG.E.128 instructions
  // compared to 64 individual STG.E instructions in non-vectorized version
}

// vectorized GMEM loads - LDG.E.128 instead of LDG.E (4x bandwidth)
// vectorized GMEM stores - STG.E.128 instead of STG.E (4x bandwidth)  
// transposed As - enables vectorized SMEM loads LDS.128 (4x fewer instructions)
// float4 alignment - guarantees 16-byte alignment for 128-bit instructions


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
    sgemm_vectorized<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
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