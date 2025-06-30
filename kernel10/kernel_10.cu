#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int autotuning_num_threads = 256;  // fixed thread count per block for all configurations
const int BM = 64; // BM - rows per block tile
const int BN = 128; // BN - columns per block tile
const int BK = 8;
const uint TN = 4;
const uint TM = 4;

// previous kernels - Block → Thread (two levels)
// this kernel - Block → Warp → Thread (three levels, hardware-aware)
// CONCEPTUAL BREAKTHROUGH - COORDINATED vs INDEPENDENT EXECUTION
// previous approach - each thread independently computes one tile
// warptiling approach - threads coordinate within warps to collectively compute larger regions,
// with each thread handling multiple smaller tiles in a synchronized pattern
// PERFORMANCE BENEFITS -
// 1. register cache locality - threads in warps access related data in coordinated patterns
// 2. instruction-level parallelism - tighter computation loops enable better hardware optimization
// 3. tensor core preparation - structure aligns with future warp-wide matrix instructions
// 4. explicit hardware alignment - computation organized around GPU's natural 32-thread execution units
// MENTAL MODEL - COORDINATED ASSEMBLY LINE
// think of a warp as 32 workers on an assembly line who must all perform the same operation
// simultaneously. instead of each worker completing an entire product independently, they
// coordinate to process larger batches more efficiently, with each worker handling multiple
// small pieces of the collective work.
const int warp_size = 32; // GPU hardware constant - 32 threads execute in lockstep per warp

// these parameters define the three-level tiling hierarchy and work distribution
const int num_threads_per_block = 128;  // total threads per block (4 warps - 128/32 = 4)

__global__ void __launch_bounds__(num_threads_per_block) sgemm_warptiling_coordinated(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  
  // these constants define the three-level structure - block → warp → thread
  
  // LEVEL 1 - BLOCK TILE DIMENSIONS (shared memory caching level)
  const int block_tile_rows = 64; // BM - rows of output computed per block
  const int block_tile_cols = 128; // BN - columns of output computed per block  
  const int block_tile_k_dim = 8; // BK - k-dimension processed per iteration
  
  // LEVEL 2 - WARP TILE DIMENSIONS (coordination level - NEW!)
  // each warp within the block computes a WM×WN region of the block's tile
  const int warp_tile_rows = 32; // WM - rows of output computed per warp
  const int warp_tile_cols = 64; // WN - columns of output computed per warp
  
  // LEVEL 3 - THREAD TILE DIMENSIONS (register level)
  const int thread_tile_rows = 4; // TM - rows computed per thread per subtile
  const int thread_tile_cols = 4; // TN - columns computed per thread per subtile
  
  // WARP COORDINATION PARAMETERS
  // these define how warps organize their collective work across multiple subtiles
  const int warp_subtile_iterations_n = 2; // WNITER - how many column subtiles per thread
  
  // DERIVED WARP ORGANIZATION (calculated from the coordination requirements)
  // this ensures that warp's total work (WM×WN) is perfectly distributed among its 32 threads
  // formula - WMITER = (WM × WN) / (warp_size × TM × TN × WNITER)
  constexpr int warp_subtile_iterations_m = (warp_tile_rows * warp_tile_cols) / 
                                           (warp_size * thread_tile_rows * thread_tile_cols * warp_subtile_iterations_n);
  // calculation - (32 × 64) / (32 × 4 × 4 × 2) = 2048 / 1024 = 2
  
  // SUBTILE DIMENSIONS (how big each subtile is)
  constexpr int warp_subtile_rows = warp_tile_rows / warp_subtile_iterations_m;  // 32/2 = 16  
  constexpr int warp_subtile_cols = warp_tile_cols / warp_subtile_iterations_n;  // 64/2 = 32
  
  // BLOCK POSITION IN GRID
  const int block_row_idx = blockIdx.y;  // which block row in grid
  const int block_col_idx = blockIdx.x;  // which block column in grid

  // WARP IDENTIFICATION AND POSITIONING (LEVEL 2 ORGANIZATION)
  // this is where we establish the warp-level coordination structure
  
  // WARP IDENTITY - which of the 4 warps (0,1,2,3) does this thread belong to?
  const int warp_id = threadIdx.x / warp_size;  // 0-3 for our 128-thread block
  
  // WARP POSITION WITHIN BLOCK - where does this warp's territory sit?
  // with BN=128, WN=64 - 2 warps span horizontally (128/64 = 2)
  // with BM=64, WM=32 - 2 warps span vertically (64/32 = 2)  
  // total - 2×2 = 4 warps per block (matches our thread count - 128/32 = 4)
  const int warp_col_in_block = warp_id % (block_tile_cols / warp_tile_cols);  // 0-1
  const int warp_row_in_block = warp_id / (block_tile_cols / warp_tile_cols);  // 0-1
  
  // THREAD POSITIONING WITHIN WARP (LEVEL 3 ORGANIZATION)
  // now we position each thread within its warp's subtile structure
  
  // THREAD IDENTITY WITHIN WARP - which of the 32 threads (0-31) in this warp?
  const int thread_id_in_warp = threadIdx.x % warp_size;  // 0-31
  
  // THREAD POSITION WITHIN SUBTILE - where does this thread compute within each subtile?
  // each subtile is 16×32, with each thread handling 4×4, so we have 4×8 thread positions per subtile
  const int threads_per_subtile_col = warp_subtile_cols / thread_tile_cols;  // 32/4 = 8
  const int thread_col_in_subtile = thread_id_in_warp % threads_per_subtile_col;  // 0-7
  const int thread_row_in_subtile = thread_id_in_warp / threads_per_subtile_col;  // 0-3
  
  // SHARED MEMORY ALLOCATION
  // maintains optimizations from previous kernels - transposed A, vectorized access
  __shared__ float shared_a_transposed[block_tile_rows * block_tile_k_dim]; // 64×8 = 512 elements
  __shared__ float shared_b_tile[block_tile_k_dim * block_tile_cols]; // 8×128 = 1024 elements

  // GLOBAL MEMORY POINTER ADVANCEMENT
  // move pointers to this block's data regions
  A += block_row_idx * block_tile_rows * K; // advance to block's A rows
  B += block_col_idx * block_tile_cols; // advance to block's B columns
  
  // WARP-SPECIFIC C POINTER ADVANCEMENT (new optimization!)
  // instead of advancing C per thread, we advance it per warp, then use relative addressing
  // this reduces redundant address calculations across threads in the same warp
  C += (block_row_idx * block_tile_rows + warp_row_in_block * warp_tile_rows) * N + 
       block_col_idx * block_tile_cols + warp_col_in_block * warp_tile_cols;

  // these calculations enable vectorized loads while working correctly across the new hierarchy
  const int a_load_row = threadIdx.x / (block_tile_k_dim / 4); // which row to load (0-31)
  const int a_load_col_group = threadIdx.x % (block_tile_k_dim / 4); // which group of 4 cols (0-1)
  constexpr int a_load_stride = (num_threads_per_block * 4) / block_tile_k_dim; // 128*4/8 = 64 rows per iteration
  
  const int b_load_row = threadIdx.x / (block_tile_cols / 4); // which row to load (0-3)
  const int b_load_col_group = threadIdx.x % (block_tile_cols / 4); // which group of 4 cols (0-31)
  constexpr int b_load_stride = num_threads_per_block / (block_tile_cols / 4); // 128/32 = 4 rows per iteration

  // WARP-COORDINATED REGISTER ALLOCATION 
  // this is where the warptiling coordination becomes visible in memory allocation
  // each thread must store results for ALL subtiles it will process in coordination with its warp
  // warp_subtile_iterations_m=2, warp_subtile_iterations_n=2, thread_tile_rows=4, thread_tile_cols=4
  // total - 2×4×2×4 = 64 elements per thread (4 subtiles × 16 elements each)
  float thread_results_all_subtiles[warp_subtile_iterations_m * thread_tile_rows * warp_subtile_iterations_n * thread_tile_cols] = {0.0};

  // WARP-LEVEL REGISTER CACHES (new approach!)
  // instead of caching data for just one thread tile, we cache data for the thread's portion
  // of ALL subtiles that this warp will process, enabling coordinated computation
  float reg_a_warp_data[warp_subtile_iterations_m * thread_tile_rows] = {0.0};  // 2×4 = 8 elements  
  float reg_b_warp_data[warp_subtile_iterations_n * thread_tile_cols] = {0.0};  // 2×4 = 8 elements

  // MAIN K-DIMENSION LOOP
  // processes matrix multiplication in chunks along the K dimension
  for (int k_block_start = 0; k_block_start < K; k_block_start += block_tile_k_dim) {
    
    // VECTORIZED A MATRIX LOADING WITH TRANSPOSE  
    // maintains optimization from previous kernels - transpose during loading for efficient access
    for (int load_offset = 0; load_offset + a_load_stride <= block_tile_rows; load_offset += a_load_stride) {
      float4 a_vector = reinterpret_cast<float4 *>(
          &A[(a_load_row + load_offset) * K + a_load_col_group * 4])[0];
      
      // store in transposed layout for stride-1 access during computation
      shared_a_transposed[(a_load_col_group * 4 + 0) * block_tile_rows + a_load_row + load_offset] = a_vector.x;
      shared_a_transposed[(a_load_col_group * 4 + 1) * block_tile_rows + a_load_row + load_offset] = a_vector.y;
      shared_a_transposed[(a_load_col_group * 4 + 2) * block_tile_rows + a_load_row + load_offset] = a_vector.z;
      shared_a_transposed[(a_load_col_group * 4 + 3) * block_tile_rows + a_load_row + load_offset] = a_vector.w;
    }

    // VECTORIZED B MATRIX LOADING
    // maintains row-major layout for efficient warp-coordinated access
    for (int load_offset = 0; load_offset + b_load_stride <= block_tile_k_dim; load_offset += b_load_stride) {
      reinterpret_cast<float4 *>(
          &shared_b_tile[(b_load_row + load_offset) * block_tile_cols + b_load_col_group * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(b_load_row + load_offset) * N + b_load_col_group * 4])[0];
    }
    
    __syncthreads(); // ensure all loading completes before computation

    // WARP-COORDINATED COMPUTATION - THE HEART OF WARPTILING
    // this is where the three-level hierarchy creates its sophisticated coordination pattern
    for (int dot_product_idx = 0; dot_product_idx < block_tile_k_dim; ++dot_product_idx) {
      
      // PHASE 1 - COORDINATED DATA LOADING FOR WARP'S COLLECTIVE WORK
      // all threads in the warp coordinate to load data for the warp's entire territory
      // this is fundamentally different from previous kernels where threads loaded independently
      
      // LOAD A DATA FOR ALL SUBTILE ROWS THIS THREAD WILL PROCESS
      // each thread loads its portion of data from ALL subtile rows in the warp's territory
      for (int warp_subtile_row = 0; warp_subtile_row < warp_subtile_iterations_m; ++warp_subtile_row) {
        for (int i = 0; i < thread_tile_rows; ++i) {
          // MEMORY ACCESS BREAKDOWN -
          // - dot_product_idx * block_tile_rows - move to correct k-slice in shared memory
          // - warp_row_in_block * warp_tile_rows - advance to this warp's row territory  
          // - warp_subtile_row * warp_subtile_rows - advance to specific subtile row
          // - thread_row_in_subtile * thread_tile_rows + i - thread's specific element
          reg_a_warp_data[warp_subtile_row * thread_tile_rows + i] =
              shared_a_transposed[(dot_product_idx * block_tile_rows) + 
                                  warp_row_in_block * warp_tile_rows + 
                                  warp_subtile_row * warp_subtile_rows +
                                  thread_row_in_subtile * thread_tile_rows + i];
        }
      }
      
      // LOAD B DATA FOR ALL SUBTILE COLUMNS THIS THREAD WILL PROCESS  
      // each thread loads its portion of data from ALL subtile columns in the warp's territory
      for (int warp_subtile_col = 0; warp_subtile_col < warp_subtile_iterations_n; ++warp_subtile_col) {
        for (int i = 0; i < thread_tile_cols; ++i) {
          // MEMORY ACCESS BREAKDOWN -
          // - dot_product_idx * block_tile_cols - move to correct k-slice in shared memory
          // - warp_col_in_block * warp_tile_cols - advance to this warp's column territory
          // - warp_subtile_col * warp_subtile_cols - advance to specific subtile column  
          // - thread_col_in_subtile * thread_tile_cols + i - thread's specific element
          reg_b_warp_data[warp_subtile_col * thread_tile_cols + i] =
              shared_b_tile[(dot_product_idx * block_tile_cols) + 
                            warp_col_in_block * warp_tile_cols + 
                            warp_subtile_col * warp_subtile_cols +
                            thread_col_in_subtile * thread_tile_cols + i];
        }
      }

      // PHASE 2 - COORDINATED COMPUTATION ACROSS WARP'S SUBTILES
      // all 32 threads in the warp now execute this computation pattern simultaneously
      // this creates the register cache locality and ILP benefits that warptiling provides

      // COMPUTATION PATTERN VISUALIZATION -
      // for each of the 4 subtiles (2×2 arrangement) -
      //   all 32 threads compute their 4×4 piece of current subtile simultaneously
      //   this creates much better register access patterns than scattered computation
      
      for (int warp_subtile_row = 0; warp_subtile_row < warp_subtile_iterations_m; ++warp_subtile_row) {
        for (int warp_subtile_col = 0; warp_subtile_col < warp_subtile_iterations_n; ++warp_subtile_col) {
          
          // COORDINATED 4×4 TILE COMPUTATION
          // all threads compute their piece of the current subtile simultaneously
          // this tight loop creates excellent instruction-level parallelism opportunities
          for (int thread_result_row = 0; thread_result_row < thread_tile_rows; ++thread_result_row) {
            for (int thread_result_col = 0; thread_result_col < thread_tile_cols; ++thread_result_col) {
              
              // RESULT STORAGE INDEXING (maintains coordination across subtiles)
              // results are stored in row-major order across all subtiles this thread processes
              int result_index = (warp_subtile_row * thread_tile_rows + thread_result_row) * 
                                (warp_subtile_iterations_n * thread_tile_cols) +
                                (warp_subtile_col * thread_tile_cols) + thread_result_col;
              
              thread_results_all_subtiles[result_index] += 
                  reg_a_warp_data[warp_subtile_row * thread_tile_rows + thread_result_row] *
                  reg_b_warp_data[warp_subtile_col * thread_tile_cols + thread_result_col];
            }
          }
        }
      }
    }
    
    // advance to next k-dimension block
    A += block_tile_k_dim; // move 8 columns right in A
    B += block_tile_k_dim * N; // move 8 rows down in B
    __syncthreads(); // ensure all computation completes before next iteration
  }

  // WARP-COORDINATED RESULTS WRITING
  // the output writing maintains the warp coordination structure, with all threads
  // in the warp writing their results to the appropriate regions of their collective territory
  
  for (int warp_subtile_row = 0; warp_subtile_row < warp_subtile_iterations_m; ++warp_subtile_row) {
    for (int warp_subtile_col = 0; warp_subtile_col < warp_subtile_iterations_n; ++warp_subtile_col) {
      
      // CALCULATE SUBTILE-SPECIFIC C POINTER
      // advance C pointer to the specific subtile this iteration will write
      float *c_subtile_base = C + (warp_subtile_row * warp_subtile_rows) * N + 
                                 warp_subtile_col * warp_subtile_cols;
      
      // VECTORIZED WRITE FOR THREAD'S 4×4 REGION WITHIN CURRENT SUBTILE
      for (int thread_result_row = 0; thread_result_row < thread_tile_rows; thread_result_row += 1) {
        for (int thread_result_col = 0; thread_result_col < thread_tile_cols; thread_result_col += 4) {
          
          // VECTORIZED READ-MODIFY-WRITE USING FLOAT4
          float4 c_vector = reinterpret_cast<float4 *>(
              &c_subtile_base[(thread_row_in_subtile * thread_tile_rows + thread_result_row) * N +
                             thread_col_in_subtile * thread_tile_cols + thread_result_col])[0];
          
          // CALCULATE INDEX INTO THREAD'S RESULT STORAGE
          const int result_base_index = (warp_subtile_row * thread_tile_rows + thread_result_row) * 
                                       (warp_subtile_iterations_n * thread_tile_cols) +
                                       warp_subtile_col * thread_tile_cols + thread_result_col;
          
          // PERFORM GEMM UPDATE - C = alpha * A*B + beta * C
          c_vector.x = alpha * thread_results_all_subtiles[result_base_index + 0] + beta * c_vector.x;
          c_vector.y = alpha * thread_results_all_subtiles[result_base_index + 1] + beta * c_vector.y;
          c_vector.z = alpha * thread_results_all_subtiles[result_base_index + 2] + beta * c_vector.z;
          c_vector.w = alpha * thread_results_all_subtiles[result_base_index + 3] + beta * c_vector.w;
          
          // VECTORIZED WRITE BACK
          reinterpret_cast<float4 *>(
              &c_subtile_base[(thread_row_in_subtile * thread_tile_rows + thread_result_row) * N +
                             thread_col_in_subtile * thread_tile_cols + thread_result_col])[0] = c_vector;
        }
      }
    }
  }
}


// ARCHITECTURAL PROGRESSION -
// 1. naive kernel - single-threaded approach
// 2. block tiling - parallel blocks with shared memory
// 3. vectorization - 128-bit memory access optimization  
// 4. bank conflict resolution - memory access pattern optimization
// 5. autotuning - systematic parameter exploration
// 6. warptiling - explicit hardware-aware coordination (THIS KERNEL)
// three-level tiling hierarchy aligns with GPU hardware structure
// warp-level coordination creates register cache locality
// coordinated computation patterns enable instruction-level parallelism  
// explicit organization around 32-thread execution units
// foundation for future tensor core optimizations
// the evolution from independent thread computation to coordinated warp computation
// represents a fundamental shift toward hardware-aware optimization, where algorithms
// are explicitly designed to match the natural execution patterns of the underlying hardware
// rather than fighting against them. this approach yields both immediate performance
// benefits and creates a foundation for even more advanced optimizations


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
  // need BM >= warp_tile_rows (which you satisfied - 128 >= 128)
  // BN >= warp_tile_cols (which you violated - 64 < 128)

  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  dim3 blockDim(num_threads_per_block, 1, 1);                            // 256 threads per block
  const int num_iterations = 10;
  
  // benchmark loop  
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i){
    // Reset C to original values before each iteration
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    sgemm_warptiling_coordinated<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
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