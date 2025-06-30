#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// this kernel represents the evolution from fixed-parameter optimization to systematic
// parameter space exploration through a flexible "warptile" organizational framework
// previous kernels had rigid one-to-one mapping: thread position → single output tile
// this kernel introduces flexibility: thread position → warptile position → multiple output tiles
// by separating "how threads are organized" from "how much work each thread does"
// we can explore hundreds of parameter combinations with a single robust implementation
// a warptile is an organizational unit that groups threads and defines how they work together
// to process larger regions of the output matrix. threads can compute multiple tiles within
// different warptiles, enabling flexible work distribution while maintaining consistent
// thread coordination patterns
// this framework enables systematic exploration of ~400 parameter combinations to find
// optimal settings for specific hardware and problem characteristics, similar to how
// libraries like cuBLAS and Triton achieve high performance across diverse scenarios


// HARDCODED CONSTANTS FOR NOW. can let the program take input to allow for runtime initialization
const int autotuning_num_threads = 256;  // fixed thread count per block for all configurations
const int BM = 256; // BM - rows per block tile
const int BN = 128; // BN - columns per block tile
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void __launch_bounds__(autotuning_num_threads) sgemm_autotuning_warptiles(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {

  const int block_row_idx = blockIdx.y; // which block row in grid
  const int block_col_idx = blockIdx.x; // which block column in grid

  // calculations define the organizational structure that enables autotuning flexibility
  // WARPTILE DIMENSIONS: the "quantum" of work organization
  // the factor of 16 is hardcoded based on optimal thread arrangement for 256-thread blocks
  // 256 threads arranged as 16×16 grid provides good balance of coordination and parallelism
  constexpr int warp_tile_rows = TM * 16; // warptile height: typically TM * 16
  constexpr int warp_tile_cols = TN * 16; // warptile width: typically TN * 16
  
  // how many warptiles fit in the block
  // this determines how many separate tiles each thread will compute examples
  // BM=128, warp_tile_rows=128 → warp_tile_iterations_m = 1 (single tile per thread)
  // BM=256, warp_tile_rows=128 → warp_tile_iterations_m = 2 (two tiles per thread)
  constexpr int warp_tile_iterations_m = CEIL_DIV(BM, warp_tile_rows);  // vertical warptiles
  constexpr int warp_tile_iterations_n = CEIL_DIV(BN, warp_tile_cols);  // horizontal warptiles

  // this creates a consistent 16×16 thread organization pattern regardless of TM/TN values
  // the key insight: thread organization stays constant while work distribution varies
  const int thread_col_in_warptile = threadIdx.x % (warp_tile_cols / TN);  // 0-15: thread column within warptile
  const int thread_row_in_warptile = threadIdx.x / (warp_tile_cols / TN);  // 0-15: thread row within warptile

  // fits exactly with 256 threads per block (16 × 16 = 256)
  // aligns well with warp execution (256 threads = 8 warps of 32 threads each)
  // provides good balance for memory coalescing and work distribution
  // hardcoded choice that simplifies implementation across parameter space
  // maintains optimizations from previous kernels: transposed A, vectorized access
  __shared__ float shared_a_tile_transposed[BM * BK]; // A matrix in column-major layout
  __shared__ float shared_b_tile[BK * BN]; // B matrix in row-major layout
  A += block_row_idx * BM * K; // move to current block's A data
  B += block_col_idx * BN; // move to current block's B data  
  C += block_row_idx * BM * N + block_col_idx * BN; // move to current block's C data
  // these calculations ensure vectorized loads work correctly across parameter variations
  const int a_load_row = threadIdx.x / (BK / 4); // which row in A to load (0 to BM-1)
  const int a_load_col_group = threadIdx.x % (BK / 4); // which group of 4 columns (0 to BK/4-1)
  constexpr int a_load_stride = (autotuning_num_threads * 4) / BK;  // stride between loading iterations
  
  const int b_load_row = threadIdx.x / (BN / 4); // which row in B to load (0 to BK-1)
  const int b_load_col_group = threadIdx.x % (BN / 4); // which group of 4 columns (0 to BN/4-1)
  constexpr int b_load_stride = autotuning_num_threads / (BN / 4);  // stride between loading iterations
  // this is where the autotuning flexibility becomes visible in memory allocation
  // each thread must allocate space for ALL tiles it will compute across all warptile iterations
  //   standard config (BM=BN=128, TM=TN=8): 1×1×8×8 = 64 elements
  //   larger blocks (BM=256, BN=128, TM=TN=8): 2×1×8×8 = 128 elements  
  //   different tiles (BM=BN=128, TM=TN=4): 1×1×4×4 = 16 elements
  float thread_output_tiles[warp_tile_iterations_m * warp_tile_iterations_n * TM * TN] = {0.0};
  
  float reg_a_column_slice[TM] = {0.0};           // register cache for A values (column slice)
  float reg_b_row_slice[TN] = {0.0};              // register cache for B values (row slice)

  // processes matrix multiplication in chunks along the K dimension
  for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {

    // the stride-based loading enables correctness across different block dimensions
    // this loop structure automatically adapts to parameter combinations where
    // thread count and block dimensions don't align perfectly
    for (int load_offset = 0; load_offset + a_load_stride <= BM; load_offset += a_load_stride) {
      float4 a_vector = reinterpret_cast<float4 *>(
          &A[(a_load_row + load_offset) * K + a_load_col_group * 4])[0];
      
      // store in transposed layout for efficient register loading during computation
      shared_a_tile_transposed[(a_load_col_group * 4 + 0) * BM + a_load_row + load_offset] = a_vector.x;
      shared_a_tile_transposed[(a_load_col_group * 4 + 1) * BM + a_load_row + load_offset] = a_vector.y;
      shared_a_tile_transposed[(a_load_col_group * 4 + 2) * BM + a_load_row + load_offset] = a_vector.z;
      shared_a_tile_transposed[(a_load_col_group * 4 + 3) * BM + a_load_row + load_offset] = a_vector.w;
    }

    // similar stride-based approach ensures correctness across parameter variations
    for (int load_offset = 0; load_offset + b_load_stride <= BK; load_offset += b_load_stride) {
      reinterpret_cast<float4 *>(
          &shared_b_tile[(b_load_row + load_offset) * BN + b_load_col_group * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(b_load_row + load_offset) * N + b_load_col_group * 4])[0];
    }
    
    __syncthreads(); // ensure all loading completes before computation

    // these nested loops enable each thread to compute multiple output tiles
    // the number of iterations scales automatically with template parameters
    //   standard config: warp_tile_iterations_m=1, warp_tile_iterations_n=1 → single iteration
    //   larger blocks: warp_tile_iterations_m=2, warp_tile_iterations_n=1 → two vertical iterations
    //   different ratios: warp_tile_iterations_m=1, warp_tile_iterations_n=2 → two horizontal iterations
    
    for (int warp_tile_m = 0; warp_tile_m < warp_tile_iterations_m; ++warp_tile_m) {
      for (int warp_tile_n = 0; warp_tile_n < warp_tile_iterations_n; ++warp_tile_n) {

        // each iteration of these loops computes one TM×TN tile
        // the warptile indices (warp_tile_m, warp_tile_n) determine which tile
        for (int dot_product_idx = 0; dot_product_idx < BK; ++dot_product_idx) {
        
          // the key insight: (warp_tile_m * warp_tile_rows) shifts the memory access
          // to different regions for different warptile iterations
          //   warp_tile_m=0: loads from rows [thread_row_in_warptile*TM + 0...TM-1]
          //   warp_tile_m=1: loads from rows [warp_tile_rows + thread_row_in_warptile*TM + 0...TM-1]
          for (int i = 0; i < TM; ++i) {
            reg_a_column_slice[i] = shared_a_tile_transposed[
                dot_product_idx * BM + 
                (warp_tile_m * warp_tile_rows) + 
                thread_row_in_warptile * TM + i
            ];
          }
          
          // === LOAD B VALUES FOR CURRENT WARPTILE POSITION ===
          // similar shifting logic for horizontal warptile positioning
          for (int i = 0; i < TN; ++i) {
            reg_b_row_slice[i] = shared_b_tile[
                dot_product_idx * BN + 
                (warp_tile_n * warp_tile_cols) + 
                thread_col_in_warptile * TN + i
            ];
          }
          
          // === COMPUTE OUTER PRODUCT AND ACCUMULATE ===
          // accumulate into the appropriate section of thread_output_tiles
          // the indexing ensures results from different warptile positions
          // are stored in separate regions of the results array
          for (int result_row = 0; result_row < TM; ++result_row) {
            for (int result_col = 0; result_col < TN; ++result_col) {
              // RESULT STORAGE LAYOUT:
              // results are stored in row-major order across warptile iterations
              // [warptile_m=0,warptile_n=0][warptile_m=0,warptile_n=1]...[warptile_m=1,warptile_n=0]...
              int result_index = (warp_tile_m * TM + result_row) * (warp_tile_iterations_n * TN) +
                                warp_tile_n * TN + result_col;
              thread_output_tiles[result_index] += 
                  reg_a_column_slice[result_row] * reg_b_row_slice[result_col];
            }
          }
        }
      }
    }
    
    __syncthreads(); // ensure all computation completes before next iteration
    
    // advance to next K-dimension block
    A += BK;           // move BK columns right in A
    B += BK * N;       // move BK rows down in B
  }

  // the output writing mirrors the computation structure: iterate over warptile positions
  // and write the corresponding results to the appropriate global memory locations
  for (int warp_tile_m = 0; warp_tile_m < warp_tile_iterations_m; ++warp_tile_m) {
    for (int warp_tile_n = 0; warp_tile_n < warp_tile_iterations_n; ++warp_tile_n) {
      
      // calculate base address for current warptile position in global C matrix
      float *c_warptile_base = C + (warp_tile_m * warp_tile_rows * N) + (warp_tile_n * warp_tile_cols);
      
      // write out TM×TN results using vectorized stores
      for (int result_row = 0; result_row < TM; result_row += 1) {
        for (int result_col = 0; result_col < TN; result_col += 4) {
          
          // vectorized read-modify-write for optimal memory bandwidth
          float4 c_vector = reinterpret_cast<float4 *>(
              &c_warptile_base[(thread_row_in_warptile * TM + result_row) * N + 
                              thread_col_in_warptile * TN + result_col])[0];
          
          // calculate index into thread_output_tiles for current warptile position
          const int result_base_index = (warp_tile_m * TM + result_row) * (warp_tile_iterations_n * TN) + 
                                       warp_tile_n * TN + result_col;
          
          // perform GEMM update: C = alpha * A*B + beta * C
          c_vector.x = alpha * thread_output_tiles[result_base_index + 0] + beta * c_vector.x;
          c_vector.y = alpha * thread_output_tiles[result_base_index + 1] + beta * c_vector.y;
          c_vector.z = alpha * thread_output_tiles[result_base_index + 2] + beta * c_vector.z;
          c_vector.w = alpha * thread_output_tiles[result_base_index + 3] + beta * c_vector.w;
          
          // vectorized write back
          reinterpret_cast<float4 *>(
              &c_warptile_base[(thread_row_in_warptile * TM + result_row) * N + 
                              thread_col_in_warptile * TN + result_col])[0] = c_vector;
        }
      }
    }
  }
}

// this kernel represents the evolution from fixed optimization strategies to flexible
// parameter exploration frameworks. instead of optimizing for one specific configuration,
// it creates a robust organizational structure that works correctly across hundreds
// of parameter combinations.
// by separating thread organization (always 16×16) from work distribution (variable
// based on template parameters), the framework enables systematic exploration of
// different performance trade-offs while maintaining implementation correctness.
// enables automatic discovery of optimal parameters for different:
// - hardware architectures (A6000 vs A100 prefer different configurations)
// - problem sizes (different matrix dimensions may favor different tile sizes)
// - optimization objectives (throughput vs latency vs memory usage)
// the hardcoded factor of 16 in warptile calculations represents a design choice
// that balances implementation complexity with parameter space coverage. while it
// constrains some possible configurations, it ensures stability and correctness
// across the parameter combinations that matter most for practical performance.

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
  // need BM >= warp_tile_rows (which you satisfied: 128 >= 128)
  // BN >= warp_tile_cols (which you violated: 64 < 128)

  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  dim3 blockDim(256, 1, 1);                            // 256 threads per block
  const int num_iterations = 10;
  
  // benchmark loop  
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i){
    // Reset C to original values before each iteration
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    sgemm_autotuning_warptiles<<<gridDim, blockDim>>>(M,N,K, alpha, d_A, d_B, beta, d_C); // CORRECT
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