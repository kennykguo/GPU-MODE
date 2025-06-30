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
const int num_threads_per_block = 128;
const int WARPSIZE = 32;

// this kernel represents an attempt to build upon the warptiling optimization (kernel #10)
// by introducing double buffering to theoretically overlap memory transfers with computation.
// warptiling kernel: introduced warp-level coordination for better register cache locality
// this kernel: adds double buffering strategy attempting to overlap loading and processing
// the core idea is to split threads into two groups that work on different phases:
// - group 0: primarily processes current data while occasionally loading future data
// - group 1: primarily loads future data while occasionally processing current data
// 1. doubled shared memory allocation (2x memory usage)
// 2. thread role specialization (splits threads into two groups)
// 3. complex control flow coordination between groups
// 4. pipelined execution attempting to overlap different operation types
// by having some threads load next chunk while others compute current chunk,
// the kernel attempts to hide memory transfer latency behind computation time.
// however, both groups still perform all computational work, so benefits are unclear.
// benefits: potential overlap of memory and compute operations
// costs: doubled memory usage, complex control flow, more synchronization points
// during development analysis, we questioned whether this approach actually provides
// performance benefits since both thread groups perform the same computational work,
// effectively doubling some operations while using the same hardware resources.

namespace double_buffering {

// VECTORIZED GLOBAL MEMORY LOADING WITH DOUBLE BUFFER SUPPORT
// this function maintains the optimizations from previous kernels (vectorization, transpose)
// while adding support for loading into specific buffer regions of doubled shared memory
template <const int block_tile_rows, const int block_tile_cols, const int block_tile_k_dim, const int a_loading_row_stride, const int b_loading_row_stride>
__device__ void load_chunk_from_global_memory(
    const int matrix_n_cols, const int matrix_k_cols, 
    float *global_a_matrix, float *global_b_matrix,
    float *shared_a_buffer, float *shared_b_buffer, 
    const int a_load_row_start, const int a_load_col_group,
    const int b_load_row_start, const int b_load_col_group) {
  
  // A MATRIX VECTORIZED LOADING WITH TRANSPOSE
  // maintains the transpose-during-load optimization for efficient shared memory access
  // loads 4 consecutive elements per thread using float4 for memory bandwidth efficiency
  for (uint row_offset = 0; row_offset + a_loading_row_stride <= block_tile_rows; row_offset += a_loading_row_stride) {
    
    // vectorized load: 4 consecutive floats (128-bit memory transaction)
    float4 a_vector_data = reinterpret_cast<float4 *>(
        &global_a_matrix[(a_load_row_start + row_offset) * matrix_k_cols + a_load_col_group * 4])[0];
    
    // transpose a matrix during storage for stride-1 access during computation
    // original layout: [row][col] → transposed layout: [col][row]
    shared_a_buffer[(a_load_col_group * 4 + 0) * block_tile_rows + a_load_row_start + row_offset] = a_vector_data.x;
    shared_a_buffer[(a_load_col_group * 4 + 1) * block_tile_rows + a_load_row_start + row_offset] = a_vector_data.y;
    shared_a_buffer[(a_load_col_group * 4 + 2) * block_tile_rows + a_load_row_start + row_offset] = a_vector_data.z;
    shared_a_buffer[(a_load_col_group * 4 + 3) * block_tile_rows + a_load_row_start + row_offset] = a_vector_data.w;
  }
  
  // B MATRIX VECTORIZED LOADING (ROW-MAJOR)
  // maintains row-major layout for efficient warp-coordinated access patterns
  // loads 4 consecutive elements per thread using float4 for memory bandwidth efficiency
  for (uint row_offset = 0; row_offset + b_loading_row_stride <= block_tile_k_dim; row_offset += b_loading_row_stride) {
    
    // direct vectorized copy: global memory → shared memory (no transpose needed)
    reinterpret_cast<float4 *>(
        &shared_b_buffer[(b_load_row_start + row_offset) * block_tile_cols + b_load_col_group * 4])[0] =
        reinterpret_cast<float4 *>(
            &global_b_matrix[(b_load_row_start + row_offset) * matrix_n_cols + b_load_col_group * 4])[0];
  }
}

// WARP-COORDINATED COMPUTATION FROM SHARED MEMORY
// this function is identical to the warptiling kernel - maintains all warp-level optimizations
// processes data from shared memory using the sophisticated warp coordination patterns
template <const int block_tile_rows, const int block_tile_cols, const int block_tile_k_dim, 
          const int warp_tile_rows, const int warp_tile_cols,
          const int warp_subtile_iterations_m, const int warp_subtile_iterations_n, 
          const int warp_subtile_rows, const int warp_subtile_cols,
          const int thread_tile_rows, const int thread_tile_cols>
__device__ void process_chunk_from_shared_memory(
    float *register_a_cache, float *register_b_cache, float *thread_output_accumulator, 
    const float *shared_a_buffer, const float *shared_b_buffer, 
    const uint warp_row_in_block, const uint warp_col_in_block,
    const uint thread_row_in_warp, const uint thread_col_in_warp) {
  
  // MAIN K-DIMENSION COMPUTATION LOOP
  // processes each k-slice of the current block tile using warp coordination
  for (uint k_slice_idx = 0; k_slice_idx < block_tile_k_dim; ++k_slice_idx) {
    
    // PHASE 1: COORDINATED DATA LOADING INTO REGISTERS
    // all 32 threads in warp coordinate to load their portions of warp's territory
    
    // load a matrix data for all subtile rows this thread will process
    // each thread loads its portion across all subtiles in the warp's row territory
    for (uint warp_subtile_row_idx = 0; warp_subtile_row_idx < warp_subtile_iterations_m; ++warp_subtile_row_idx) {
      for (uint thread_row_element = 0; thread_row_element < thread_tile_rows; ++thread_row_element) {
        
        // complex addressing: k-slice + warp territory + subtile + thread position
        register_a_cache[warp_subtile_row_idx * thread_tile_rows + thread_row_element] =
            shared_a_buffer[(k_slice_idx * block_tile_rows) + 
                            warp_row_in_block * warp_tile_rows + 
                            warp_subtile_row_idx * warp_subtile_rows +
                            thread_row_in_warp * thread_tile_rows + thread_row_element];
      }
    }
    
    // load b matrix data for all subtile columns this thread will process  
    // each thread loads its portion across all subtiles in the warp's column territory
    for (uint warp_subtile_col_idx = 0; warp_subtile_col_idx < warp_subtile_iterations_n; ++warp_subtile_col_idx) {
      for (uint thread_col_element = 0; thread_col_element < thread_tile_cols; ++thread_col_element) {
        
        // complex addressing: k-slice + warp territory + subtile + thread position
        register_b_cache[warp_subtile_col_idx * thread_tile_cols + thread_col_element] =
            shared_b_buffer[(k_slice_idx * block_tile_cols) + 
                            warp_col_in_block * warp_tile_cols + 
                            warp_subtile_col_idx * warp_subtile_cols +
                            thread_col_in_warp * thread_tile_cols + thread_col_element];
      }
    }
    
    // PHASE 2: COORDINATED OUTER PRODUCT COMPUTATION
    // all threads compute their contribution to warp's collective matrix multiplication
    // this nested loop structure enables register cache locality and instruction-level parallelism
    
    for (uint warp_subtile_row_idx = 0; warp_subtile_row_idx < warp_subtile_iterations_m; ++warp_subtile_row_idx) {
      for (uint warp_subtile_col_idx = 0; warp_subtile_col_idx < warp_subtile_iterations_n; ++warp_subtile_col_idx) {
        
        // compute outer product between thread's a-data and b-data for current subtile
        for (uint thread_result_row = 0; thread_result_row < thread_tile_rows; ++thread_result_row) {
          for (uint thread_result_col = 0; thread_result_col < thread_tile_cols; ++thread_result_col) {
            
            // accumulate into thread's result storage with proper indexing across all subtiles
            // result storage is organized as: [subtile_row][thread_row][subtile_col][thread_col]
            thread_output_accumulator[(warp_subtile_row_idx * thread_tile_rows + thread_result_row) * 
                                      (warp_subtile_iterations_n * thread_tile_cols) +
                                      (warp_subtile_col_idx * thread_tile_cols) + thread_result_col] +=
                register_a_cache[warp_subtile_row_idx * thread_tile_rows + thread_result_row] *
                register_b_cache[warp_subtile_col_idx * thread_tile_cols + thread_result_col];
          }
        }
      }
    }
  }
}

} // namespace double_buffering

// MAIN DOUBLE BUFFERING SGEMM KERNEL
// this kernel builds upon warptiling by attempting to overlap memory operations with computation
// through a dual-buffer strategy that splits threads into specialized groups
template <const int block_tile_rows, const int block_tile_cols, const int block_tile_k_dim, 
          const int warp_tile_rows, const int warp_tile_cols,
          const int warp_subtile_iterations_n, const int thread_tile_rows, const int thread_tile_cols, 
          const int total_threads_per_block>
__global__ void __launch_bounds__(total_threads_per_block) sgemm_double_buffering_experimental(
        const int matrix_m_rows, const int matrix_n_cols, const int matrix_k_cols,
        const float gemm_alpha_scalar, float *global_a_matrix, float *global_b_matrix, 
        const float gemm_beta_scalar, float *global_c_matrix) {
  
  // BLOCK POSITIONING IN GRID
  const uint block_row_idx = blockIdx.y;  // which block row in computational grid
  const uint block_col_idx = blockIdx.x;  // which block column in computational grid
  
  // WARP IDENTIFICATION AND POSITIONING
  // maintains the warptiling warp coordination structure from kernel #10
  const uint warp_id_in_block = threadIdx.x / WARPSIZE;  // which warp (0-7 for 256 threads)
  const uint warp_col_in_block = warp_id_in_block % (block_tile_cols / warp_tile_cols);  // warp's column position
  const uint warp_row_in_block = warp_id_in_block / (block_tile_cols / warp_tile_cols);  // warp's row position
  
  // WARP SUBTILE ORGANIZATION CALCULATIONS
  // these maintain the sophisticated subtile math from warptiling for coordinated computation
  constexpr uint warp_subtile_iterations_m = (warp_tile_rows * warp_tile_cols) / 
                                             (WARPSIZE * thread_tile_rows * thread_tile_cols * warp_subtile_iterations_n);
  constexpr uint warp_subtile_rows = warp_tile_rows / warp_subtile_iterations_m;
  constexpr uint warp_subtile_cols = warp_tile_cols / warp_subtile_iterations_n;
  
  // THREAD POSITIONING WITHIN WARP
  // maintains the warptiling thread coordination patterns for optimal register access
  const uint thread_id_in_warp = threadIdx.x % WARPSIZE;                                      // [0-31] position within warp
  const uint thread_col_in_warp = thread_id_in_warp % (warp_subtile_cols / thread_tile_cols); // column position in subtile
  const uint thread_row_in_warp = thread_id_in_warp / (warp_subtile_cols / thread_tile_cols); // row position in subtile
  
  // DOUBLE BUFFERING SHARED MEMORY ALLOCATION
  // key innovation: doubled shared memory to enable dual-buffer strategy
  // this is the primary architectural change from the warptiling kernel
  //
  // MEMORY LAYOUT:
  // buffer 0: shared_a_buffers[0 to (block_tile_rows * block_tile_k_dim - 1)]
  // buffer 1: shared_a_buffers[block_tile_rows * block_tile_k_dim to (2 * block_tile_rows * block_tile_k_dim - 1)]
  __shared__ float shared_a_buffers[2 * block_tile_rows * block_tile_k_dim];   // doubled A matrix storage
  __shared__ float shared_b_buffers[2 * block_tile_k_dim * block_tile_cols];   // doubled B matrix storage
  
  // THREAD GROUP SPECIALIZATION
  // critical design decision: split threads into two groups based on thread id
  // this creates the foundation for the double buffering coordination strategy
  //
  // group 0 (first half): primarily processes data, occasionally loads future data
  // group 1 (second half): primarily loads future data, occasionally processes data
  bool thread_group_id = threadIdx.x >= (total_threads_per_block / 2);
  
  // GLOBAL MEMORY POINTER ADVANCEMENT
  // advance matrix pointers to this block's computational territory
  global_a_matrix += block_row_idx * block_tile_rows * matrix_k_cols;  // move to block's row region
  global_b_matrix += block_col_idx * block_tile_cols;                  // move to block's column region
  
  // advance output pointer to this warp's specific territory within the block
  // this optimization reduces redundant address calculations across threads in same warp
  global_c_matrix += (block_row_idx * block_tile_rows + warp_row_in_block * warp_tile_rows) * matrix_n_cols + 
                     block_col_idx * block_tile_cols + warp_col_in_block * warp_tile_cols;
  
  // VECTORIZED LOADING INDEX CALCULATIONS
  // these calculations enable float4 vectorized loads while working with the dual-group strategy
  // note: calculations use (total_threads_per_block / 2) to reflect that loading is split between groups
  //
  // IMPORTANT DESIGN DECISION: loading calculations assume only half the threads participate
  // in each loading operation, reflecting the group specialization strategy
  const uint a_load_row_start = (threadIdx.x % (total_threads_per_block / 2)) / (block_tile_k_dim / 4);
  const uint a_load_col_group = (threadIdx.x % (total_threads_per_block / 2)) % (block_tile_k_dim / 4);
  constexpr uint a_loading_row_stride = ((total_threads_per_block / 2) * 4) / block_tile_k_dim;
  
  const uint b_load_row_start = (threadIdx.x % (total_threads_per_block / 2)) / (block_tile_cols / 4);
  const uint b_load_col_group = (threadIdx.x % (total_threads_per_block / 2)) % (block_tile_cols / 4);
  constexpr uint b_loading_row_stride = (total_threads_per_block / 2) / (block_tile_cols / 4);
  
  // REGISTER ALLOCATION FOR WARP COORDINATION
  // maintains the warptiling register organization for coordinated computation
  // each thread stores results for multiple subtiles it processes in coordination with warp
  float thread_output_accumulator[warp_subtile_iterations_m * thread_tile_rows * 
                                  warp_subtile_iterations_n * thread_tile_cols] = {0.0};
  
  // register caches for warp-level coordination (maintains warptiling optimization)
  float register_a_cache[warp_subtile_iterations_m * thread_tile_rows] = {0.0};
  float register_b_cache[warp_subtile_iterations_n * thread_tile_cols] = {0.0};
  
  // INITIAL BUFFER LOADING
  // group 0 loads the very first chunk to establish the pipeline
  // group 1 waits, creating the initial condition for the double buffering pattern
  if (thread_group_id == 0) {
    double_buffering::load_chunk_from_global_memory<
        block_tile_rows, block_tile_cols, block_tile_k_dim, a_loading_row_stride, b_loading_row_stride>(
        matrix_n_cols, matrix_k_cols, global_a_matrix, global_b_matrix, 
        shared_a_buffers, shared_b_buffers,  // load into buffer 0
        a_load_row_start, a_load_col_group, b_load_row_start, b_load_col_group);
  }
  __syncthreads();  // ensure initial loading completes before main pipeline begins
  
  // MAIN DOUBLE BUFFERING PIPELINE
  // this is where the complex coordination between groups attempts to create overlap
  // each iteration processes 2 * block_tile_k_dim worth of k-dimension
  // while one group processes current data from one buffer,
  // the other group loads future data into the alternate buffer
  // both groups end up processing all chunks through their respective processFromSmem calls,
  // which means computational work is effectively duplicated rather than divided
  for (uint k_block_start = 0; k_block_start < matrix_k_cols; k_block_start += 2 * block_tile_k_dim) {
    
    if (thread_group_id == 0) {
      // step 1: process current chunk from buffer 0
      double_buffering::process_chunk_from_shared_memory<
          block_tile_rows, block_tile_cols, block_tile_k_dim, warp_tile_rows, warp_tile_cols,
          warp_subtile_iterations_m, warp_subtile_iterations_n, warp_subtile_rows, warp_subtile_cols,
          thread_tile_rows, thread_tile_cols>(
          register_a_cache, register_b_cache, thread_output_accumulator, 
          shared_a_buffers, shared_b_buffers,  // process from buffer 0
          warp_row_in_block, warp_col_in_block, thread_row_in_warp, thread_col_in_warp);
      __syncthreads();
      
      // step 2: process next chunk from buffer 1 (if available)
      if (k_block_start + block_tile_k_dim < matrix_k_cols) {
        double_buffering::process_chunk_from_shared_memory<
            block_tile_rows, block_tile_cols, block_tile_k_dim, warp_tile_rows, warp_tile_cols,
            warp_subtile_iterations_m, warp_subtile_iterations_n, warp_subtile_rows, warp_subtile_cols,
            thread_tile_rows, thread_tile_cols>(
            register_a_cache, register_b_cache, thread_output_accumulator, 
            shared_a_buffers + (block_tile_rows * block_tile_k_dim),    // process from buffer 1
            shared_b_buffers + (block_tile_k_dim * block_tile_cols),
            warp_row_in_block, warp_col_in_block, thread_row_in_warp, thread_col_in_warp);
      }
      __syncthreads();
      
      // step 3: load future chunk into buffer 0 for next iteration
      if (k_block_start + 2 * block_tile_k_dim < matrix_k_cols) {
        double_buffering::load_chunk_from_global_memory<
            block_tile_rows, block_tile_cols, block_tile_k_dim, a_loading_row_stride, b_loading_row_stride>(
            matrix_n_cols, matrix_k_cols, 
            global_a_matrix + 2 * block_tile_k_dim, global_b_matrix + 2 * block_tile_k_dim * matrix_n_cols,
            shared_a_buffers, shared_b_buffers,  // load into buffer 0
            a_load_row_start, a_load_col_group, b_load_row_start, b_load_col_group);
      }
      
    } else {
      // step 1: load next chunk into buffer 1 while group 0 processes buffer 0
      if (k_block_start + block_tile_k_dim < matrix_k_cols) {
        double_buffering::load_chunk_from_global_memory<
            block_tile_rows, block_tile_cols, block_tile_k_dim, a_loading_row_stride, b_loading_row_stride>(
            matrix_n_cols, matrix_k_cols, 
            global_a_matrix + block_tile_k_dim, global_b_matrix + block_tile_k_dim * matrix_n_cols,
            shared_a_buffers + (block_tile_rows * block_tile_k_dim),    // load into buffer 1
            shared_b_buffers + (block_tile_k_dim * block_tile_cols),
            a_load_row_start, a_load_col_group, b_load_row_start, b_load_col_group);
      }
      __syncthreads();
      
      // step 2: process current chunk from buffer 0 (same as group 0)
      double_buffering::process_chunk_from_shared_memory<
          block_tile_rows, block_tile_cols, block_tile_k_dim, warp_tile_rows, warp_tile_cols,
          warp_subtile_iterations_m, warp_subtile_iterations_n, warp_subtile_rows, warp_subtile_cols,
          thread_tile_rows, thread_tile_cols>(
          register_a_cache, register_b_cache, thread_output_accumulator, 
          shared_a_buffers, shared_b_buffers,  // process from buffer 0
          warp_row_in_block, warp_col_in_block, thread_row_in_warp, thread_col_in_warp);
      __syncthreads();
      
      // step 3: process next chunk from buffer 1 (same as group 0)
      if (k_block_start + block_tile_k_dim < matrix_k_cols) {
        double_buffering::process_chunk_from_shared_memory<
            block_tile_rows, block_tile_cols, block_tile_k_dim, warp_tile_rows, warp_tile_cols,
            warp_subtile_iterations_m, warp_subtile_iterations_n, warp_subtile_rows, warp_subtile_cols,
            thread_tile_rows, thread_tile_cols>(
            register_a_cache, register_b_cache, thread_output_accumulator, 
            shared_a_buffers + (block_tile_rows * block_tile_k_dim),    // process from buffer 1
            shared_b_buffers + (block_tile_k_dim * block_tile_cols),
            warp_row_in_block, warp_col_in_block, thread_row_in_warp, thread_col_in_warp);
      }
    }
    
    // advance to next set of k-dimension blocks (note: 2 * block_tile_k_dim advance)
    global_a_matrix += 2 * block_tile_k_dim;                    // advance 2 k-blocks in A matrix
    global_b_matrix += 2 * block_tile_k_dim * matrix_n_cols;    // advance 2 k-blocks in B matrix
    __syncthreads();  // ensure all groups complete before next iteration
  }
  
  // maintains the sophisticated output writing from warptiling with vectorized stores
  // each thread writes its accumulated results to the appropriate global memory locations
  // threads iterate through their accumulated subtiles and write each one to global memory
  // using vectorized float4 stores for optimal memory bandwidth utilization
  for (uint warp_subtile_row_idx = 0; warp_subtile_row_idx < warp_subtile_iterations_m; ++warp_subtile_row_idx) {
    for (uint warp_subtile_col_idx = 0; warp_subtile_col_idx < warp_subtile_iterations_n; ++warp_subtile_col_idx) {
      
      // calculate base pointer for current subtile's output region
      float *subtile_output_base = global_c_matrix + 
                                   (warp_subtile_row_idx * warp_subtile_rows) * matrix_n_cols + 
                                   warp_subtile_col_idx * warp_subtile_cols;
      
      // write thread's contribution to current subtile using vectorized stores
      for (uint thread_result_row = 0; thread_result_row < thread_tile_rows; thread_result_row += 1) {
        for (uint thread_result_col = 0; thread_result_col < thread_tile_cols; thread_result_col += 4) {
          
          // vectorized read-modify-write: load existing C values, update with GEMM result, store back
          float4 c_vector_values = reinterpret_cast<float4 *>(
              &subtile_output_base[(thread_row_in_warp * thread_tile_rows + thread_result_row) * matrix_n_cols +
                                   thread_col_in_warp * thread_tile_cols + thread_result_col])[0];
          
          // calculate index into thread's accumulated results
          const int result_base_index = (warp_subtile_row_idx * thread_tile_rows + thread_result_row) * 
                                       (warp_subtile_iterations_n * thread_tile_cols) +
                                       warp_subtile_col_idx * thread_tile_cols + thread_result_col;
          
          // perform GEMM update: C = alpha * (A * B) + beta * C
          c_vector_values.x = gemm_alpha_scalar * thread_output_accumulator[result_base_index + 0] + gemm_beta_scalar * c_vector_values.x;
          c_vector_values.y = gemm_alpha_scalar * thread_output_accumulator[result_base_index + 1] + gemm_beta_scalar * c_vector_values.y;
          c_vector_values.z = gemm_alpha_scalar * thread_output_accumulator[result_base_index + 2] + gemm_beta_scalar * c_vector_values.z;
          c_vector_values.w = gemm_alpha_scalar * thread_output_accumulator[result_base_index + 3] + gemm_beta_scalar * c_vector_values.w;
          
          // vectorized store back to global memory
          reinterpret_cast<float4 *>(
              &subtile_output_base[(thread_row_in_warp * thread_tile_rows + thread_result_row) * matrix_n_cols +
                                   thread_col_in_warp * thread_tile_cols + thread_result_col])[0] = c_vector_values;
        }
      }
    }
  }
}

// 1. maintains all warptiling optimizations (warp coordination, register cache locality)
// 2. adds double buffering strategy with doubled shared memory allocation
// 3. introduces thread group specialization for pipeline coordination
// 4. attempts to overlap memory transfers with computation operations
// benefits: potential overlap of memory bandwidth and compute utilization
// costs: doubled memory usage, complex control flow, additional synchronization overhead
// during development, we identified that both thread groups perform identical computational work
// through their respective processFromSmem calls, effectively duplicating computation rather
// than creating true parallelism between memory and compute operations.
// this raises questions about whether the double buffering strategy provides genuine
// performance benefits or if the increased complexity and memory usage offset any gains.
// this kernel demonstrates the importance of critically analyzing optimization strategies
// rather than assuming complexity equals performance. sophisticated techniques like double
// buffering require careful analysis to ensure they provide genuine benefits rather than
// just increased implementation complexity.
// the evolution from simple blocking to warptiling to double buffering shows how
// GPU optimization often involves exploring increasingly complex coordination strategies,
// but each additional layer of complexity must be justified by measurable performance gains.


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
  dim3 blockDim(num_threads_per_block, 1, 1);                            // 256 threads per block
  const int num_iterations = 10;
  
  // benchmark loop  
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < num_iterations; ++i){
    // Reset C to original values before each iteration
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    sgemm_double_buffering_experimental<64, 128, 8, 32, 64, 2, 4, 4, 128><<<gridDim, blockDim>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C);
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