/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved. */

#include <stdio.h>
#include <assert.h>

// cuda runtime error checking wrapper
inline cudaError_t check_cuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// thread block configuration constants
const int TILE_DIM = 32;   // shared memory tile dimension
const int BLOCK_ROWS = 8;  // threads per block in y-direction
const int NUM_REPS = 100;  // benchmark repetitions

// performance validation and bandwidth calculation
void postprocess(const float *ref, const float *res, int n, float ms) {
  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// optimal matrix transpose kernel with bank conflict avoidance
__global__ void transpose_no_bank_conflicts(float *output_data, const float *input_data) {
  // shared memory tile with padding to avoid bank conflicts
  // +1 padding ensures different rows map to different banks
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
  
  // calculate global thread coordinates for input matrix
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  // load input data into shared memory tile
  // each thread loads TILE_DIM/BLOCK_ROWS elements vertically
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    // row-major indexing: tile[local_y][local_x] = input[(global_y)*width + global_x]
    tile[threadIdx.y+j][threadIdx.x] = input_data[(y+j)*width + x];
  }

  // synchronize all threads in block before proceeding to write phase
  __syncthreads();

  // transpose the block coordinates for output
  // swap blockIdx.x and blockIdx.y to achieve transpose
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  // write transposed data from shared memory to output
  // access pattern: tile[local_x][local_y] to achieve transpose
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    output_data[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

int main(int argc, char **argv) {
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx * ny * sizeof(float);
  dim3 dim_grid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dim_block(TILE_DIM, BLOCK_ROWS, 1);

  int dev_id = 0;

  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, dev_id));
  printf("\ndevice : %s\n", prop.name);
  printf("matrix size: %d %d, block size: %d %d, tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dim_grid.x, dim_grid.y, dim_grid.z, dim_block.x, dim_block.y, dim_block.z);
  check_cuda(cudaSetDevice(dev_id));

  // host memory allocation
  float *h_input_data = (float*)malloc(mem_size);
  float *h_output_data = (float*)malloc(mem_size);
  float *gold = (float*)malloc(mem_size);
  
  // device memory allocation
  float *d_input_data, *d_output_data;
  check_cuda(cudaMalloc(&d_input_data, mem_size));
  check_cuda(cudaMalloc(&d_output_data, mem_size));
    
  // initialize input data
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_input_data[j*nx + i] = j*nx + i;

  // generate reference result for validation
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_input_data[i*nx + j];
  
  // copy input data to device
  check_cuda(cudaMemcpy(d_input_data, h_input_data, mem_size, cudaMemcpyHostToDevice));
  
  // timing events
  cudaEvent_t start_event, stop_event;
  check_cuda(cudaEventCreate(&start_event));
  check_cuda(cudaEventCreate(&stop_event));
  float ms;

  // benchmark optimal transpose kernel
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
  printf("%25s", "conflict-free transpose");
  check_cuda(cudaMemset(d_output_data, 0, mem_size));
  
  // warmup run
  transpose_no_bank_conflicts<<<dim_grid, dim_block>>>(d_output_data, d_input_data);
  
  // timed execution
  check_cuda(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; i++)
    transpose_no_bank_conflicts<<<dim_grid, dim_block>>>(d_output_data, d_input_data);
  check_cuda(cudaEventRecord(stop_event, 0));
  check_cuda(cudaEventSynchronize(stop_event));
  check_cuda(cudaEventElapsedTime(&ms, start_event, stop_event));
  
  // copy result back and validate
  check_cuda(cudaMemcpy(h_output_data, d_output_data, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_output_data, nx * ny, ms);
  
  // cleanup resources
  check_cuda(cudaEventDestroy(start_event));
  check_cuda(cudaEventDestroy(stop_event));
  check_cuda(cudaFree(d_output_data));
  check_cuda(cudaFree(d_input_data));
  free(h_input_data);
  free(h_output_data);
  free(gold);
}