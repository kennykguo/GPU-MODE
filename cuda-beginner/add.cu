#include <iostream>
#include <math.h>
 
// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
 int N = 1<<20; // 1M elements
 
  float *x, *y, *sum;
  cudaMallocManaged(&x, N*sizeof(float)); // allocate on unified emmory
  cudaMallocManaged(&y, N*sizeof(float));
  // prefetch the x and y arrays to the GPU
  cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
 
  add <<<1, 1>>>(N, x, y);
  // wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;


  // free memory
  cudaFree(x);
  cudaFree(y);
 
 return 0;
}