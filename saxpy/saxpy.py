import numpy as np
import time
from typing import Callable, Tuple

# method 1: numpy saxpy (cpu baseline)
def saxpy_numpy(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """basic numpy implementation running on cpu"""
    return a * x + y

# method 2: cupy saxpy (gpu numpy-like)
try:
    import cupy as cp
    
    def saxpy_cupy(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """cupy implementation - numpy-like api for gpu"""
        # transfer to gpu
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        
        # compute on gpu
        result_gpu = a * x_gpu + y_gpu
        
        # transfer back to cpu
        return cp.asnumpy(result_gpu)
    
    cupy_available = True
except ImportError:
    cupy_available = False
    print("cupy not available")

# method 3: pycuda saxpy (low-level gpu access)
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    
    # cuda kernel source code
    cuda_kernel_source = """
    __global__ void saxpy_kernel(int n, float a, float *x, float *y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            y[i] = a * x[i] + y[i];
        }
    }
    """
    
    # compile kernel
    mod = SourceModule(cuda_kernel_source)
    saxpy_cuda_kernel = mod.get_function("saxpy_kernel")
    
    def saxpy_pycuda(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """pycuda implementation with custom kernel"""
        n = x.shape[0]
        
        # allocate gpu memory and copy data
        x_gpu = gpuarray.to_gpu(x.astype(np.float32))
        y_gpu = gpuarray.to_gpu(y.astype(np.float32))
        
        # calculate grid dimensions
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        # launch kernel
        saxpy_cuda_kernel(
            np.int32(n),
            np.float32(a),
            x_gpu.gpudata,
            y_gpu.gpudata,
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # copy result back to cpu
        return y_gpu.get()
    
    pycuda_available = True
except ImportError:
    pycuda_available = False
    print("pycuda not available")

# method 4: numba cuda saxpy
try:
    from numba import cuda, float32
    
    @cuda.jit
    def saxpy_numba_kernel(n, a, x, y):
        """numba cuda kernel"""
        i = cuda.grid(1)
        if i < n:
            y[i] = a * x[i] + y[i]
    
    def saxpy_numba(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """numba cuda implementation"""
        n = x.shape[0]
        
        # allocate gpu memory and copy data
        x_gpu = cuda.to_device(x.astype(np.float32))
        y_gpu = cuda.to_device(y.astype(np.float32))
        
        # calculate grid dimensions
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        # launch kernel
        saxpy_numba_kernel[blocks_per_grid, threads_per_block](n, a, x_gpu, y_gpu)
        
        # copy result back to cpu
        return y_gpu.copy_to_host()
    
    numba_available = True
except ImportError:
    numba_available = False
    print("numba not available")

# method 5: original copperhead-style implementation (conceptual)
def saxpy_copperhead_style(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """copperhead-style list comprehension (running on cpu)"""
    # note: original copperhead is no longer maintained
    # this shows the syntax style mentioned in the blog
    return np.array([a * xi + yi for xi, yi in zip(x, y)])

# timing utility
def time_function(func: Callable, name: str, *args) -> Tuple[float, np.ndarray]:
    """time a function and return duration and result"""
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    
    duration_ms = (end - start) * 1000
    print(f"{name}: {duration_ms:.2f} ms")
    return duration_ms, result

# verification function
def verify_result(result: np.ndarray, reference: np.ndarray, tolerance: float = 1e-5) -> bool:
    """verify result against reference"""
    return np.allclose(result, reference, rtol=tolerance, atol=tolerance)

def main():
    # test parameters
    n = 1 << 20  # 1m elements
    a = 2.0
    
    # generate test data
    np.random.seed(42)
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    
    print(f"testing saxpy with {n} elements")
    print("=" * 50)
    
    # reference implementation
    duration_ref, y_ref = time_function(saxpy_numpy, "numpy (cpu)", a, x, y)
    
    # test available implementations
    results = {}
    
    # copperhead-style (cpu)
    duration, result = time_function(saxpy_copperhead_style, "copperhead-style", a, x, y)
    results["copperhead-style"] = (duration, verify_result(result, y_ref))
    
    # cupy
    if cupy_available:
        duration, result = time_function(saxpy_cupy, "cupy", a, x, y)
        results["cupy"] = (duration, verify_result(result, y_ref))
    
    # pycuda
    if pycuda_available:
        duration, result = time_function(saxpy_pycuda, "pycuda", a, x, y)
        results["pycuda"] = (duration, verify_result(result, y_ref))
    
    # numba
    if numba_available:
        # warm up numba jit
        _ = saxpy_numba(a, x[:1000], y[:1000])
        duration, result = time_function(saxpy_numba, "numba cuda", a, x, y)
        results["numba cuda"] = (duration, verify_result(result, y_ref))
    
    # summary
    print("\n" + "=" * 50)
    print("summary:")
    print(f"{'method':<15} {'time (ms)':<12} {'speedup':<10} {'correct':<8}")
    print("-" * 50)
    print(f"{'numpy (cpu)':<15} {duration_ref:<12.2f} {'1.00x':<10} {'✓':<8}")
    
    for method, (duration, correct) in results.items():
        speedup = duration_ref / duration if duration > 0 else 0
        correct_str = "✓" if correct else "✗"
        print(f"{method:<15} {duration:<12.2f} {speedup:<10.2f}x {correct_str:<8}")

# additional utility functions for advanced usage
def compare_memory_usage():
    """compare memory usage of different implementations"""
    import psutil
    import os
    
    n = 1 << 18  # smaller size for memory comparison
    a = 2.0
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    
    process = psutil.Process(os.getpid())
    
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
    
    print(f"\nmemory usage comparison (n={n}):")
    print("-" * 40)
    
    # baseline
    mem_start = get_memory_mb()
    result_numpy = saxpy_numpy(a, x, y)
    mem_numpy = get_memory_mb() - mem_start
    print(f"numpy: {mem_numpy:.1f} mb")
    
    # cupy
    if cupy_available:
        mem_start = get_memory_mb()
        result_cupy = saxpy_cupy(a, x, y)
        mem_cupy = get_memory_mb() - mem_start
        print(f"cupy: {mem_cupy:.1f} mb")

def benchmark_sizes():
    """benchmark different array sizes"""
    sizes = [1 << i for i in range(16, 22)]  # 64k to 2m elements
    a = 2.0
    
    print(f"\nperformance scaling:")
    print("-" * 60)
    print(f"{'size':<10} {'numpy':<12} {'cupy':<12} {'speedup':<10}")
    print("-" * 60)
    
    for n in sizes:
        x = np.random.rand(n).astype(np.float32)
        y = np.random.rand(n).astype(np.float32)
        
        # numpy timing
        start = time.perf_counter()
        result_numpy = saxpy_numpy(a, x, y)
        time_numpy = (time.perf_counter() - start) * 1000
        
        # cupy timing (if available)
        if cupy_available:
            start = time.perf_counter()
            result_cupy = saxpy_cupy(a, x, y)
            time_cupy = (time.perf_counter() - start) * 1000
            speedup = time_numpy / time_cupy if time_cupy > 0 else 0
            
            print(f"{n:<10} {time_numpy:<12.2f} {time_cupy:<12.2f} {speedup:<10.2f}x")
        else:
            print(f"{n:<10} {time_numpy:<12.2f} {'n/a':<12} {'n/a':<10}")

if __name__ == "__main__":
    main()
    
    # optional extended benchmarks
    print("\n" + "=" * 50)
    print("extended benchmarks:")
    compare_memory_usage()
    benchmark_sizes()

"""
installation requirements:

# basic numpy
pip install numpy

# for gpu acceleration
pip install cupy-cuda11x  # or cupy-cuda12x for cuda 12
pip install pycuda
pip install numba

# alternative: use conda
conda install numpy cupy pycuda numba -c conda-forge

# note: copperhead is no longer maintained
# for modern python gpu computing, use:
# - cupy: numpy-like api for gpu
# - pycuda: low-level cuda access
# - numba: jit compilation for gpu
# - jax: functional programming for gpu/tpu
"""