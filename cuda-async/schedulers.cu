// analysis of cuda scheduler behavior on different gpu architectures
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel(float *data, int offset, int delay_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    // artificial work to make kernel duration visible
    float temp = data[idx];
    for (int i = 0; i < delay_factor; ++i) {
        temp = temp * 1.001f + 0.001f;
    }
    data[idx] = temp;
}

void analyze_scheduler_behavior() {
    printf("=== cuda scheduler behavior analysis ===\n\n");
    
    const int nStreams = 4;
    const int streamSize = 1024;
    const int streamBytes = streamSize * sizeof(float);
    const int blockSize = 256;
    const int delay_factor = 10000;  // make kernels take measurable time
    
    float *a, *d_a;
    cudaMallocHost(&a, nStreams * streamBytes);
    cudaMalloc(&d_a, nStreams * streamBytes);
    
    cudaStream_t stream[nStreams];
    cudaEvent_t events[nStreams * 6];  // h2d_start, h2d_end, k_start, k_end, d2h_start, d2h_end
    
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
        for (int j = 0; j < 6; ++j) {
            cudaEventCreate(&events[i * 6 + j]);
        }
    }
    
    // initialize data
    for (int i = 0; i < nStreams * streamSize; ++i) {
        a[i] = 1.0f;
    }
    
    printf("testing asynchronous version 2 pattern:\n");
    printf("loop 1: queue all h2d copies\n");
    printf("loop 2: queue all kernels (back-to-back in different streams)\n");
    printf("loop 3: queue all d2h copies\n\n");
    
    // version 2: separate loops (the problematic pattern on c2050)
    cudaEvent_t overall_start, overall_end;
    cudaEventCreate(&overall_start);
    cudaEventCreate(&overall_end);
    
    cudaEventRecord(overall_start, 0);
    
    // loop 1: queue all h2d copies
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(events[i * 6 + 0], stream[i]);  // h2d_start
        cudaMemcpyAsync(&d_a[offset], &a[offset], 
                       streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaEventRecord(events[i * 6 + 1], stream[i]);  // h2d_end
    }
    
    // loop 2: queue all kernels back-to-back
    printf("queueing kernels back-to-back:\n");
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(events[i * 6 + 2], stream[i]);  // kernel_start
        printf("  kernel %d queued in stream %d\n", i, i);
        test_kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset, delay_factor);
        cudaEventRecord(events[i * 6 + 3], stream[i]);  // kernel_end
    }
    
    // loop 3: queue all d2h copies
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(events[i * 6 + 4], stream[i]);  // d2h_start
        cudaMemcpyAsync(&a[offset], &d_a[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(events[i * 6 + 5], stream[i]);  // d2h_end
    }
    
    cudaEventRecord(overall_end, 0);
    cudaDeviceSynchronize();
    
    // analyze timing results
    printf("\n=== timing analysis ===\n");
    
    float overall_time;
    cudaEventElapsedTime(&overall_time, overall_start, overall_end);
    printf("overall execution time: %.3f ms\n\n", overall_time);
    
    // detailed per-stream timing
    for (int i = 0; i < nStreams; ++i) {
        float h2d_start_time, h2d_duration, kernel_start_time, kernel_duration;
        float d2h_start_time, d2h_duration;
        
        cudaEventElapsedTime(&h2d_start_time, overall_start, events[i * 6 + 0]);
        cudaEventElapsedTime(&h2d_duration, events[i * 6 + 0], events[i * 6 + 1]);
        cudaEventElapsedTime(&kernel_start_time, overall_start, events[i * 6 + 2]);
        cudaEventElapsedTime(&kernel_duration, events[i * 6 + 2], events[i * 6 + 3]);
        cudaEventElapsedTime(&d2h_start_time, overall_start, events[i * 6 + 4]);
        cudaEventElapsedTime(&d2h_duration, events[i * 6 + 4], events[i * 6 + 5]);
        
        printf("stream %d timeline:\n", i);
        printf("  h2d:    %.3f -> %.3f ms (duration: %.3f ms)\n", 
               h2d_start_time, h2d_start_time + h2d_duration, h2d_duration);
        printf("  kernel: %.3f -> %.3f ms (duration: %.3f ms)\n", 
               kernel_start_time, kernel_start_time + kernel_duration, kernel_duration);
        printf("  d2h:    %.3f -> %.3f ms (duration: %.3f ms)\n", 
               d2h_start_time, d2h_start_time + d2h_duration, d2h_duration);
        printf("\n");
    }
    
    // check for the c2050 scheduler issue
    printf("=== scheduler issue analysis ===\n");
    
    // find when last kernel ends and when first d2h starts
    float last_kernel_end = 0;
    float first_d2h_start = 1000000;  // large number
    
    for (int i = 0; i < nStreams; ++i) {
        float kernel_end_time, d2h_start_time;
        cudaEventElapsedTime(&kernel_end_time, overall_start, events[i * 6 + 3]);
        cudaEventElapsedTime(&d2h_start_time, overall_start, events[i * 6 + 4]);
        
        if (kernel_end_time > last_kernel_end) {
            last_kernel_end = kernel_end_time;
        }
        if (d2h_start_time < first_d2h_start) {
            first_d2h_start = d2h_start_time;
        }
    }
    
    printf("last kernel completion: %.3f ms\n", last_kernel_end);
    printf("first d2h transfer start: %.3f ms\n", first_d2h_start);
    printf("gap between kernel end and d2h start: %.3f ms\n", first_d2h_start - last_kernel_end);
    
    if (first_d2h_start >= last_kernel_end) {
        printf("\n*** c2050-style scheduler behavior detected! ***\n");
        printf("d2h transfers delayed until all kernels complete\n");
        printf("this reduces parallelism and hurts performance\n");
    } else {
        printf("\nmodern scheduler behavior: d2h can start immediately after individual kernel completion\n");
    }
    
    // cleanup
    cudaEventDestroy(overall_start);
    cudaEventDestroy(overall_end);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
        for (int j = 0; j < 6; ++j) {
            cudaEventDestroy(events[i * 6 + j]);
        }
    }
    cudaFree(d_a);
    cudaFreeHost(a);
}

void demonstrate_improved_version() {
    printf("\n=== improved version (interleaved) ===\n");
    
    const int nStreams = 4;
    const int streamSize = 1024;
    const int streamBytes = streamSize * sizeof(float);
    const int blockSize = 256;
    const int delay_factor = 10000;
    
    float *a, *d_a;
    cudaMallocHost(&a, nStreams * streamBytes);
    cudaMalloc(&d_a, nStreams * streamBytes);
    
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    
    // initialize data
    for (int i = 0; i < nStreams * streamSize; ++i) {
        a[i] = 1.0f;
    }
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    printf("interleaved version: h2d -> kernel -> d2h for each stream\n");
    
    cudaEventRecord(start, 0);
    
    // interleaved version: complete each stream's work before moving to next
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        
        printf("stream %d: h2d -> kernel -> d2h\n", i);
        
        // h2d copy
        cudaMemcpyAsync(&d_a[offset], &a[offset], 
                       streamBytes, cudaMemcpyHostToDevice, stream[i]);
        
        // kernel
        test_kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset, delay_factor);
        
        // d2h copy
        cudaMemcpyAsync(&a[offset], &d_a[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    
    float interleaved_time;
    cudaEventElapsedTime(&interleaved_time, start, end);
    printf("interleaved execution time: %.3f ms\n", interleaved_time);
    
    printf("\ninterleaved version avoids scheduler issue by:\n");
    printf("- not queueing multiple kernels back-to-back\n");
    printf("- allowing each d2h to start immediately after its kernel\n");
    printf("- maintains good overlap between streams\n");
    
    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_a);
    cudaFreeHost(a);
}

void explain_scheduler_evolution() {
    printf("\n=== cuda scheduler evolution ===\n\n");
    
    printf("c2050 (fermi) scheduler behavior:\n");
    printf("- when multiple kernels queued back-to-back in different streams\n");
    printf("- scheduler attempts concurrent kernel execution\n");
    printf("- delays kernel completion signals until ALL kernels finish\n");
    printf("- this blocks dependent d2h transfers\n");
    printf("- result: no overlap between kernels and d2h transfers\n\n");
    
    printf("modern gpu scheduler improvements:\n");
    printf("- individual kernel completion signals sent immediately\n");
    printf("- d2h transfers can start as soon as their specific kernel finishes\n");
    printf("- better parallelism between computation and communication\n");
    printf("- version 2 pattern works much better on modern hardware\n\n");
    
    printf("performance impact calculation (from blog post):\n");
    printf("sequential time: 12 units (4 * (h2d + kernel + d2h))\n");
    printf("ideal async time: 6 units (pipeline with perfect overlap)\n");
    printf("c2050 version 2: 9 units (h2d overlaps with kernels, but d2h is serial)\n");
    printf("efficiency on c2050: 9/12 = 75%% of sequential time\n");
    printf("modern gpus achieve closer to ideal 6 units\n");
}

int main() {
    // get gpu properties to understand current hardware
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("current gpu: %s\n", prop.name);
    printf("compute capability: %d.%d\n", prop.major, prop.minor);
    printf("concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("async engine count: %d\n\n", prop.asyncEngineCount);
    
    analyze_scheduler_behavior();
    demonstrate_improved_version();
    explain_scheduler_evolution();
    
    return 0;
}