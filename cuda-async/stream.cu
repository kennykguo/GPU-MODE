// detailed analysis of cuda stream synchronization
#include <cuda_runtime.h>
#include <stdio.h>

// kernel for demonstration
__global__ void kernel(float *data, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    data[idx] = data[idx] * 2.0f + 1.0f;
}

void demonstrate_stream_synchronization() {
    const int nStreams = 4;
    const int streamSize = 1024;
    const int streamBytes = streamSize * sizeof(float);
    const int blockSize = 256;
    
    // allocate host and device memory
    float *a, *d_a;
    cudaMallocHost(&a, nStreams * streamBytes);
    cudaMalloc(&d_a, nStreams * streamBytes);
    
    // create streams
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    
    // initialize host data
    for (int i = 0; i < nStreams * streamSize; ++i) {
        a[i] = i;
    }
    
    printf("=== stream synchronization analysis ===\n\n");
    
    // the problematic-looking version from your question
    printf("version 2: separate loops (appears problematic but actually works)\n");
    printf("timeline analysis:\n\n");
    
    // loop 1: queue all host-to-device copies
    printf("loop 1 - queueing h2d copies:\n");
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        printf("  stream[%d]: queue h2d copy (offset=%d)\n", i, offset);
        cudaMemcpyAsync(&d_a[offset], &a[offset], 
                       streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    
    // loop 2: queue all kernels 
    printf("\nloop 2 - queueing kernels:\n");
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        printf("  stream[%d]: queue kernel (depends on h2d copy completion)\n", i);
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    }
    
    // loop 3: queue all device-to-host copies
    printf("\nloop 3 - queueing d2h copies:\n");
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        printf("  stream[%d]: queue d2h copy (depends on kernel completion)\n", i);
        cudaMemcpyAsync(&a[offset], &d_a[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    // wait for all operations to complete
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamSynchronize(stream[i]);
    }
    
    printf("\n=== why this works: stream ordering guarantees ===\n");
    printf("each stream maintains fifo (first-in-first-out) execution order\n\n");
    
    // demonstrate the execution timeline
    printf("actual execution timeline:\n");
    printf("time ->  0    1    2    3    4    5    6\n");
    printf("stream0: h2d0 kern0 d2h0\n");
    printf("stream1:      h2d1 kern1 d2h1\n");
    printf("stream2:           h2d2 kern2 d2h2\n");
    printf("stream3:                h2d3 kern3 d2h3\n");
    printf("\nkey insight: kern0 waits for h2d0, kern1 waits for h2d1, etc.\n");
    
    // cleanup
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_a);
    cudaFreeHost(a);
}

// more detailed demonstration with events to show timing
void demonstrate_with_events() {
    printf("\n=== detailed timing demonstration ===\n");
    
    const int nStreams = 2;  // simplified for clarity
    const int streamSize = 1024;
    const int streamBytes = streamSize * sizeof(float);
    const int blockSize = 256;
    
    float *a, *d_a;
    cudaMallocHost(&a, nStreams * streamBytes);
    cudaMalloc(&d_a, nStreams * streamBytes);
    
    cudaStream_t stream[nStreams];
    cudaEvent_t h2d_start[nStreams], h2d_end[nStreams];
    cudaEvent_t kernel_start[nStreams], kernel_end[nStreams];
    cudaEvent_t d2h_start[nStreams], d2h_end[nStreams];
    
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&h2d_start[i]);
        cudaEventCreate(&h2d_end[i]);
        cudaEventCreate(&kernel_start[i]);
        cudaEventCreate(&kernel_end[i]);
        cudaEventCreate(&d2h_start[i]);
        cudaEventCreate(&d2h_end[i]);
    }
    
    // initialize data
    for (int i = 0; i < nStreams * streamSize; ++i) {
        a[i] = i;
    }
    
    // version 2 with detailed event recording
    printf("executing version 2 with event timing...\n");
    
    // queue all h2d copies with events
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(h2d_start[i], stream[i]);
        cudaMemcpyAsync(&d_a[offset], &a[offset], 
                       streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaEventRecord(h2d_end[i], stream[i]);
    }
    
    // queue all kernels with events
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(kernel_start[i], stream[i]);
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        cudaEventRecord(kernel_end[i], stream[i]);
    }
    
    // queue all d2h copies with events  
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaEventRecord(d2h_start[i], stream[i]);
        cudaMemcpyAsync(&a[offset], &d_a[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(d2h_end[i], stream[i]);
    }
    
    // synchronize and print timing
    cudaDeviceSynchronize();
    
    printf("\ntiming results (relative to stream 0 h2d start):\n");
    for (int i = 0; i < nStreams; ++i) {
        float h2d_time, kernel_time, d2h_time;
        float h2d_start_time, kernel_start_time, d2h_start_time;
        
        cudaEventElapsedTime(&h2d_start_time, h2d_start[0], h2d_start[i]);
        cudaEventElapsedTime(&kernel_start_time, h2d_start[0], kernel_start[i]);
        cudaEventElapsedTime(&d2h_start_time, h2d_start[0], d2h_start[i]);
        
        cudaEventElapsedTime(&h2d_time, h2d_start[i], h2d_end[i]);
        cudaEventElapsedTime(&kernel_time, kernel_start[i], kernel_end[i]);
        cudaEventElapsedTime(&d2h_time, d2h_start[i], d2h_end[i]);
        
        printf("stream %d:\n", i);
        printf("  h2d:    start=%.3fms, duration=%.3fms\n", h2d_start_time, h2d_time);
        printf("  kernel: start=%.3fms, duration=%.3fms\n", kernel_start_time, kernel_time);
        printf("  d2h:    start=%.3fms, duration=%.3fms\n", d2h_start_time, d2h_time);
        printf("\n");
    }
    
    // cleanup
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(h2d_start[i]);
        cudaEventDestroy(h2d_end[i]);
        cudaEventDestroy(kernel_start[i]);
        cudaEventDestroy(kernel_end[i]);
        cudaEventDestroy(d2h_start[i]);
        cudaEventDestroy(d2h_end[i]);
    }
    cudaFree(d_a);
    cudaFreeHost(a);
}

// comparison: what would happen with improper synchronization
void demonstrate_race_condition() {
    printf("\n=== what a real race condition looks like ===\n");
    printf("(this is NOT what happens in the original code)\n\n");
    
    const int streamSize = 1024;
    const int streamBytes = streamSize * sizeof(float);
    const int blockSize = 256;
    
    float *a, *d_a;
    cudaMallocHost(&a, streamBytes);
    cudaMalloc(&d_a, streamBytes);
    
    cudaStream_t copyStream, kernelStream;
    cudaStreamCreate(&copyStream);
    cudaStreamCreate(&kernelStream);  // different stream!
    
    // initialize data
    for (int i = 0; i < streamSize; ++i) {
        a[i] = i;
    }
    
    printf("incorrect version (would cause race condition):\n");
    printf("// copy in one stream\n");
    printf("cudaMemcpyAsync(d_a, a, bytes, h2d, copyStream);\n");
    printf("// kernel in different stream - potential race!\n");
    printf("kernel<<<..., kernelStream>>>(d_a, 0);\n\n");
    
    printf("this WOULD be problematic because:\n");
    printf("- copy and kernel are in different streams\n");
    printf("- no explicit synchronization between streams\n");
    printf("- kernel might start before copy completes\n\n");
    
    // but in the original code, everything is in the same stream per chunk
    printf("original code avoids this by using same stream per data chunk\n");
    
    cudaStreamDestroy(copyStream);
    cudaStreamDestroy(kernelStream);
    cudaFree(d_a);
    cudaFreeHost(a);
}

int main() {
    demonstrate_stream_synchronization();
    demonstrate_with_events();
    demonstrate_race_condition();
    
    printf("\n=== summary ===\n");
    printf("the original code works because:\n");
    printf("1. each stream maintains fifo execution order\n");
    printf("2. kernel[i] is queued after h2d[i] in same stream[i]\n");
    printf("3. cuda runtime automatically waits for h2d[i] before starting kernel[i]\n");
    printf("4. operations in different streams can overlap for performance\n");
    
    return 0;
}