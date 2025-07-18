// 1d grid of 1d blocks
// blockIdx.x: current block's x-coordinate in grid
// blockDim.x: number of threads per block in x dimension  
// threadIdx.x: current thread's x-coordinate within block
// formula: block_offset + thread_offset_within_block
__device__ int getGlobalIdx_1D_1D(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// 1d grid of 2d blocks
// blockDim.y * blockDim.x: total threads per block (2d block flattened)
// threadIdx.y * blockDim.x: row offset within block (thread's y-coord * width)
// threadIdx.x: column offset within block
__device__ int getGlobalIdx_1D_2D(){
    return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

// 1d grid of 3d blocks  
// blockDim.x * blockDim.y * blockDim.z: total threads per block (3d block flattened)
// threadIdx.z * blockDim.y * blockDim.x: z-layer offset (z-coord * area_per_layer)
// threadIdx.y * blockDim.x: y-row offset within z-layer
// threadIdx.x: x-column offset within row
__device__ int getGlobalIdx_1D_3D(){
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z + 
           threadIdx.z * blockDim.y * blockDim.x + 
           threadIdx.y * blockDim.x + 
           threadIdx.x;
}

// 2d grid of 1d blocks
// gridDim.x: number of blocks in x dimension of grid
// blockIdx.y * gridDim.x: row offset in grid (current_row * blocks_per_row)
// blockIdx.x: column offset in grid  
// blockId: flattened block index in 2d grid
__device__ int getGlobalIdx_2D_1D(){
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 2d grid of 2d blocks
// blockId: flattened block index using row-major ordering
// blockDim.x * blockDim.y: total threads per block
// threadIdx.y * blockDim.x: thread's row offset within block
// threadIdx.x: thread's column offset within row
__device__ int getGlobalIdx_2D_2D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + 
                   (threadIdx.y * blockDim.x) + 
                   threadIdx.x;
    return threadId;
}

// 2d grid of 3d blocks
// blockId: flattened 2d grid position  
// blockDim.x * blockDim.y * blockDim.z: total threads per 3d block
// threadIdx.z * (blockDim.x * blockDim.y): offset to correct z-layer
// threadIdx.y * blockDim.x: offset to correct row within layer
// threadIdx.x: offset to correct column within row
__device__ int getGlobalIdx_2D_3D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + 
                   (threadIdx.z * (blockDim.x * blockDim.y)) + 
                   (threadIdx.y * blockDim.x) + 
                   threadIdx.x;
    return threadId;
}

// 3d grid of 1d blocks
// gridDim.x * gridDim.y * blockIdx.z: offset for z-layers of blocks
// blockIdx.y * gridDim.x: offset for y-rows of blocks  
// blockIdx.x: offset for x-column of blocks
// blockId: flattened position in 3d grid using z*area + y*width + x
__device__ int getGlobalIdx_3D_1D(){
    int blockId = blockIdx.x + 
                  blockIdx.y * gridDim.x + 
                  gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 3d grid of 2d blocks
// blockId: 3d grid flattened to linear index
// blockDim.x * blockDim.y: threads per 2d block
// threadIdx.y * blockDim.x: row offset within 2d block
// threadIdx.x: column offset within row
__device__ int getGlobalIdx_3D_2D(){
    int blockId = blockIdx.x + 
                  blockIdx.y * gridDim.x + 
                  gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + 
                   (threadIdx.y * blockDim.x) + 
                   threadIdx.x;
    return threadId;
}

// 3d grid of 3d blocks
// blockId: 3d grid position flattened using z*area + y*width + x formula
// blockDim.x * blockDim.y * blockDim.z: total threads per 3d block
// threadIdx.z * (blockDim.x * blockDim.y): z-layer offset within block
// threadIdx.y * blockDim.x: y-row offset within z-layer  
// threadIdx.x: x-column offset within row
__device__ int getGlobalIdx_3D_3D(){
    int blockId = blockIdx.x + 
                  blockIdx.y * gridDim.x + 
                  gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + 
                   (threadIdx.z * (blockDim.x * blockDim.y)) + 
                   (threadIdx.y * blockDim.x) + 
                   threadIdx.x;
    return threadId;
}