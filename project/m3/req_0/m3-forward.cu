#include <cmath>
#include <iostream>
#include <vector>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define STREAM_COUNT 5

__global__ void matrix_unrolling_kernel(const float *input, float *output, const int Batch, const int Channel, const int Height, const int Width, const int K, const int inib, const int slice) {
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int w_base = Channel * K * K;
    int w_slice  = slice * Height_out * Width_out;

    if (h_idx < w_base && w_idx < w_slice) {
        int f_idx = w_idx % (Height_out * Width_out);
        int b = w_idx / (Height_out * Width_out);
        int c = h_idx / (K * K);
        int k_idx = h_idx % (K * K);
        int h = k_idx / K + f_idx / Width_out;
        int w = k_idx % K + f_idx % Width_out;
        
        size_t in_idx = (size_t)(b + inib) * (Channel * Height * Width) + (size_t)c  * (Height * Width) + (size_t)h  * Width + (size_t)w;

        size_t out_idx = (size_t)h_idx * ((size_t)Batch * Height_out * Width_out) + (size_t)b  * (Height_out * Width_out) + (size_t)f_idx + (size_t)inib * (Height_out * Width_out);
        
        output[out_idx] = input[in_idx];
    }
}

// Tiled matrix multiplication kernel (shared memory) with streaming offsets
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns, int Batch, int Height_out, int Width_out, int inib, int slice) 
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float val = 0;
    int bCols = slice * Height_out * Width_out;

    for (int tileID = 0; tileID < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; tileID++) {
        if (row < numARows && tileID * TILE_WIDTH + tx < numAColumns) 
        {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileID * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (tileID * TILE_WIDTH + ty < numBRows && col < bCols) 
        {
            tileB[ty][tx] = B[((size_t) tileID * TILE_WIDTH + ty) * ((size_t)Batch * Height_out * Width_out) + (inib * Height_out * Width_out) + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < bCols) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < bCols) {
        C[row * ((size_t)Batch * Height_out * Width_out) + (inib * Height_out * Width_out) + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) 
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;
    int slices = Batch / STREAM_COUNT;
    int w_base = Channel * K * K;

    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr,  Map_out * Channel * K * K * sizeof(float));
    cudaMalloc((void**)device_output_ptr,Batch * Map_out * Height_out * Width_out * sizeof(float));

    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    cudaStream_t stream_arr[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&stream_arr[i]);
    }

    for (int i = 0; i < STREAM_COUNT; i++) {
        int inib = i * slices;
        const float *slice1 = host_input + inib * Channel * Height * Width;
        float *slice2 = *device_input_ptr + inib * Channel * Height * Width;

        cudaMemcpyAsync(slice2, slice1, (size_t)slices * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice, stream_arr[i]);

        int w_slice = slices * Height_out * Width_out;
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 DimGrid((w_slice + TILE_WIDTH - 1) / TILE_WIDTH, (w_base + TILE_WIDTH - 1) / TILE_WIDTH, 1);
        matrix_unrolling_kernel<<<DimGrid, DimBlock, 0, stream_arr[i]>>>(*device_input_ptr, unrolled_matrix, Batch, Channel, Height, Width, K, inib, slices);

        dim3 matmul_grid_dim(((slices * Height_out * Width_out) + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);
        dim3 matmul_block_dim(TILE_WIDTH, TILE_WIDTH, 1);
        
        matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim, 0, stream_arr[i]>>>(*device_mask_ptr, unrolled_matrix, matmul_output, Map_out, Channel * K * K, Channel * K * K, Batch * Height_out * Width_out, Map_out, Batch * Height_out * Width_out, Batch, Height_out, Width_out, inib, slices);
    }

    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(stream_arr[i]);
    }

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, *device_output_ptr, Map_out, Batch, out_image_size
    );

    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamDestroy(stream_arr[i]);
    }

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}