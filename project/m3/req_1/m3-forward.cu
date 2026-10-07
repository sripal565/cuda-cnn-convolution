#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>

using namespace nvcuda::wmma;

#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__global__ void matmul_conv_fused(const float* mask, const float* input, float* output, int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int w_base     = Channel * K * K;

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tempC[TILE_WIDTH][TILE_WIDTH];

    //fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    __syncthreads();

    int numTiles = (w_base + WMMA_K - 1) / WMMA_K;
    for (int t = 0; t < numTiles; ++t) {
        int baseK = t * WMMA_K;

        int a_col = baseK + threadIdx.x;
        if (row < Map_out && a_col < w_base) {
            tileA[threadIdx.y][threadIdx.x] = mask[row * w_base + a_col];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int k = baseK + threadIdx.y;
        if (k < w_base && col < (Batch * Height_out * Width_out)) {
            int b_n = col / (Height_out * Width_out);
            int h_out = (col % (Height_out * Width_out)) / Width_out;
            int w_out = (col % (Height_out * Width_out)) % Width_out;
            int c = k / (K * K);
            int v = (k % (K * K)) / K;
            int h = (k % (K * K)) % K;
            int in_idx = b_n * (Channel * Height * Width) + c * (Height * Width) + (h_out + v) * Width + (w_out + h);
            tileB[threadIdx.y][threadIdx.x] = input[in_idx];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Load into fragments
        load_matrix_sync(a_frag, &tileA[0][0], TILE_WIDTH);
        load_matrix_sync(b_frag, &tileB[0][0], TILE_WIDTH);

        mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store results to shared
    store_matrix_sync(&tempC[0][0], c_frag, TILE_WIDTH, mem_row_major);
    __syncthreads();

    // Write output
    if (row < Map_out && col < (Batch * Height_out * Width_out)) {
        int b_n   = col / (Height_out * Width_out);
        int h_out = (col % (Height_out * Width_out)) / Width_out;
        int w_out = (col % (Height_out * Width_out)) % Width_out;
        int out_idx = b_n * (Map_out * Height_out * Width_out) + row * (Height_out * Width_out) + h_out * Width_out + w_out;
        output[out_idx] = tempC[threadIdx.y][threadIdx.x];
    }
}

// prolog, epilog, and get_device_properties unchanged
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    cudaMalloc((void**) device_input_ptr, size_t(Batch) * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**) device_output_ptr, size_t(Batch) * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, size_t(Map_out) * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_mask_ptr, host_mask, size_t(Map_out) * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, size_t(Batch) * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width - K + 1;
    int cols = Batch * Height_out * Width_out;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_conv_fused<<<DimGrid, DimBlock>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    cudaMemcpy(host_output, device_output, size_t(Batch) * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
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
