#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define OFFSET_H TILE_WIDTH/2

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
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
    int Height_out = Height - K + 1;
    int Width_out = Width  - K + 1;
    int w_base = Channel * K * K;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    int b = blockIdx.z;
    int m_base = blockIdx.y * TILE_WIDTH;
    int row0 = threadIdx.y + m_base;
    int row1 = row0 + OFFSET_H;

    float val0 = 0;
    float val1 = 0;
    int numTiles = (w_base + TILE_WIDTH - 1) / TILE_WIDTH;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    for (int t = 0; t < numTiles; t++) {

        if (row0 < Map_out && t * TILE_WIDTH + threadIdx.x < w_base) 
        {
            tileA[threadIdx.y][threadIdx.x] = mask[row0 * w_base + t * TILE_WIDTH + threadIdx.x];
        } 
        else 
        {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        if (row1 < Map_out && t * TILE_WIDTH + threadIdx.x < w_base) 
        {
            tileA[threadIdx.y + OFFSET_H][threadIdx.x] = mask[row1 * w_base + t * TILE_WIDTH + threadIdx.x];
        } 
        else 
        {
            tileA[threadIdx.y + OFFSET_H][threadIdx.x] = 0;
        }

        int k0 = t * TILE_WIDTH + threadIdx.y;
        int k1 = k0 + OFFSET_H;

        if (b < Batch && k0 < w_base && col < (Height_out * Width_out)) {
            int h_out = (col / Width_out);
            int h = ((k0 % (K*K)) / K);
            int w_out = (col % Width_out);
            int v = ((k0 % (K*K)) % K);
            int c = k0 / (K*K);

            if (c < Channel && (h_out + h) < Height && (w_out + v) < Width) 
            {
                int in_idx = b*(Channel*Height*Width) + c * (Height*Width) + (h_out + h) * Width + (w_out + v);
                tileB[threadIdx.y][threadIdx.x] = input[in_idx];
            } 
            else 
            {
                tileB[threadIdx.y][threadIdx.x] = 0;
            }

        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        if (b < Batch && k1 < w_base && col < (Height_out * Width_out)) {
            int h_out = (col / Width_out);
            int h = ((k1 % (K*K)) / K);
            int w_out = (col % Width_out);
            int v = ((k1 % (K*K)) % K);
            int c = k1 / (K*K);

            if (c < Channel && (h_out + h) < Height && (w_out + v) < Width) 
            {
                int in_idx = b*(Channel*Height*Width) + c * (Height*Width) + (h_out + h) * Width + (w_out + v);
                tileB[threadIdx.y + OFFSET_H][threadIdx.x] = input[in_idx];
            } 
            else 
            {
                tileB[threadIdx.y + OFFSET_H][threadIdx.x] = 0;
            }

        } else {
            tileB[threadIdx.y + OFFSET_H][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            val0 += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
            val1 += tileA[threadIdx.y + OFFSET_H][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (col < (Height_out * Width_out)) {
        int w_out = col % Width_out;
        int h_out = col / Width_out;
        
        if (row0 < Map_out) 
        {
            int out_idx = b*Map_out*Height_out*Width_out + row0*Height_out*Width_out + h_out*Width_out + w_out;
            output[out_idx] = val0;
        }
        if (row1 < Map_out) 
        {
            int out_idx = b*Map_out*Height_out*Width_out + row1*Height_out*Width_out + h_out*Width_out + w_out;
            output[out_idx] = val1;
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
    cudaMalloc(device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc(device_mask_ptr, Map_out  * Channel * K * K * sizeof(float));
    cudaMalloc(device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out  * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width  - K + 1;

    dim3 DimBlock(TILE_WIDTH, OFFSET_H, 1);
    dim3 DimGrid(((Height_out * Width_out)  + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH, Batch);
    matmul_conv_fused<<<DimGrid, DimBlock>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
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