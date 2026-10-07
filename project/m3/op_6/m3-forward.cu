#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32
#define OUT_TILE 8
#define IN_TILE (TILE_WIDTH/OUT_TILE)

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output, int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width  - K + 1;

    const int w_base = Channel * K * K;
    const int cols = Batch * Height_out * Width_out;

    
    int tx  = threadIdx.x;
    int rowy = blockIdx.y * TILE_WIDTH;
    int row  = rowy + tx;

    int colx = blockIdx.x * OUT_TILE;

    float sumReg[OUT_TILE];

    for(int out=0; out<OUT_TILE; out++)
    {
        sumReg[out] = 0.0f;
    }

    __shared__ float sharedN[IN_TILE][OUT_TILE];

    int i = tx / OUT_TILE;  
    int j = tx % OUT_TILE;
    int col = colx + j;
    int b = col / (Height_out * Width_out);
    int h_out = (col % (Height_out * Width_out)) / Width_out;
    int w_out = (col % (Height_out * Width_out)) % Width_out;

    for(int t=0; t < (w_base + IN_TILE - 1)/IN_TILE; t++) {
        int k_base = t * IN_TILE;

        float maskReg[IN_TILE];
        for(int in=0; in<IN_TILE; in++) {
            int k = k_base + in;
            if (row < Map_out && k < w_base) {
                maskReg[in] = mask[row * w_base + k];
            } else {
                maskReg[in] = 0.0f;
            }
        }

        int k = k_base + i; 

        if (k < w_base && col < cols) {
            int c = k / (K*K);
            int v = (k % (K*K)) / K;
            int h = (k % (K*K)) % K;
            sharedN[i][j] = input[ b*(Channel*Height*Width) + c*(Height*Width) + (h_out+v)*Width + (w_out+h)
            ];
        } else {
            sharedN[i][j] = 0.0f;
        }

        __syncthreads();

        for(int in=0; in<IN_TILE; in++){
            float m_val = maskReg[in];
            for(int out=0; out<OUT_TILE; out++){
                sumReg[out] += m_val * sharedN[in][out];
            }
        }

        __syncthreads();
    }

    if (row < Map_out) {
        for(int out=0; out<OUT_TILE; out++){
            int col = colx + out;
            if (col < cols) {
                b     = col / (Height_out * Width_out);
                h_out = (col % (Height_out * Width_out)) / Width_out;
                w_out = (col % (Height_out * Width_out)) % Width_out;
                int out_idx =  b*(Map_out*Height_out*Width_out) + row*(Height_out*Width_out) + h_out*Width_out + w_out;
                output[out_idx] = sumReg[out];
            }
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
    cudaMalloc((void**) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    int cols = Batch * Height_out * Width_out;

    dim3 DimBlock(TILE_WIDTH, 1, 1);
    dim3 DimGrid((cols + OUT_TILE - 1)/OUT_TILE, (Map_out + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    matmul_conv_fused<<<DimGrid,DimBlock>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
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