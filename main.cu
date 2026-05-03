#include <cuda_runtime.h>
#include <iostream>

__global__ void fill_doubled_array(int *arr, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        arr[i] = 2 * i;
    }
}

int main(void)
{
    constexpr int N = 200;
    constexpr int threads_per_block = 64;
    constexpr int blocks = (N + threads_per_block - 1) / threads_per_block;

    int host_data[N];
    int *device_data;

    cudaMalloc(&device_data, N * sizeof(int));

    fill_doubled_array<<<blocks, threads_per_block>>>(device_data, N);

    cudaDeviceSynchronize();
    cudaMemcpy(host_data, device_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i{}; i < N; i++)
        std::cout << host_data[i] << " ";

    std::cout << '\n';

    cudaFree(device_data);

    return 0;
}