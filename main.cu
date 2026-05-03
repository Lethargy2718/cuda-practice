#include <cuda_runtime.h>
#include <iostream>

__global__ void fill_array(int *arr, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        arr[i] = 2 * i;
    }
}

int main(void)
{
    constexpr int N = 1'000'000'000;
    constexpr int threads_per_block = 256;
    constexpr int blocks = (N + threads_per_block - 1) / threads_per_block;

    int *host_data = new int[N];
    int *device_data;

    cudaMalloc(&device_data, N * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark kernel
    cudaEventRecord(start);
    fill_array<<<blocks, threads_per_block>>>(device_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    std::cout << "Kernel execution: " << kernel_ms / 1000.0 << " s\n";

    // Benchmark memcpy (Device to Host)
    cudaEventRecord(start);
    cudaMemcpy(host_data, device_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memcpy_ms = 0;
    cudaEventElapsedTime(&memcpy_ms, start, stop);
    std::cout << "Memcpy (D2H): " << memcpy_ms / 1000.0 << " s\n";

    // Calculate bandwidth
    float gb = (N * sizeof(int)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Data transferred: " << gb << " GB\n";
    std::cout << "Copy bandwidth: " << (gb / (memcpy_ms / 1000.0)) << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_data);
    delete[] host_data;

    return 0;
}