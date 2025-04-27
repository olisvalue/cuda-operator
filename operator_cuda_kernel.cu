#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA-ядро: y[i] = x[i] * x[i]
__global__ void square_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    size_t n
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * x[idx];
    }
}

void square_cuda(at::Tensor x, at::Tensor y) {
    const size_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    square_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in square_kernel: %s\n", cudaGetErrorString(err));
    }
}
