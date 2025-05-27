#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

template <typename T>
__global__ void real_dequantize_scaled_kernel(
    const T* Yq,
    T* Y,
    float s,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(Ypp[idx]) / s;
        Y[idx] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void real_dequantize_dual_scaled_kernel(
    const T* Yp, const T* Yn,
    T* Y,
    float s_pq, float s_nq,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val_pq = static_cast<float>(Yp[idx]) / s_pq;
        float val_nq = static_cast<float>(Yn[idx]) / s_nq;

        float sum = val_pq + val_nq;
        Y[idx] = static_cast<T>(sum);
    }
}