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
        float val = static_cast<float>(Yq[idx]) / s;
        Y[idx] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void real_dequantize_dual_scaled_kernel(
    const float* Yp, const float* Yn,
    T* Y,
    float s_pq, float s_nq,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val_pq = Yp[idx] / s_pq;
        float val_nq = Yn[idx] / s_nq;

        float sum = val_pq + val_nq;
        Y[idx] = static_cast<T>(sum);
    }
}

//////////////////////////////////////////////////////////////////////
//// Axis Dequant for Y = (batch, out)
//// When weight axis = 1, scale_w = (out,)
//// When input axis = -1, scale_i = (batch,)
//////////////////////////////////////////////////////////////////////