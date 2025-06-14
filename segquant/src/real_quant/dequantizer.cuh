#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

template <typename T>
__global__ void real_dequantize_scaled_kernel(
    const float* Yq,
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
//// Axis Dequant for Y = (..., out)
//// When weight use axis, scale_w = (out,), so row_flag = 0
//// When input use axis, scale_i = (...,), so row_flag = 1
//////////////////////////////////////////////////////////////////////
template <int row_flag>
__device__ __forceinline__ float get_scale_with_row_col(int idx, const float* scales, size_t last_features) {
    float scale_val;
    if constexpr (row_flag == 1) {
        int row = idx / last_features;
        scale_val = scales[row];
    }
    else {
        int col = idx % last_features;
        scale_val = scales[col];
    }
    return scale_val;
}

template <typename T, int row_flag>
__global__ void real_dequantize_scaled_kernel(
    const float* Yq,
    T* Y,
    const float* s,
    size_t last_features,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float scale_val = get_scale_with_row_col<row_flag>(idx, s, last_features);
        float val = static_cast<float>(Yq[idx]) / scale_val;
        Y[idx] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void real_dequantize_scaled_kernel(
    const float* Yq,
    T* Y,
    const float* s_i, const float* s_w,
    size_t last_features,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        int row = idx / last_features;
        int col = idx % last_features;

        float scale_i = s_i[row];
        float scale_w = s_w[col];

        float denom = fmaxf(scale_i * scale_w, 1e-8f);
        float val = static_cast<float>(Yq[idx]) / denom;

        Y[idx] = static_cast<T>(val);
    }
}