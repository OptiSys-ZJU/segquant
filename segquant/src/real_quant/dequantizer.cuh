#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>


__device__ __forceinline__ float get_scale_with_axis_n_segment(int idx, const float* scale_x, const float* scale_w, size_t segment_stride, size_t last_features, bool enable_scale_x_broadcast, bool enable_scale_w_broadcast) {
    // output (segments, ..., M, N)
    // scale x (segments, 1) or (segments, ..., M)
    // scale w: (segments, 1) or (segments, N)
    int seg_id = idx / segment_stride;
    int offset = idx % segment_stride;
    int token_id = offset / last_features;
    int channel_id = offset % last_features;

    float scale_x_val;
    float scale_w_val;
    if (enable_scale_x_broadcast) {
        scale_x_val = scale_x[seg_id];
    }
    else {
        scale_x_val = scale_x[seg_id * segment_stride + token_id];
    }

    if (enable_scale_w_broadcast) {
        scale_w_val = scale_w[seg_id];
    }
    else {
        scale_w_val = scale_w[seg_id * last_features + channel_id];
    }
    return scale_x_val * scale_w_val;
}

template <typename T>
__global__ void real_dequantize_scaled_kernel(
    const float* Yq,
    T* Y,
    const float* scale_x, const float* scale_w,
    size_t segment_stride, size_t last_features,
    bool enable_scale_x_broadcast, bool enable_scale_w_broadcast,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float scale = get_scale_with_axis_n_segment(idx, scale_x, scale_w, segment_stride, last_features, enable_scale_x_broadcast, enable_scale_w_broadcast);
        float val = static_cast<float>(Yq[idx]) / scale;
        Y[idx] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void real_dequantize_dual_scaled_kernel(
    const float* Yp, const float* Yn,
    T* Y,
    const float* pos_scale_x, const float* neg_scale_x, const float* scale_w,
    size_t segment_stride, size_t last_features,
    bool enable_scale_x_broadcast, bool enable_scale_w_broadcast,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float pos_scale = get_scale_with_axis_n_segment(idx, pos_scale_x, scale_w, segment_stride, last_features, enable_scale_x_broadcast, enable_scale_w_broadcast);
        float neg_scale = get_scale_with_axis_n_segment(idx, neg_scale_x, scale_w, segment_stride, last_features, enable_scale_x_broadcast, enable_scale_w_broadcast);

        float val_pq = Yp[idx] / denom_pq;
        float val_nq = Yn[idx] / denom_nq;

        float sum = val_pq + val_nq;
        Y[idx] = static_cast<T>(sum);
    }
}

template <typename T>
__global__ void real_dequantize_dual_scaled_kernel(
    const float* Yp, const float* Yn,
    T* Y,
    float s_pq, float s_nq,
    float s_w,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr float epsilon = 1e-8f;
    float denom_pq = fmaxf(s_pq * s_w, epsilon);
    float denom_nq = fmaxf(s_nq * s_w, epsilon);

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val_pq = Yp[idx] / denom_pq;
        float val_nq = Yn[idx] / denom_nq;

        float sum = val_pq + val_nq;
        Y[idx] = static_cast<T>(sum);
    }
}