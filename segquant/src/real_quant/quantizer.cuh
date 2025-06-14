#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <torch/extension.h>
#include <cutlass/numeric_types.h>

//////////////////////////////////////////////////////////////////////
////////// Store data func
//////////////////////////////////////////////////////////////////////
template <typename OutputType, typename StoreType>
__device__ __forceinline__ void store_data(StoreType* out, int idx, float scaled_val);

template <>
__device__ __forceinline__ void store_data<__nv_fp8_e4m3, __nv_fp8_e4m3>(__nv_fp8_e4m3* out, int idx, float scaled_val) {
    out[idx] = static_cast<__nv_fp8_e4m3>(scaled_val);
}

template <>
__device__ __forceinline__ void store_data<__nv_fp8_e4m3, uint8_t>(uint8_t* out, int idx, float scaled_val) {
    reinterpret_cast<__nv_fp8_e4m3*>(out)[idx] = static_cast<__nv_fp8_e4m3>(scaled_val);
}

template <>
__device__ __forceinline__ void store_data<int8_t, int8_t>(int8_t* out, int idx, float scaled_val) {
    float clipped = fminf(fmaxf(scaled_val, -128.0f), 127.0f);
    out[idx] = static_cast<int8_t>(rintf(clipped));
}

template <>
__device__ __forceinline__ void store_data<cutlass::int4b_t, uint8_t>(uint8_t* out, int idx, float scaled_val) {
    float clipped = fminf(fmaxf(scaled_val, -8.0f), 7.0f);
    int ivalue = static_cast<int>(rintf(clipped)) & 0xF;
    int byte_idx = idx / 2;
    bool is_low = (idx % 2 == 0);
    uint8_t old = out[byte_idx];
    if (is_low) {
        old = (old & 0xF0) | ivalue;
    } else {
        old = (old & 0x0F) | (ivalue << 4);
    }
    out[byte_idx] = old;
}

//////////////////////////////////////////////////////////////////////
////////// Scaled kernel
//////////////////////////////////////////////////////////////////////
template <typename T, typename OutputType, typename StoreType>
__global__ void real_quantize_scaled_kernel(
    const T* inputs,
    float scale_x,
    size_t n,
    StoreType* Xq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);
        float scaled = val * scale_x;
        store_data<OutputType, StoreType>(Xq, idx, scaled);
    }
}

template <typename T, typename OutputType, typename StoreType>
__global__ void real_quantize_dual_scaled_kernel(
    const T *inputs,
    float pos_scale_x, float neg_scale_x,
    size_t n,
    StoreType* Xp,
    StoreType* Xn
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);

        if (val >= 0) {
            float scaled = val * pos_scale_x;
            store_data<OutputType, StoreType>(Xp, idx, scaled);
            store_data<OutputType, StoreType>(Xn, idx, 0.0f);
        } else {
            // val < 0, neg_scale_x > 0 --> scaled < 0
            float scaled = val * neg_scale_x;
            store_data<OutputType, StoreType>(Xn, idx, scaled);
            store_data<OutputType, StoreType>(Xp, idx, 0.0f);
        }
    }
}

__device__ __forceinline__ float get_scale_with_axis(int idx, const float* scales, size_t last_features) {
    // weight: idx (out, in), s: (out,)
    // input: idx (..., in), s: (..., )
    float scale_val;
    int row = idx / last_features;
    return scales[row];
}

template <typename T, typename OutputType, typename StoreType>
__global__ void real_quantize_scaled_kernel(
    const T* inputs,
    const float* scale_x,
    size_t last_features,
    size_t n,
    StoreType* Xq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);
        float scale_val = get_scale_with_axis(idx, scale_x, last_features);
        float scaled = val * scale_val;
        store_data<OutputType, StoreType>(Xq, idx, scaled);
    }
}

template <typename T, typename OutputType, typename StoreType, int row_flag>
__global__ void real_quantize_dual_scaled_kernel(
    const T *inputs,
    const float* pos_scale_x, const float* neg_scale_x,
    size_t last_features,
    size_t n,
    StoreType* Xp,
    StoreType* Xn
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);
        if (val >= 0) {
            float pos_scale_x = get_scale_with_axis(idx, pos_scale_x, last_features);
            float scaled = val * pos_scale_x;
            store_data<OutputType, StoreType>(Xp, idx, scaled);
            store_data<OutputType, StoreType>(Xn, idx, 0.0f);
        } else {
            // val < 0, neg_scale_x > 0 --> scaled < 0
            float neg_scale_x = get_scale_with_axis(idx, neg_scale_x, last_features);
            float scaled = val * neg_scale_x;
            store_data<OutputType, StoreType>(Xn, idx, scaled);
            store_data<OutputType, StoreType>(Xp, idx, 0.0f);
        }
    }
}