#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <torch/extension.h>

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