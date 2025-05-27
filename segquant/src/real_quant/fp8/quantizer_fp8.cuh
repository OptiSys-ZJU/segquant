#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

template <typename T, typename OutputType>
__device__ inline void store_fp8(OutputType* out, int idx, float scaled_val);

template <typename T>
__device__ inline void store_fp8(__nv_fp8_e4m3* out, int idx, float scaled_val) {
    out[idx] = static_cast<__nv_fp8_e4m3>(scaled_val);
}

template <typename T>
__device__ inline void store_fp8(uint8_t* out, int idx, float scaled_val) {
    __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(scaled_val);
    out[idx] = *reinterpret_cast<uint8_t*>(&fp8_val);
}

template <typename T, typename OutputType>
__global__ void real_quantize_e4m3fy_scaled_kernel(
    const T* inputs,
    float scale_x,
    size_t n,
    OutputType* Xq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);
        float scaled = val * scale_x;
        store_fp8<T>(Xq, idx, scaled);
    }
}

template <typename T>
__global__ void real_quantize_e4m3fy_dual_scaled_kernel(
    const T *inputs,
    float pos_scale_x, float neg_scale_x,
    size_t n,
    OutputType* Xp,
    OutputType* Xn
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);

        if (val >= 0) {
            float scaled = val * pos_scale_x;
            store_fp8<T>(Xp, idx, scaled);
            store_fp8<T>(Xn, idx, 0.0f);
        } else {
            // val < 0, neg_scale_x > 0 --> scaled < 0
            float scaled = val * neg_scale_x;
            store_fp8<T>(Xn, idx, scaled);
            store_fp8<T>(Xp, idx, 0.0f);
        }
    }
}