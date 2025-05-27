/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
__global__ void real_quantize_e4m3fy_dual_scaled_split_kernel(
    const T *inputs,
    float pos_scale_x, float neg_scale_x,
    size_t n,
    __nv_fp8_e4m3 *Xp,
    __nv_fp8_e4m3 *Xn
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);

        if (val >= 0) {
            float scaled = val * pos_scale_x;
            Xp[idx] = static_cast<__nv_fp8_e4m3>(scaled);
            Xn[idx] = static_cast<__nv_fp8_e4m3>(0.0f);
        } else {
            // val < 0, neg_scale_x < 0 --> scaled > 0
            float scaled = val * neg_scale_x;
            Xn[idx] = static_cast<__nv_fp8_e4m3>(scaled);
            Xp[idx] = static_cast<__nv_fp8_e4m3>(0.0f);
        }
    }
}

template <typename T>
__global__ void real_dequantize_e4m3fy_dual_scaled_combine_kernel(
    const T* Ypp, const T* Ypn, const T* Ynp, const T* Ynn,
    T* Y,
    float s_pp, float s_pn, float s_np, float s_nn,
    size_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val_pp = static_cast<float>(Ypp[idx]) / s_pp;
        float val_pn = static_cast<float>(Ypn[idx]) / s_pn;
        float val_np = static_cast<float>(Ynp[idx]) / s_np;
        float val_nn = static_cast<float>(Ynn[idx]) / s_nn;

        float sum = val_pp + val_pn + val_np + val_nn;
        Y[idx] = static_cast<T>(sum);
    }
}

template <typename T>
__global__ void real_quantize_e4m3fy_scaled_kernel(
    const T *inputs,
    float scale_x,
    size_t n,
    __nv_fp8_e4m3 *Xq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        float val = static_cast<float>(inputs[idx]);
        float scaled = val * scale_x;
        Xq[idx] = static_cast<__nv_fp8_e4m3>(scaled);
    }
}

template <typename T>
__global__ void real_dequantize_e4m3fy_scaled_kernel(
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