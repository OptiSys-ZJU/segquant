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

template <typename T> __global__ void fake_e4m3fy_kernel(const T *inputs, size_t n, T *outputs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        outputs[idx] = static_cast<T>(
            static_cast<float>(static_cast<__nv_fp8_e4m3>(static_cast<float>(inputs[idx]))));
    }
}

at::Tensor fake_e4m3fy_cuda(at::Tensor inputs) {
    size_t numel = inputs.numel();
    auto outputs = torch::empty_like(inputs);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "fake_e4m3fy_cuda", [&] {
        fake_e4m3fy_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>());
    });
    return outputs;
}