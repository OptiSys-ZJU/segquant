#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cutlass/util/device_memory.h>

#define INSTANTIATE_QUANTIZE_WEIGHTS(T) \
template<> \
void real_quantized_quantize_weights<T>(at::Tensor weights, at::Tensor outputs, float scale_w); \
template<> \
void real_quantized_quantize_weights<T>(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w);

template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, float scale_w);
template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w);

template<typename input_type, typename weight_type, typename scale_x_type, typename scale_w_type>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, scale_x_type scale_x, scale_w_type scale_w);

template<typename input_type, typename weight_type, typename scale_x_type, typename scale_w_type>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, scale_x_type pos_scale_x, scale_x_type neg_scale_x, scale_w_type scale_w);

template <typename T>
struct StoreType;

template <>
struct StoreType<int8_t> {
    using type = int8_t;
};

template <>
struct StoreType<__nv_fp8_e4m3> {
    using type = uint8_t;
};

template <>
struct StoreType<cutlass::int4b_t> {
    using type = uint8_t;
};

template <>
struct StoreType<at::Half> {
    using type = at::Half;
};
template <typename T>
struct CUDAStoreType {
    using type = typename StoreType<T>::type;
};

template <>
struct CUDAStoreType<at::Half> {
    using type = cutlass::half_t;
};

inline void tensor_check(at::Tensor x) {
    TORCH_CHECK(x.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(x.is_cuda(), "tensor must be a CUDA tensor");
};

inline void tensor_check(at::Tensor x, at::ScalarType dtype) {
    TORCH_CHECK(x.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(x.is_cuda(), "tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == dtype, "tensor dtype failed");
};

inline void tensor_extra_check_int4(at::Tensor weights, at::Tensor outputs) {
    TORCH_CHECK(weights.numel() % 2 == 0, "Quantization to int4 requires the number of elements to be even");
    TORCH_CHECK(outputs.numel() * 2 == weights.numel(), "output numel must be half of weight");
}

inline void cuda_device_check(at::Tensor x) {
    int current_device;
    cudaGetDevice(&current_device);
    int target_device = x.device().index();
    if (current_device != target_device) {
        cudaSetDevice(target_device);
    }
}

template<typename A>
void register_quantweight_module(pybind11::module_& m) {
    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, float scale_w) {
            tensor_check(weights);
            tensor_check(outputs, c10::CppTypeToScalarType<typename StoreType<A>::type>::value);
            if constexpr(std::is_same<A, cutlass::int4b_t>::value) {
                tensor_extra_check_int4(weights, outputs);
            }

            cuda_device_check(weights);
            real_quantized_quantize_weights<A>(weights, outputs, scale_w);
        },
        "Quantize weights to lowbit format",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w").noconvert()
    );

    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, at::Tensor scale_w) {
            tensor_check(weights);
            tensor_check(outputs, c10::CppTypeToScalarType<typename StoreType<A>::type>::value);
            tensor_check(scale_w, at::kFloat);
            if constexpr(std::is_same<A, cutlass::int4b_t>::value) {
                tensor_extra_check_int4(weights, outputs);
            }

            cuda_device_check(weights);
            real_quantized_quantize_weights<A>(weights, outputs, scale_w);
        },
        "Quantize weights to lowbit format with axis",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w")
    );
}

template<typename A, typename B>
void register_gemm_module(pybind11::module_& m, const std::string& prefix) {
    m.def((prefix + "_real_quantized_gemm_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            
            cuda_device_check(weights);
            return real_quantized_gemm_scaled<A, B, float, float>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x").noconvert(),
        py::arg("scale_w").noconvert()
    );

    m.def((prefix + "_real_quantized_gemm_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, float scale_x, at::Tensor scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(scale_w, at::kFloat);

            cuda_device_check(weights);
            return real_quantized_gemm_scaled<A, B, float, at::Tensor>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x").noconvert(),
        py::arg("scale_w")
    );

    m.def((prefix + "_real_quantized_gemm_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, at::Tensor scale_x, float scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(scale_x, at::kFloat);
            
            cuda_device_check(weights);
            return real_quantized_gemm_scaled<A, B, at::Tensor, float>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w").noconvert()
    );

    m.def((prefix + "_real_quantized_gemm_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, at::Tensor scale_x, at::Tensor scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(scale_x, at::kFloat);
            tensor_check(scale_w, at::kFloat);

            cuda_device_check(weights);
            return real_quantized_gemm_scaled<A, B, at::Tensor, at::Tensor>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w")
    );

    m.def((prefix + "_real_quantized_gemm_dual_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            
            cuda_device_check(weights);
            return real_quantized_gemm_dual_scaled<A, B, float, float>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x").noconvert(),
        py::arg("neg_scale_x").noconvert(),
        py::arg("scale_w").noconvert()
    );

    m.def((prefix + "_real_quantized_gemm_dual_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, at::Tensor scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(scale_w, at::kFloat);

            cuda_device_check(weights);
            return real_quantized_gemm_dual_scaled<A, B, float, at::Tensor>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x").noconvert(),
        py::arg("neg_scale_x").noconvert(),
        py::arg("scale_w")
    );

    m.def((prefix + "_real_quantized_gemm_dual_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, at::Tensor pos_scale_x, at::Tensor neg_scale_x, float scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(pos_scale_x, at::kFloat);
            tensor_check(neg_scale_x, at::kFloat);

            cuda_device_check(weights);
            return real_quantized_gemm_dual_scaled<A, B, at::Tensor, float>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w").noconvert()
    );

    m.def((prefix + "_real_quantized_gemm_dual_scaled").c_str(),
        [](at::Tensor inputs, at::Tensor weights, at::Tensor pos_scale_x, at::Tensor neg_scale_x, at::Tensor scale_w) {
            tensor_check(inputs);
            tensor_check(weights, c10::CppTypeToScalarType<typename StoreType<B>::type>::value);
            tensor_check(pos_scale_x, at::kFloat);
            tensor_check(neg_scale_x, at::kFloat);
            tensor_check(scale_w, at::kFloat);

            cuda_device_check(weights);
            return real_quantized_gemm_dual_scaled<A, B, at::Tensor, at::Tensor>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w")
    );
}
