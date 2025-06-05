#include <ATen/ATen.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, float scale_w);
template<>
void real_quantized_quantize_weights<__nv_fp8_e4m3>(at::Tensor weights, at::Tensor outputs, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_scaled<__nv_fp8_e4m3>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_dual_scaled<__nv_fp8_e4m3>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, float scale_w) {
            TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(outputs.is_contiguous(), "output must be contiguous");
            TORCH_CHECK(outputs.is_cuda(), "output must be a CUDA tensor");
            TORCH_CHECK(outputs.dtype() == at::kByte, "output must be uint8");
            real_quantized_quantize_weights<__nv_fp8_e4m3>(weights, outputs, scale_w);
        },
        "Quantize weights to E4M3 format",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be uint8");

            return real_quantized_gemm_scaled<__nv_fp8_e4m3>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled FP8 GEMM with E4M3 quantization",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_dual_scaled",
        [](at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be uint8");

            return real_quantized_gemm_dual_scaled<__nv_fp8_e4m3>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled FP8 GEMM with E4M3 quantization",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w")
    );
}