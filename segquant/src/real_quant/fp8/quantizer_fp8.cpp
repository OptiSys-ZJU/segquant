#include <ATen/ATen.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
 
at::Tensor real_quantized_e4m3fy_quantize_weights(at::Tensor weights, float scale_w);

at::Tensor real_quantized_e4m3fy_gemm_scaled(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
 
at::Tensor real_quantized_e4m3fy_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("real_quantized_e4m3fy_quantize_weights",
        [](at::Tensor weights, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            return real_quantized_e4m3fy_quantize_weights(weights, scale_w);
        },
        "Quantize weights to E4M3 format",
        py::arg("weights"),
        py::arg("scale_w")
    );

    m.def("real_quantized_e4m3fy_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be uint8");

            return real_quantized_e4m3fy_gemm_scaled(inputs, weights, scale_x, scale_w);
        },
        "Run scaled FP8 GEMM with E4M3 quantization",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w")
    );

    m.def("real_quantized_e4m3fy_gemm_dual_scaled",
        [](at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be uint8");

            return real_quantized_e4m3fy_gemm_dual_scaled(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled FP8 GEMM with E4M3 quantization",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w")
    );
}