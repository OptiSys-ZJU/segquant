#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cutlass/numeric_types.h>

template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, float scale_w);
template<>
void real_quantized_quantize_weights<cutlass::int4b_t>(at::Tensor weights, at::Tensor outputs, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_scaled<cutlass::int4b_t>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_dual_scaled<cutlass::int4b_t>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, float scale_w) {
            TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(weights.numel() % 2 == 0, "Quantization to int4 requires the number of elements to be even");
            TORCH_CHECK(outputs.is_contiguous(), "output must be contiguous");
            TORCH_CHECK(outputs.is_cuda(), "output must be a CUDA tensor");
            TORCH_CHECK(outputs.dtype() == at::kByte, "output must be uint8");
            TORCH_CHECK(outputs.numel() * 2 == weights.numel(), "output numel must be half of weight");
            real_quantized_quantize_weights<cutlass::int4b_t>(weights, outputs, scale_w);
        },
        "Quantize weights to int4 format",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be uint8");

            return real_quantized_gemm_scaled<cutlass::int4b_t>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_dual_scaled",
        [](at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kByte, "weights tensor must be int8");

            return real_quantized_gemm_dual_scaled<cutlass::int4b_t>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w")
    );
}