#include <ATen/ATen.h>
#include <torch/extension.h>

template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, float scale_w);
template<>
void real_quantized_quantize_weights<int8_t>(at::Tensor weights, at::Tensor outputs, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_scaled<int8_t>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);

template<typename T>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
template<>
at::Tensor real_quantized_gemm_dual_scaled<int8_t>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, float scale_w) {
            TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(outputs.is_contiguous(), "output must be contiguous");
            TORCH_CHECK(outputs.is_cuda(), "output must be a CUDA tensor");
            TORCH_CHECK(outputs.dtype() == at::kChar, "output must be int8");
            real_quantized_quantize_weights<int8_t>(weights, outputs, scale_w);
        },
        "Quantize weights to int8 format",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w").noconvert()
    );

    m.def("real_quantized_quantize_weights",
        [](at::Tensor weights, at::Tensor outputs, at::Tensor scale_w) {
            TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

            TORCH_CHECK(outputs.is_contiguous(), "output must be contiguous");
            TORCH_CHECK(outputs.is_cuda(), "output must be a CUDA tensor");
            TORCH_CHECK(outputs.dtype() == at::kChar, "output must be int8");

            TORCH_CHECK(scale_w.is_contiguous(), "scale_w must be contiguous");
            TORCH_CHECK(scale_w.is_cuda(), "scale_w must be a CUDA tensor");
            TORCH_CHECK(scale_w.dtype() == at::kFloat, "scale_w must be Float");

            real_quantized_quantize_weights_axis<int8_t>(weights, outputs, scale_w);
        },
        "Quantize weights to int8 format with axis",
        py::arg("weights"),
        py::arg("outputs"),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kChar, "weights tensor must be int8");

            return real_quantized_gemm_scaled<int8_t>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x").noconvert(),
        py::arg("scale_w").noconvert()
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, float scale_x, at::Tensor scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kChar, "weights tensor must be int8");

            TORCH_CHECK(scale_w.is_contiguous(), "scale_w must be contiguous");
            TORCH_CHECK(scale_w.is_cuda(), "scale_w must be a CUDA tensor");
            TORCH_CHECK(scale_w.dtype() == at::kFloat, "scale_w must be Float");

            return real_quantized_gemm_scaled<int8_t>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x").noconvert(),
        py::arg("scale_w")
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, at::Tensor scale_x, float scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kChar, "weights tensor must be int8");

            TORCH_CHECK(scale_x.is_contiguous(), "scale_x must be contiguous");
            TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");
            TORCH_CHECK(scale_x.dtype() == at::kFloat, "scale_x must be Float");

            return real_quantized_gemm_scaled<int8_t>(inputs, weights, scale_x, scale_w);
        },
        "Run scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("scale_x"),
        py::arg("scale_w").noconvert()
    );

    m.def("real_quantized_gemm_scaled",
        [](at::Tensor inputs, at::Tensor weights, at::Tensor scale_x, at::Tensor scale_w) {
            TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
            TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
            TORCH_CHECK(weights.dtype() == at::kChar, "weights tensor must be int8");

            TORCH_CHECK(scale_x.is_contiguous(), "scale_x must be contiguous");
            TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");
            TORCH_CHECK(scale_x.dtype() == at::kFloat, "scale_x must be Float");

            TORCH_CHECK(scale_w.is_contiguous(), "scale_w must be contiguous");
            TORCH_CHECK(scale_w.is_cuda(), "scale_w must be a CUDA tensor");
            TORCH_CHECK(scale_w.dtype() == at::kFloat, "scale_w must be Float");

            return real_quantized_gemm_scaled<int8_t>(inputs, weights, scale_x, scale_w);
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
            TORCH_CHECK(weights.dtype() == at::kChar, "weights tensor must be int8");

            return real_quantized_gemm_dual_scaled<int8_t>(inputs, weights, pos_scale_x, neg_scale_x, scale_w);
        },
        "Run dual scaled int8 GEMM",
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("pos_scale_x"),
        py::arg("neg_scale_x"),
        py::arg("scale_w")
    );
}