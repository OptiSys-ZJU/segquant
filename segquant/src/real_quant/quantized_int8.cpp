#include "seg_utils.h"

INSTANTIATE_QUANTIZE_WEIGHTS(int8_t)

#define SPECIALIZATION(A, W) \
    template<> at::Tensor real_quantized_gemm_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor); \
    template<> at::Tensor real_quantized_gemm_dual_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);

#define X(A, W) SPECIALIZATION(A, W)
X(int8_t, int8_t)
#undef X


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_quantweight_module<int8_t>(m);
    register_gemm_module<int8_t, int8_t>(m, "Wint8Aint8");
}