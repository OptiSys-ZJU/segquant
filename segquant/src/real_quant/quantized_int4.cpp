#include <cutlass/util/device_memory.h>
#include "seg_utils.h"

INSTANTIATE_QUANTIZE_WEIGHTS(cutlass::int4b_t)
#define SPECIALIZATION(A, W) \
    template<> at::Tensor real_quantized_gemm_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor); \
    template<> at::Tensor real_quantized_gemm_dual_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);

#define X(A, W) SPECIALIZATION(A, W)
X(cutlass::int4b_t, cutlass::int4b_t)
#undef X

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_quantweight_module<cutlass::int4b_t>(m);
    register_gemm_module<cutlass::int4b_t, cutlass::int4b_t>(m, "Wint4Aint4");
}