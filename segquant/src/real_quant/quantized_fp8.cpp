#include <cuda_fp8.h>
#include "seg_utils.h"

INSTANTIATE_QUANTIZE_WEIGHTS(__nv_fp8_e4m3)
#define SPECIALIZATION(A, W) \
    template<> at::Tensor real_quantized_gemm_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor); \
    template<> at::Tensor real_quantized_gemm_dual_scaled<A, W>(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);

#define X(A, W) SPECIALIZATION(A, W)
X(__nv_fp8_e4m3, __nv_fp8_e4m3)
#undef X

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_quantweight_module<__nv_fp8_e4m3>(m);
    register_gemm_module<__nv_fp8_e4m3, __nv_fp8_e4m3>(m, "Wfpe4m3Afpe4m3");
}