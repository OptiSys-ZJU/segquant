#include <cuda_fp8.h>
#include "seg_utils.h"

INSTANTIATE_QUANTIZE_WEIGHTS(__nv_fp8_e4m3)
#define SPECIALIZATION(A, W, SX, SW) \
    template<> at::Tensor real_quantized_gemm_scaled<A, W, SX, SW>(at::Tensor, at::Tensor, SX, SW); \
    template<> at::Tensor real_quantized_gemm_dual_scaled<A, W, SX, SW>(at::Tensor, at::Tensor, SX, SX, SW);

#define EXPAND_SW(A, W, SX) \
    SPECIALIZATION(A, W, SX, float) \
    SPECIALIZATION(A, W, SX, at::Tensor)

#define EXPAND_SX(A, W) \
    EXPAND_SW(A, W, float) \
    EXPAND_SW(A, W, at::Tensor)

#define X(A, W) EXPAND_SX(A, W)
X(__nv_fp8_e4m3, __nv_fp8_e4m3)
#undef X

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_quantweight_module<__nv_fp8_e4m3>(m);
    register_gemm_module<__nv_fp8_e4m3, __nv_fp8_e4m3>(m, "Wfpe4m3Afpe4m3");
}