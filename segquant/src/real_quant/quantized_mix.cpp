#include <cuda_fp8.h>
#include <cutlass/util/device_memory.h>
#include "seg_utils.h"

// #define MIX_AW_PAIRS \
//     X(int8_t, int8_t) \
//     X(int8_t, cutlass::int4b_t) \
//     X(cutlass::int4b_t, cutlass::int4b_t) \
//     X(at::Half, int8_t) \
//     X(__nv_fp8_e4m3, __nv_fp8_e4m3)

#define MIX_AW_PAIRS \
    X(int8_t, int8_t) \
    X(__nv_fp8_e4m3, __nv_fp8_e4m3) \
    X(cutlass::int4b_t, cutlass::int4b_t) \
    X(int8_t, cutlass::int4b_t) \
    X(at::Half, int8_t)

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
MIX_AW_PAIRS
#undef X


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // register_quantweight_module is unnecessary
    register_gemm_module<int8_t, int8_t>(m, "Wint8Aint8");
    register_gemm_module<__nv_fp8_e4m3, __nv_fp8_e4m3>(m, "Wfpe4m3Afpe4m3");
    register_gemm_module<cutlass::int4b_t, cutlass::int4b_t>(m, "Wint4Aint4");
    register_gemm_module<int8_t, cutlass::int4b_t>(m, "Wint4Aint8");
    register_gemm_module<at::Half, int8_t>(m, "Wint8Afp16");

    // register_gemm_module<at::Half, cutlass::int4b_t>(m, "Wint4Afp16");
}