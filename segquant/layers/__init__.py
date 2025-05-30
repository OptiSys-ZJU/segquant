from segquant.utils.extension import load_real_quant_fp8_ext, load_real_quant_int4_ext, load_real_quant_int8_ext

def build_ext_dict():
    d = {}
    ext_fp8 = load_real_quant_fp8_ext(required=False)
    d["fpe4m3"] = {
        "gemm_scaled_fn": getattr(ext_fp8, "real_quantized_gemm_scaled", None),
        "gemm_dual_scaled_fn": getattr(ext_fp8, "real_quantized_gemm_dual_scaled", None),
    }

    ext_int8 = load_real_quant_int8_ext(required=False)
    d["int8"] = {
        "gemm_scaled_fn": getattr(ext_int8, "real_quantized_gemm_scaled", None),
        "gemm_dual_scaled_fn": getattr(ext_int8, "real_quantized_gemm_dual_scaled", None),
    }

    ext_int4 = load_real_quant_int4_ext(required=False)
    d["int4"] = {
        "gemm_scaled_fn": getattr(ext_int4, "real_quantized_gemm_scaled", None),
        "gemm_dual_scaled_fn": getattr(ext_int4, "real_quantized_gemm_dual_scaled", None),
    }

    return d

ext_dict = build_ext_dict()
