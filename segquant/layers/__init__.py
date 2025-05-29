from segquant.utils.extension import load_real_quant_fp8_ext

def build_ext_dict():
    d = {}
    ext_fp8 = load_real_quant_fp8_ext(required=False)
    d["fp8"] = {
        "gemm_scaled_fn": getattr(ext_fp8, "real_quantized_e4m3fy_gemm_scaled", None),
        "gemm_dual_scaled_fn": getattr(ext_fp8, "real_quantized_e4m3fy_gemm_dual_scaled", None),
    }

    return d

ext_dict = build_ext_dict()
