from segquant.utils.extension import load_real_quant_fp8_ext, load_real_quant_int4_ext, load_real_quant_int8_ext, load_real_quant_mix_ext

def build_ext_dict():
    d = {}

    def add_dict(func):
        ext, prefix = func(required=False)
        for p in prefix:
            d[p] = {
                "gemm_scaled_fn": getattr(ext, f"{p}_real_quantized_gemm_scaled", None),
                "gemm_dual_scaled_fn": getattr(ext, f"{p}_real_quantized_gemm_dual_scaled", None),
            }

    add_dict(load_real_quant_fp8_ext)
    add_dict(load_real_quant_int8_ext)
    add_dict(load_real_quant_int4_ext)
    # add_dict(load_real_quant_mix_ext)

    return d

ext_dict = build_ext_dict()
