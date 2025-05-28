from .utils.extension import load_fake_quant_fp8_ext, load_real_quant_fp8_ext

load_fake_quant_fp8_ext(verbose=False, required=False)
load_real_quant_fp8_ext(verbose=True, required=True)
