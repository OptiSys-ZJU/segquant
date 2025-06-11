from .utils.extension import load_fake_quant_fp8_ext, load_real_quant_fp8_ext, load_real_quant_int8_ext, load_real_quant_int4_ext

# Load extensions on-demand rather than forcing compilation at import time
# Comment out extensions you don't need to avoid compilation overhead
#load_fake_quant_fp8_ext(verbose=True, required=True)
#load_real_quant_fp8_ext(verbose=True, required=True)
load_real_quant_int8_ext(verbose=True, required=True)  # Most common for testing
#load_real_quant_int4_ext(verbose=True, required=True)