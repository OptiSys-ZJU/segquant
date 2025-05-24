from torch.utils.cpp_extension import load

fake_quant_fp8_extension = None

def load_fake_quant_fp8_ext():
    global fake_quant_fp8_extension
    if fake_quant_fp8_extension is None:
        try:
            fake_quant_fp8_extension = load(
                name="segquant_fake_quant_fp8",
                sources=["segquant/src/fake_quant/quantizer_fp8.cpp", "segquant/src/fake_quant/quantizer_fp8.cu"],
            )
        except Exception as e:
            print(e)
    return fake_quant_fp8_extension