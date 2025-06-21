"""
This module provides utility functions for loading pre-built C++/CUDA extensions.
Use this instead of extension.py when extensions are pre-built during installation.
"""
import os
import types
import torch

_loaded_extensions = {}

cutlass_path = os.environ.get('CUTLASS_PATH', '/usr/local/cutlass')
print(f"Use Cutlass Path [{cutlass_path}]")

supported_type = [
    'Wint8Aint8',
    'Wint4Aint8',
    'Wint4Aint4',
    'Wint8Afp16',
    'Wfpe4m3Afpe4m3',
    'Wfp16Afp16',
]


def load_prebuilt_extension(name: str, verbose: bool = False, required: bool = False):
    """Load a pre-built extension."""
    if name in _loaded_extensions:
        if verbose:
            print(f"[INFO] Extension already loaded: {name}")
        return _loaded_extensions[name]

    try:
        if verbose:
            print(f"[INFO] Loading pre-built extension: {name}")
        
        # Import the pre-built extension
        extension = __import__(name)
        
        if verbose:
            print(f"[INFO] Successfully loaded pre-built extension: {name}")

        _loaded_extensions[name] = extension
        return extension

    except (ImportError, AttributeError) as e:
        if verbose:
            print(f"[ERROR] Failed to load pre-built extension: {name}\n{e}")
        if required:
            raise RuntimeError(f"Required pre-built extension '{name}' failed to load.") from e
        return None


def load_fake_quant_fp8_ext(verbose=False, required=False):
    """
    Load the pre-built fake quantization extension for FP8 quantization.
    """
    return load_prebuilt_extension(
        name="segquant_fake_quant_fp8",
        verbose=verbose,
        required=required,
    )


def load_real_quant_fp8_ext(verbose=False, required=False):
    """
    Load the pre-built real quantization extension for FP8 quantization.
    """
    ext = load_prebuilt_extension(
        name="segquant_real_quant_fp8",
        verbose=verbose,
        required=required,
    )
    
    if ext is None:
        return None

    def create_quantized_weights(self, x):
        return torch.empty_like(x, dtype=torch.uint8)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wfpe4m3Afpe4m3',)


def load_real_quant_int8_ext(verbose=False, required=False):
    """
    Load the pre-built real quantization extension for INT8 quantization.
    """
    ext = load_prebuilt_extension(
        name="segquant_real_quant_int8",
        verbose=verbose,
        required=required,
    )
    
    if ext is None:
        return None

    def create_quantized_weights(self, x):
        return torch.empty_like(x, dtype=torch.int8)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wint8Aint8',)


def load_real_quant_int4_ext(verbose=False, required=False):
    """
    Load the pre-built real quantization extension for INT4 quantization.
    """
    ext = load_prebuilt_extension(
        name="segquant_real_quant_int4",
        verbose=verbose,
        required=required,
    )
    
    if ext is None:
        return None

    def create_quantized_weights(self, x):
        num_elements = x.numel()
        num_bytes = (num_elements + 1) // 2
        return torch.empty(num_bytes, dtype=torch.uint8, device=x.device)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wint4Aint4',)


def load_real_quant_mix_ext(verbose=False, required=False):
    """
    Load the pre-built real quantization extension for MIX quantization.
    """
    ext = load_prebuilt_extension(
        name="segquant_real_quant_mix",
        verbose=verbose,
        required=required,
    )
    
    if ext is None:
        return None

    def create_quantized_weights(self, x):
        return torch.empty_like(x, dtype=torch.uint8)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wint8Aint8', 'Wint4Aint8', 'Wint4Aint4', 'Wint8Afp16', 'Wfpe4m3Afpe4m3') 