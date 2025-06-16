"""
This module provides utility functions for loading C++/CUDA extensions
and a specific extension for FP8 fake quantization.
"""
import os
import types
from torch.utils.cpp_extension import load
import torch

_loaded_extensions = {}

cutlass_path = os.environ.get('CUTLASS_PATH', '/usr/local/cutlass')
print(f"Use Cutlass Path [{cutlass_path}]")

supported_type = [
    'Wint8Aint8',
    'Wint4Aint8',
    'Wint4Aint4',
    'Wint4Afp16',
    'Wint8Afp16',
    'Wfpe4m3Afpe4m3',
    'Wfp16Afp16',
]


def load_extension(
    name: str,
    sources: list,
    include_dirs: list = None,
    verbose: bool = False,
    required: bool = False,
    **kwargs,
):
    if name in _loaded_extensions:
        if verbose:
            print(f"[INFO] Extension already loaded: {name}")
        return _loaded_extensions[name]

    try:
        if verbose:
            print(f"[INFO] Attempting to load extension: {name}")
            print(f"[INFO] Sources: {sources}")
            if include_dirs:
                print(f"[INFO] Include dirs: {include_dirs}")

        extension = load(
            name=name,
            sources=sources,
            extra_include_paths=include_dirs or [],
            verbose=verbose,
            **kwargs,
        )

        if verbose:
            print(f"[INFO] Successfully loaded extension: {name}")

        _loaded_extensions[name] = extension
        return extension

    except (RuntimeError, ImportError, TypeError) as e:
        if verbose:
            print(f"[ERROR] Failed to load extension: {name}\n{e}")
        if required:
            raise RuntimeError(f"Required extension '{name}' failed to load.") from e
        return None

def load_fake_quant_fp8_ext(verbose=False, required=False):
    """
    Load the fake quantization extension for FP8 quantization.
    Args:
        verbose (bool): If True, prints additional information during loading.
        required (bool): If True, raises an error if the extension fails to load.
    Returns:
        module: The loaded extension object, or None if it fails to load and `required` is False.
    """
    return load_extension(
        name="segquant_fake_quant_fp8",
        sources=[
            "segquant/src/fake_quant/quantizer_fp8.cpp",
            "segquant/src/fake_quant/quantizer_fp8.cu",
        ],
        verbose=verbose,
        required=required,
    )

def load_real_quant_fp8_ext(verbose=False, required=False):
    """
    Load the real quantization extension for FP8 quantization.
    Args:
        verbose (bool): If True, prints additional information during loading.
        required (bool): If True, raises an error if the extension fails to load.
    Returns:
        module: The loaded extension object, or None if it fails to load and `required` is False.
    """
    ext = load_extension(
        name="segquant_real_quant_fp8",
        sources=[
            "segquant/src/real_quant/quantized_fp8.cpp",
            "segquant/src/real_quant/real_gemm.cu",
        ],
        include_dirs=[
            f'{cutlass_path}/include'
        ],
        verbose=verbose,
        required=required,
        extra_cflags=['-DSEGQUANT_FP8'],
        extra_cuda_cflags=['-DSEGQUANT_FP8'],
    )

    def create_quantized_weights(self, x):
        return torch.empty_like(x, dtype=torch.uint8)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wfpe4m3Afpe4m3',)

def load_real_quant_int8_ext(verbose=False, required=False):
    """
    Load the real quantization extension for INT8 quantization.
    Args:
        verbose (bool): If True, prints additional information during loading.
        required (bool): If True, raises an error if the extension fails to load.
    Returns:
        module: The loaded extension object, or None if it fails to load and `required` is False.
    """
    ext = load_extension(
        name="segquant_real_quant_int8",
        sources=[
            "segquant/src/real_quant/quantized_int8.cpp",
            "segquant/src/real_quant/real_gemm.cu",
        ],
        include_dirs=[
            f'{cutlass_path}/include'
        ],
        verbose=verbose,
        required=required,
        extra_cflags=['-DSEGQUANT_INT8'],
        extra_cuda_cflags=['-DSEGQUANT_INT8'],
    )

    def create_quantized_weights(self, x):
        return torch.empty_like(x, dtype=torch.int8)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wint8Aint8',)

def load_real_quant_int4_ext(verbose=False, required=False):
    """
    Load the real quantization extension for INT4 quantization.
    Args:
        verbose (bool): If True, prints additional information during loading.
        required (bool): If True, raises an error if the extension fails to load.
    Returns:
        module: The loaded extension object, or None if it fails to load and `required` is False.
    """
    ext = load_extension(
        name="segquant_real_quant_int4",
        sources=[
            "segquant/src/real_quant/quantized_int4.cpp",
            "segquant/src/real_quant/real_gemm.cu",
        ],
        include_dirs=[
            f'{cutlass_path}/include'
        ],
        verbose=verbose,
        required=required,
        extra_cflags=['-DSEGQUANT_INT4'],
        extra_cuda_cflags=['-DSEGQUANT_INT4'],
    )

    def create_quantized_weights(self, x):
        num_elements = x.numel()
        num_bytes = (num_elements + 1) // 2
        return torch.empty(num_bytes, dtype=torch.uint8, device=x.device)

    ext.create_quantized_weights = types.MethodType(create_quantized_weights, ext)
    return ext, ('Wint4Aint4',)


def load_real_quant_mix_ext(verbose=False, required=False):
    """
    Load the real quantization extension for MIX quantization.
    Args:
        verbose (bool): If True, prints additional information during loading.
        required (bool): If True, raises an error if the extension fails to load.
    Returns:
        module: The loaded extension object, or None if it fails to load and `required` is False.
    """
    ext = load_extension(
        name="segquant_real_quant_mix",
        sources=[
            "segquant/src/real_quant/quantized_mix.cpp",
            "segquant/src/real_quant/real_gemm.cu",
        ],
        include_dirs=[
            f'{cutlass_path}/include'
        ],
        verbose=verbose,
        required=required,
        extra_cflags=['-DSEGQUANT_MIX'],
        extra_cuda_cflags=['-DSEGQUANT_MIX'],
    )

    return ext, supported_type