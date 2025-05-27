"""
This module provides utility functions for loading C++/CUDA extensions
and a specific extension for FP8 fake quantization.
"""

from torch.utils.cpp_extension import load

_loaded_extensions = {}


def load_extension(
    name: str, sources: list, verbose: bool = False, required: bool = False
):
    """
    Load a C++/CUDA extension with the given name and sources.
    If the extension is already loaded, it returns the existing instance.
    If the extension fails to load and `required` is True, it raises an error.
    If `verbose` is True, it prints additional information about the loading process.
    """
    if name in _loaded_extensions:
        if verbose:
            print(f"[INFO] Extension already loaded: {name}")
        return _loaded_extensions[name]

    try:
        if verbose:
            print(f"[INFO] Attempting to load extension: {name}")
            print(f"[INFO] Sources: {sources}")

        extension = load(name=name, sources=sources, verbose=verbose)

        if verbose:
            print(f"[INFO] Successfully loaded extension: {name}")

        _loaded_extensions[name] = extension
        return extension

    except (RuntimeError, ImportError) as e:
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
    return load_extension(
        name="segquant_real_quant_fp8",
        sources=[
            "segquant/src/real_quant/quantizer_fp8.cpp",
            "segquant/src/real_quant/quantized_fp8_gemm.cpp",
        ],
        verbose=verbose,
        required=required,
    )
