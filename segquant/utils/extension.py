import os
from torch.utils.cpp_extension import load
from typing import Optional

_loaded_extensions = {}


def load_extension(
    name: str, sources: list, verbose: bool = False, required: bool = False
):
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

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to load extension: {name}\n{e}")

        if required:
            raise RuntimeError(f"Required extension '{name}' failed to load.") from e
        else:
            return None


def load_fake_quant_fp8_ext(verbose=False, required=False):
    return load_extension(
        name="segquant_fake_quant_fp8",
        sources=[
            "segquant/src/fake_quant/quantizer_fp8.cpp",
            "segquant/src/fake_quant/quantizer_fp8.cu",
        ],
        verbose=verbose,
        required=required,
    )
