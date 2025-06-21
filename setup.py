from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUTLASS path from environment
cutlass_path = os.environ.get('CUTLASS_PATH', '/usr/local/cutlass')

# Define all CUDA extensions
cuda_extensions = [
    # Fake quantization FP8
    CUDAExtension(
        name='segquant_fake_quant_fp8',
        sources=[
            'segquant/src/fake_quant/quantizer_fp8.cpp',
            'segquant/src/fake_quant/quantizer_fp8.cu',
        ],
        include_dirs=[
            torch.utils.cpp_extension.include_paths(),
            'segquant/src/real_quant',
        ],
        libraries=['cudart'],
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    ),
    
    # Real quantization FP8
    CUDAExtension(
        name='segquant_real_quant_fp8',
        sources=[
            'segquant/src/real_quant/quantized_fp8.cpp',
            'segquant/src/real_quant/real_gemm.cu',
        ],
        include_dirs=[
            torch.utils.cpp_extension.include_paths(),
            'segquant/src/real_quant',
            f'{cutlass_path}/include',
        ],
        libraries=['cudart', 'cublas'],
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
            'cxx': ['-O3', '-DSEGQUANT_FP8'],
            'nvcc': ['-O3', '--use_fast_math', '-DSEGQUANT_FP8']
        }
    ),
    
    # Real quantization INT8
    CUDAExtension(
        name='segquant_real_quant_int8',
        sources=[
            'segquant/src/real_quant/quantized_int8.cpp',
            'segquant/src/real_quant/real_gemm.cu',
        ],
        include_dirs=[
            torch.utils.cpp_extension.include_paths(),
            'segquant/src/real_quant',
            f'{cutlass_path}/include',
        ],
        libraries=['cudart', 'cublas'],
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
            'cxx': ['-O3', '-DSEGQUANT_INT8'],
            'nvcc': ['-O3', '--use_fast_math', '-DSEGQUANT_INT8']
        }
    ),
    
    # Real quantization INT4
    CUDAExtension(
        name='segquant_real_quant_int4',
        sources=[
            'segquant/src/real_quant/quantized_int4.cpp',
            'segquant/src/real_quant/real_gemm.cu',
        ],
        include_dirs=[
            torch.utils.cpp_extension.include_paths(),
            'segquant/src/real_quant',
            f'{cutlass_path}/include',
        ],
        libraries=['cudart', 'cublas'],
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
            'cxx': ['-O3', '-DSEGQUANT_INT4'],
            'nvcc': ['-O3', '--use_fast_math', '-DSEGQUANT_INT4']
        }
    ),
    
    # Real quantization MIX
    CUDAExtension(
        name='segquant_real_quant_mix',
        sources=[
            'segquant/src/real_quant/quantized_mix.cpp',
            'segquant/src/real_quant/real_gemm.cu',
        ],
        include_dirs=[
            torch.utils.cpp_extension.include_paths(),
            'segquant/src/real_quant',
            f'{cutlass_path}/include',
            'segquant/src/real_quant/cutlass_extensions/include',
        ],
        libraries=['cudart', 'cublas'],
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
            'cxx': ['-O3', '-DSEGQUANT_MIX'],
            'nvcc': ['-O3', '--use_fast_math', '-DSEGQUANT_MIX']
        }
    ),
]

setup(
    name='segquant',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cuda_extensions,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
    ],
) 