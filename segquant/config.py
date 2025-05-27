"""
This module defines configuration settings and enumerations for quantization
and model optimization. It includes data types, optimization methods,
segmentation patterns for linear layers, and a default quantization configuration.

Classes:
    DType: Enum representing data types used in quantization and model optimization.
    Optimum: Enum representing optimization methods for quantization.
    SegPattern: Enum representing patterns for segmenting linear layers in quantization.

Variables:
    default_quantize_config: A dictionary containing the default configuration
    for quantization, including data types, optimization methods, and segmentation patterns.
"""

from enum import Enum

class DType(Enum):
    """Data types used in quantization and model optimization."""
    INT4 = "int4"
    INT6 = "int6"
    INT8 = "int8"
    FP8E5M2 = "fpe5m2"
    FP8E4M3 = "fpe4m3"
    FP16 = "fp16"


class Optimum(Enum):
    """Optimization methods for quantization."""
    DEFAULT = "default"
    SMOOTH = "smooth"
    SVD = "svd"


class SegPattern(Enum):
    """Patterns for segmenting linear layers in quantization."""
    LINEAR2CHUNK = "linear_to_chunk"
    LINEAR2SPLIT = "linear_to_split"
    CONCAT2LINEAR = "concat_to_linear"
    STACK2LINEAR = "stack_to_linear"
    ACTIVATION2LINEAR = "activation_to_linear"

    @classmethod
    def seg(cls):
        """Return patterns that are used for segmenting linear layers."""
        return [
            SegPattern.LINEAR2CHUNK,
            SegPattern.LINEAR2SPLIT,
            SegPattern.CONCAT2LINEAR,
            SegPattern.STACK2LINEAR,
        ]

    @classmethod
    def all(cls):
        """Return all patterns, including those that do not segment linear layers."""
        return [
            cls.LINEAR2CHUNK,
            cls.LINEAR2SPLIT,
            cls.CONCAT2LINEAR,
            cls.STACK2LINEAR,
            cls.ACTIVATION2LINEAR,
        ]


default_quantize_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": True,
        "real_quant": False,
        "input_axis": None,
        "weight_axis": None,
        "search_patterns": SegPattern.all(),
    },
}
