from enum import Enum

class DType(Enum):
    INT4 = 'int4'
    INT6 = 'int6'
    INT8 = 'int8'
    FP8E5M2 = 'fpe5m2'
    FP8E4M3 = 'fpe4m3'
    FP16 = 'fp16'

class Optimum(Enum):
    DEFAULT = 'default'
    SMOOTH = 'smooth'
    SVD = 'svd'

class SegPattern(Enum):
    Linear2Chunk = 'linear_to_chunk'
    Linear2Split = 'linear_to_split'
    Concat2Linear = 'concat_to_linear'
    Stack2Linear = 'stack_to_linear'
    Activation2Linear = 'activation_to_linear'

    def seg():
        return [SegPattern.Linear2Chunk, SegPattern.Linear2Split, SegPattern.Concat2Linear, SegPattern.Stack2Linear]

    def all():
        return [SegPattern.Linear2Chunk, SegPattern.Linear2Split, SegPattern.Concat2Linear, SegPattern.Stack2Linear, SegPattern.Activation2Linear]

default_quantize_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": True,
        "input_axis": None,
        "weight_axis": None,
        'search_patterns': SegPattern.all(),
    },
}