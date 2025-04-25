from enum import Enum

class DType(Enum):
    INT4 = 'int4'
    INT4SVD = 'int4-svd'
    INT8 = 'int8'
    INT8SMOOTH = 'int8-smooth'
    FP8E5M2 = 'fpe5m2'
    FP8E4M3 = 'fpe4m3'
    FP16 = 'fp16'

class SegPattern(Enum):
    Linear2Chunk = 'linear_to_chunk'
    Linear2Split = 'linear_to_split'
    Concat2Linear = 'concat_to_linear'
    Stack2Linear = 'stack_to_linear'
    Activation2Linear = 'activation_to_linear'

    def all():
        return [SegPattern.Linear2Chunk, SegPattern.Linear2Split, SegPattern.Concat2Linear, SegPattern.Stack2Linear, SegPattern.Activation2Linear]

default_quantize_config = {
    "default": {
        "enable": True,
        "dtype": DType.INT8,
        "seglinear": True,
        'search_patterns': SegPattern.all(),
    },
}