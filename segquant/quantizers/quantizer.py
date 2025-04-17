from abc import ABC, abstractmethod
import torch

class BaseQuantizer(ABC):
    @abstractmethod
    def quantize(self, x):
        pass

    def calibrate(self, x):
        pass

class FakeQuantizer(BaseQuantizer):
    def quantize(self, x):
        pass

    def calibrate(self, x):
        pass

    @staticmethod
    def dequantize(y_quantized: torch.Tensor, input_quantizer, weight_quantizer, input, weight) -> torch.Tensor:
        y_quantized = torch.clamp(y_quantized, min=input_quantizer.qmin, max=input_quantizer.qmax)

        scale_x = input_quantizer.scale
        zero_point_x = input_quantizer.zero_point
        scale_w = weight_quantizer.scale
        zero_point_w = weight_quantizer.zero_point

        if zero_point_w == 0 and zero_point_x == 0:
            y_dequant = y_quantized * scale_x * scale_w
            return y_dequant
        else:
            raise ValueError('Not implemented')

class QuantizerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(quantizer_cls):
            cls._registry[name] = quantizer_cls
            return quantizer_cls
        return wrapper

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def create(cls, name, **kwargs):
        quantizer_cls = cls.get(name)
        if quantizer_cls is None:
            raise ValueError(f"Quantizer '{name}' not found in registry.")
        return quantizer_cls(**kwargs)

@QuantizerRegistry.register("int8")
class IntQuantizer(FakeQuantizer):
    def __init__(self, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.qmin = -2 ** num_bits
        self.qmax = 2 ** num_bits - 1

        self.amax = None
        self.amin = None
        self.zero_point = None

    def calibrate(self, x):
        if self.symmetric:
            max_val = x.abs().max().item()
            if self.amax is None or self.amax < max_val:
                self.amax = max_val

            self.scale = self.amax / self.qmax
            self.zero_point = 0
        else:
            min_val = x.min().item()
            if self.amin is None or self.amin > min_val:
                self.amin = min_val
            max_val = x.max().item()
            if self.amax is None or self.amax < max_val:
                self.amax = max_val

            self.scale = (self.amax - self.amin) / (self.qmax - self.qmin)
            self.zero_point = int(round(self.qmin - self.amin / self.scale))

    def fake_quantize(self, x: torch.Tensor):
        x_int = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax)
        x_dequant = (x_int - self.zero_point) * self.scale
        return x_dequant

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quantize(x)

        x_int = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax)
        return x_int

    def __repr__(self):
        if self.symmetric:
            return f"IntQuantizer(num_bits={self.num_bits}, symmetric={self.symmetric}, amax={self.amax}, amin={self.amin}, scale={self.scale}, zero_point={self.zero_point})"
        else:
            return f"IntQuantizer(num_bits={self.num_bits}, symmetric={self.symmetric}, amax={self.amax}, scale={self.scale})"


@QuantizerRegistry.register("fpe4m3")
class FloatQuantizer(FakeQuantizer):
    def __init__(self, exp_bits=4, mant_bits=3, use_sign=True, clip_max=None):
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.use_sign = use_sign
        self.clip_max = clip_max

        self.exp_max = 2 ** exp_bits - 1
        self.mant_max = 2 ** mant_bits - 1
        self.bias = (2 ** (exp_bits - 1)) - 1

    def quantize(self, x):
        x_fp = x.clone()

        if self.clip_max is not None:
            x_fp = torch.clamp(x_fp, -self.clip_max, self.clip_max)

        if self.use_sign:
            sign = (x_fp < 0).to(torch.int32)
            x_fp = x_fp.abs()
        else:
            sign = torch.zeros_like(x_fp, dtype=torch.int32)

        eps = 1e-6
        exponent = torch.floor(torch.log2(x_fp + eps))
        mantissa = x_fp / (2 ** exponent + eps) - 1

        exponent_q = torch.clamp((exponent + self.bias).round(), 0, self.exp_max).to(torch.int32)
        mantissa_q = torch.clamp((mantissa * self.mant_max).round(), 0, self.mant_max).to(torch.int32)

        return sign, exponent_q, mantissa_q

    def dequantize(self, sign, exponent_q, mantissa_q):
        exponent = exponent_q - self.bias
        x_recon = (1 + mantissa_q.float() / self.mant_max) * (2 ** exponent.float())

        if self.use_sign:
            sign_factor = 1 - 2 * sign
            return sign_factor * x_recon
        else:
            return x_recon

    def __call__(self, x):
        sign, exp, mant = self.quantize(x)
        return self.dequantize(sign, exp, mant)

    def __repr__(self):
        return (f"FloatQuantizer(exp_bits={self.exp_bits}, "
                f"mant_bits={self.mant_bits}, use_sign={self.use_sign}, "
                f"clip_max={self.clip_max})")

@QuantizerRegistry.register("int16")
def int6_factory():
    return IntQuantizer(num_bits=16)

@QuantizerRegistry.register("int6")
def int6_factory():
    return IntQuantizer(num_bits=6)

@QuantizerRegistry.register("int4")
def int4_factory():
    return IntQuantizer(num_bits=4)

@QuantizerRegistry.register("fpe5m2")
def fp8e5m2_factory():
    return FloatQuantizer(exp_bits=5, mant_bits=2)