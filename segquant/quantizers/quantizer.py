from abc import ABC, abstractmethod
from segquant.utils.extension import load_fake_quant_fp8_ext
import torch

class BaseQuantizer(ABC):
    @abstractmethod
    def quantize(self, x):
        pass

    def calibrate(self, x):
        pass

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
class IntQuantizer(BaseQuantizer):
    def __init__(self, num_bits=8, symmetric=True, axis=None, dual_scale=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.axis = axis
        self.dual_scale = dual_scale

        self.qmin = -2 ** (num_bits - 1)
        self.qmax = 2 ** (num_bits - 1) - 1

        # init
        self.amax = None
        self.amin = None
        self.zero_point = None

        # for dual-scale
        self.neg_amax = None
        self.pos_amax = None
        self.neg_scale = None
        self.pos_scale = None

    def calibrate(self, x: torch.Tensor):
        epsilon = 1.0 / (1 << 24)

        if self.symmetric:
            if self.dual_scale:
                neg_x = x[x < 0].abs()
                pos_x = x[x > 0].abs()

                if self.axis is None:
                    neg_max = neg_x.max().item() if neg_x.numel() > 0 else 0.0
                    pos_max = pos_x.max().item() if pos_x.numel() > 0 else 0.0

                    self.neg_amax = max(self.neg_amax or 0.0, neg_max)
                    self.pos_amax = max(self.pos_amax or 0.0, pos_max)

                    self.neg_scale = self.qmin / (-self.neg_amax) if self.neg_amax > epsilon else 1.0
                    self.pos_scale = self.qmax / self.pos_amax if self.pos_amax > epsilon else 1.0
                    self.zero_point = 0
                else:
                    neg_max = neg_x.amax(dim=self.axis, keepdim=False) if neg_x.numel() > 0 else torch.zeros(x.size(self.axis), device=x.device)
                    pos_max = pos_x.amax(dim=self.axis, keepdim=False) if pos_x.numel() > 0 else torch.zeros(x.size(self.axis), device=x.device)

                    if self.neg_amax is None:
                        self.neg_amax = neg_max
                        self.pos_amax = pos_max
                    else:
                        self.neg_amax = torch.maximum(self.neg_amax, neg_max)
                        self.pos_amax = torch.maximum(self.pos_amax, pos_max)

                    self.neg_scale = self.qmin / (-self.neg_amax.clamp(min=epsilon))
                    self.pos_scale = self.qmax / self.pos_amax.clamp(min=epsilon)
                    self.zero_point = 0

            else:
                if self.axis is None:
                    max_val = x.abs().max().item()
                    self.amax = max(self.amax or 0.0, max_val)
                    self.scale = self.qmax / self.amax if self.amax > epsilon else 1.0
                    self.zero_point = 0
                else:
                    max_val = x.abs().amax(dim=self.axis, keepdim=False)
                    self.amax = torch.maximum(self.amax, max_val) if self.amax is not None else max_val
                    self.scale = self.qmax / self.amax.clamp(min=epsilon)
                    self.zero_point = 0
        else:
            # todo: scale = q/a
            if self.axis is None:
                min_val = x.min().item()
                max_val = x.max().item()
                self.amin = min(self.amin if self.amin is not None else min_val, min_val)
                self.amax = max(self.amax if self.amax is not None else max_val, max_val)
                self.scale = (self.amax - self.amin) / (self.qmax - self.qmin)
                self.zero_point = int(round(self.qmin - self.amin / self.scale))
            else:
                min_val = x.amin(dim=self.axis, keepdim=False)
                max_val = x.amax(dim=self.axis, keepdim=False)
                self.amin = torch.minimum(self.amin, min_val) if self.amin is not None else min_val
                self.amax = torch.maximum(self.amax, max_val) if self.amax is not None else max_val
                self.scale = (self.amax - self.amin) / (self.qmax - self.qmin)
                self.zero_point = (self.qmin - self.amin / self.scale).round().to(torch.int)
    
    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.symmetric and self.dual_scale:
            if self.axis is not None:
                shape = [1] * x.dim()
                shape[self.axis] = -1
                pos_scale = self.pos_scale.view(shape)
                neg_scale = self.neg_scale.view(shape)
            else:
                pos_scale = self.pos_scale
                neg_scale = self.neg_scale

            x_quant = torch.where(
                x >= 0,
                torch.clamp(torch.round(x * pos_scale), 0, self.qmax),
                torch.clamp(torch.round(x * neg_scale), self.qmin, 0)
            )
            x_dequant = torch.where(
                x >= 0,
                x_quant / pos_scale,
                x_quant / neg_scale
            )
            return x_dequant.to(x.dtype)

        else:
            if self.axis is not None:
                shape = [1] * x.dim()
                shape[self.axis] = -1
                scale = self.scale.view(shape)
                zero_point = self.zero_point.view(shape) if isinstance(self.zero_point, torch.Tensor) else self.zero_point
                scale = scale.T
            else:
                scale = self.scale
                zero_point = self.zero_point
            
            x_int = torch.round(x * scale) + zero_point
            x_int = torch.clamp(x_int, self.qmin, self.qmax)
            x_dequant = (x_int - zero_point) / scale
            return x_dequant.to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quantize(x)

    def __repr__(self):
        if self.symmetric:
            if self.dual_scale:
                return (f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, dual_scale=True, axis={self.axis}, "
                        f"neg_amax={self.neg_amax:.4f}, pos_amax={self.pos_amax:.4f})")
            else:
                if isinstance(self.amax, torch.Tensor):
                    amin = self.amax.min().item()
                    amax = self.amax.max().item()
                    amax_str = f"[{amin:.4f}, {amax:.4f}]"
                    # amax_str = ', '.join([f"{x:.4f}" for x in self.amax.tolist()])
                    return (f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, dual_scale=False, axis={self.axis}, "
                            f"amax=[{amax_str}])")
                else:
                    return (f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, dual_scale=False, axis={self.axis}, "
                            f"amax={self.amax:.4f})")
        else:
            return (f"IntQuantizer(num_bits={self.num_bits}, symmetric=False, axis={self.axis}, "
                    f"amin={self.amin:.4f}, amax={self.amax:.4f}, zero_point={self.zero_point:.4f})")

@QuantizerRegistry.register("fpe4m3")
class FloatQuantizer:
    def __init__(self, exp_bits=4, mant_bits=3, axis=None, dual_scale=False):
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.axis = axis
        self.dual_scale = dual_scale

        if exp_bits == 4 and mant_bits == 3:
            # nvidia's fpe4m3
            self.fp_max = 448.0
            self.fp_min = (2 ** (-6)) * (240 / 448.0)
        else:
            self.fp_max = (2 ** (2 ** exp_bits - 2)) * (2 - 2 ** (-mant_bits))
            self.fp_min = 2 ** (1 - (2 ** (exp_bits - 1)))

        self.scale = None
        self.zero_point = 0

        self.neg_scale = None
        self.pos_scale = None

        self.neg_amax = None
        self.pos_amax = None
        self.amax = None

    def calibrate(self, x: torch.Tensor):
        epsilon = 1.0 / (1 << 24)

        if self.dual_scale:
            neg_x = x[x < 0].abs()
            pos_x = x[x > 0].abs()

            if self.axis is None:
                neg_max = neg_x.max().item() if neg_x.numel() > 0 else 0.0
                pos_max = pos_x.max().item() if pos_x.numel() > 0 else 0.0

                self.neg_amax = max(self.neg_amax or 0.0, neg_max)
                self.pos_amax = max(self.pos_amax or 0.0, pos_max)

                self.neg_scale = self.fp_max / self.neg_amax if self.neg_amax > epsilon else 1.0
                self.pos_scale = self.fp_max / self.pos_amax if self.pos_amax > epsilon else 1.0
            else:
                neg_max = neg_x.amax(dim=self.axis, keepdim=False) if neg_x.numel() > 0 else torch.zeros(x.size(self.axis), device=x.device)
                pos_max = pos_x.amax(dim=self.axis, keepdim=False) if pos_x.numel() > 0 else torch.zeros(x.size(self.axis), device=x.device)

                if self.neg_amax is None:
                    self.neg_amax = neg_max
                    self.pos_amax = pos_max
                else:
                    self.neg_amax = torch.maximum(self.neg_amax, neg_max)
                    self.pos_amax = torch.maximum(self.pos_amax, pos_max)

                self.neg_scale = self.fp_max / self.neg_amax.clamp(min=epsilon)
                self.pos_scale = self.fp_max / self.pos_amax.clamp(min=epsilon)
        else:
            if self.axis is None:
                max_val = x.abs().max().item()
                self.amax = max(self.amax or 0.0, max_val)
                self.scale = self.fp_max / self.amax if self.amax > epsilon else 1.0
                self.zero_point = 0
            else:
                max_val = x.abs().amax(dim=self.axis, keepdim=False)
                self.amax = torch.maximum(self.amax, max_val) if self.amax is not None else max_val
                self.scale = self.fp_max / self.amax.clamp(min=epsilon)
                self.zero_point = 0

    def _simulate_e4m3(self, x: torch.Tensor):
        return load_fake_quant_fp8_ext().fake_e4m3fy(x)

    def _simulate_fp(self, x: torch.Tensor, fp_min: float, fp_max: float, mant_bits: int) -> torch.Tensor:
        x_abs = x.abs()
        sign = x.sign()

        x_clamped = torch.clamp(x_abs, min=0.0, max=fp_max)

        # todo: Subnormal
        x_clamped = torch.where(x_clamped < fp_min, torch.zeros_like(x_clamped), x_clamped)

        exponent = torch.floor(torch.log2(torch.clamp(x_clamped, min=fp_min)))
        mantissa = x_clamped / (2 ** exponent)
        mantissa_rounded = torch.round((mantissa - 1) * (2 ** mant_bits)) / (2 ** mant_bits)
        simulated = (1 + mantissa_rounded) * (2 ** exponent)

        return simulated * sign

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        zero_mask = x.abs() < 1.0 / (1 << 24)

        if self.dual_scale:
            if self.axis is not None:
                shape = [1] * x.dim()
                shape[self.axis] = -1
                pos_scale = self.pos_scale.view(shape)
                neg_scale = self.neg_scale.view(shape)
            else:
                pos_scale = self.pos_scale
                neg_scale = self.neg_scale

            x_scaled = torch.where(
                x >= 0,
                x * pos_scale,
                x * neg_scale
            )

            if self.exp_bits == 4 and self.mant_bits == 3:
                x_quant = self._simulate_e4m3(x_scaled)
            else:
                x_quant = self._simulate_fp(x_scaled, self.fp_min, self.fp_max, self.mant_bits)

            x_dequant = torch.where(
                x >= 0,
                x_quant / pos_scale,
                x_quant / neg_scale
            )
        else:
            if self.axis is not None:
                shape = [1] * x.dim()
                shape[self.axis] = -1
                scale = self.scale.view(shape)
            else:
                scale = self.scale

            x_scaled = x * scale
            if self.exp_bits == 4 and self.mant_bits == 3:
                x_quant = self._simulate_e4m3(x_scaled)
            else:
                x_quant = self._simulate_fp(x_scaled, self.fp_min, self.fp_max, self.mant_bits)
            x_dequant = x_quant / scale

        x_dequant[zero_mask] = 0.0
        return x_dequant.to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quantize(x)

    def __repr__(self):
        if self.dual_scale:
            return (f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, axis={self.axis}, dual_scale=True, "
                    f"neg_amax={self.neg_amax:.4f}, pos_amax={self.pos_amax:.4f})")
        else:
            return (f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, axis={self.axis}, dual_scale=False, "
                    f"amax={self.amax:.4f})")

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