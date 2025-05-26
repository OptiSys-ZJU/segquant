"""
This module defines quantizers for tensor quantization.

It includes base classes and implementations for integer and floating-point quantizers,
as well as a registry for dynamically registering and retrieving quantizer classes.
"""

from abc import ABC, abstractmethod
import torch
from segquant.utils.extension import load_fake_quant_fp8_ext


class BaseQuantizer(ABC):
    """Base class for quantizers.
    This class defines the interface for quantizers,
    including methods for calibration and quantization.
    Subclasses should implement the `quantize` method.
    """
    @abstractmethod
    def quantize(self, x):
        """Quantize the input tensor `x`."""

    def calibrate(self, x):
        """Calibrate the quantizer using the input tensor `x`.
        This method can be used to compute scale and zero-point values based on the input data.
        """


class QuantizerRegistry:
    """Registry for quantizers.
    This class allows for dynamic registration and retrieval of quantizer classes.
    Quantizers can be registered with a name and later retrieved or instantiated.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a quantizer class with a name.
        Args:
            name (str): The name to register the quantizer class under.
        Returns:
            function: A decorator that registers the quantizer class.
        """
        def wrapper(quantizer_cls):
            cls._registry[name] = quantizer_cls
            return quantizer_cls

        return wrapper

    @classmethod
    def get(cls, name):
        """Retrieve a quantizer class by name.
        Args:
            name (str): The name of the quantizer class to retrieve.
        Returns:
            type: The quantizer class registered under the given name, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def create(cls, name, **kwargs):
        """Create an instance of a quantizer class by name.
        Args:
            name (str): The name of the quantizer class to create.
            **kwargs: Additional keyword arguments to pass to the quantizer class constructor.
        Returns:
            BaseQuantizer: An instance of the quantizer class registered under the given name.
        Raises:
            ValueError: If the quantizer class with the given name is not found in the registry.
        """
        quantizer_cls = cls.get(name)
        if quantizer_cls is None:
            raise ValueError(f"Quantizer '{name}' not found in registry.")
        return quantizer_cls(**kwargs)


@QuantizerRegistry.register("int8")
class IntQuantizer(BaseQuantizer):
    """Integer quantizer class for quantizing tensors to int8 or other integer formats.
    This class supports symmetric and asymmetric quantization, as well as dual-scale quantization.
    It can be used to quantize tensors to a specified number of bits,
    with options for axis and dual-scale.
    Args:
        num_bits (int): Number of bits for quantization (default: 8).
        symmetric (bool): Whether to use symmetric quantization (default: True).
        axis (int or None): Axis along which to compute the scale and zero-point (default: None).
        dual_scale (bool): Whether to use dual-scale quantization (default: False).
    """
    def __init__(self, num_bits=8, symmetric=True, axis=None, dual_scale=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.axis = axis
        self.dual_scale = dual_scale

        self.qmin = -(2 ** (num_bits - 1))
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

        self.scale = 1.0

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

                    self.neg_scale = (
                        self.qmin / (-self.neg_amax) if self.neg_amax > epsilon else 1.0
                    )
                    self.pos_scale = (
                        self.qmax / self.pos_amax if self.pos_amax > epsilon else 1.0
                    )
                    self.zero_point = 0
                else:
                    neg_max = (
                        neg_x.amax(dim=self.axis, keepdim=False)
                        if neg_x.numel() > 0
                        else torch.zeros(x.size(self.axis), device=x.device)
                    )
                    pos_max = (
                        pos_x.amax(dim=self.axis, keepdim=False)
                        if pos_x.numel() > 0
                        else torch.zeros(x.size(self.axis), device=x.device)
                    )

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
                    self.amax = (
                        torch.maximum(self.amax, max_val)
                        if self.amax is not None
                        else max_val
                    )
                    self.scale = self.qmax / self.amax.clamp(min=epsilon)
                    self.zero_point = 0
        else:
            if self.axis is None:
                min_val = x.min().item()
                max_val = x.max().item()
                self.amin = min(
                    self.amin if self.amin is not None else min_val, min_val
                )
                self.amax = max(
                    self.amax if self.amax is not None else max_val, max_val
                )
                self.scale = (self.qmax - self.qmin) / (self.amax - self.amin)
                self.zero_point = int(round(self.qmin - self.amin * self.scale))
            else:
                min_val = x.amin(dim=self.axis, keepdim=False)
                max_val = x.amax(dim=self.axis, keepdim=False)
                self.amin = (
                    torch.minimum(self.amin, min_val)
                    if self.amin is not None
                    else min_val
                )
                self.amax = (
                    torch.maximum(self.amax, max_val)
                    if self.amax is not None
                    else max_val
                )
                self.scale = (self.qmax - self.qmin) / (self.amax - self.amin)
                self.zero_point = (
                    (self.qmin - self.amin * self.scale).round().to(torch.int)
                )

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
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
                torch.clamp(torch.round(x * neg_scale), self.qmin, 0),
            )
            x_dequant = torch.where(x >= 0, x_quant / pos_scale, x_quant / neg_scale)
            return x_dequant.to(x.dtype)

        if self.axis is not None:
            shape = [1] * x.dim()
            shape[self.axis] = -1
            scale = self.scale.view(shape)
            zero_point = (
                self.zero_point.view(shape)
                if isinstance(self.zero_point, torch.Tensor)
                else self.zero_point
            )
            scale = scale.T
        else:
            scale = self.scale
            zero_point = self.zero_point

        x_int = torch.round(x * scale) + zero_point
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) / scale
        return x_dequant.to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._fake_quantize(x)

    def __repr__(self):
        if self.symmetric:
            if self.dual_scale:
                return (
                    f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, "
                    f"dual_scale=True, axis={self.axis}, "
                    f"neg_amax={self.neg_amax:.4f}, pos_amax={self.pos_amax:.4f})"
                )
            if isinstance(self.amax, torch.Tensor):
                amin = self.amax.min().item()
                amax = self.amax.max().item()
                amax_str = f"[{amin:.4f}, {amax:.4f}]"
                return (
                    f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, "
                    f"dual_scale=False, axis={self.axis}, "
                    f"amax=[{amax_str}])"
                )
            return (
                f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, "
                f"dual_scale=False, axis={self.axis}, "
                f"amax={self.amax:.4f})"
            )
        return (
            f"IntQuantizer(num_bits={self.num_bits}, symmetric=False, axis={self.axis}, "
            f"amin={self.amin:.4f}, amax={self.amax:.4f}, zero_point={self.zero_point:.4f})"
        )


@QuantizerRegistry.register("fpe4m3")
class FloatQuantizer(BaseQuantizer):
    """
    Float quantizer class for quantizing tensors to NVIDIA's fpe4m3 format
    or other floating-point formats.
    This class supports dual-scale quantization and can be configured
    with different exponent and mantissa bits.
    Args:
        exp_bits (int): Number of exponent bits (default: 4).
        mant_bits (int): Number of mantissa bits (default: 3).
        axis (int or None): Axis along which to compute the scale and zero-point (default: None).
        dual_scale (bool): Whether to use dual-scale quantization (default: False).
    """
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

                self.neg_scale = (
                    self.fp_max / self.neg_amax if self.neg_amax > epsilon else 1.0
                )
                self.pos_scale = (
                    self.fp_max / self.pos_amax if self.pos_amax > epsilon else 1.0
                )
            else:
                neg_max = (
                    neg_x.amax(dim=self.axis, keepdim=False)
                    if neg_x.numel() > 0
                    else torch.zeros(x.size(self.axis), device=x.device)
                )
                pos_max = (
                    pos_x.amax(dim=self.axis, keepdim=False)
                    if pos_x.numel() > 0
                    else torch.zeros(x.size(self.axis), device=x.device)
                )

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
                self.amax = (
                    torch.maximum(self.amax, max_val)
                    if self.amax is not None
                    else max_val
                )
                self.scale = self.fp_max / self.amax.clamp(min=epsilon)
                self.zero_point = 0

    def _simulate_e4m3(
        self, x: torch.Tensor, fp_min: float, fp_max: float, mant_bits: int
    ):
        ext = load_fake_quant_fp8_ext(required=False)
        if ext is not None:
            return ext.fake_e4m3fy(x)
        return self._simulate_fp(x, fp_min, fp_max, mant_bits)

    @staticmethod
    def _simulate_fp(
        x: torch.Tensor, fp_min: float, fp_max: float, mant_bits: int
    ) -> torch.Tensor:
        x_abs = x.abs()
        sign = x.sign()

        x_clamped = torch.clamp(x_abs, min=0.0, max=fp_max)

        x_clamped = torch.where(
            x_clamped < fp_min, torch.zeros_like(x_clamped), x_clamped
        )

        exponent = torch.floor(torch.log2(torch.clamp(x_clamped, min=fp_min)))
        mantissa = x_clamped / (2 ** exponent)
        mantissa_rounded = torch.round((mantissa - 1) * (2 ** mant_bits)) / (
            2 ** mant_bits
        )
        simulated = (1 + mantissa_rounded) * (2 ** exponent)

        return simulated * sign

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
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

            x_scaled = torch.where(x >= 0, x * pos_scale, x * neg_scale)

            if self.exp_bits == 4 and self.mant_bits == 3:
                x_quant = self._simulate_e4m3(
                    x_scaled, self.fp_min, self.fp_max, self.mant_bits
                )
            else:
                x_quant = self._simulate_fp(
                    x_scaled, self.fp_min, self.fp_max, self.mant_bits
                )

            x_dequant = torch.where(x >= 0, x_quant / pos_scale, x_quant / neg_scale)
        else:
            if self.axis is not None:
                shape = [1] * x.dim()
                shape[self.axis] = -1
                scale = self.scale.view(shape)
            else:
                scale = self.scale

            x_scaled = x * scale
            if self.exp_bits == 4 and self.mant_bits == 3:
                x_quant = self._simulate_e4m3(
                    x_scaled, self.fp_min, self.fp_max, self.mant_bits
                )
            else:
                x_quant = self._simulate_fp(
                    x_scaled, self.fp_min, self.fp_max, self.mant_bits
                )
            x_dequant = x_quant / scale

        x_dequant[zero_mask] = 0.0
        return x_dequant.to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._fake_quantize(x)

    def __repr__(self):
        if self.dual_scale:
            return (
                f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, "
                f"axis={self.axis}, dual_scale=True, "
                f"neg_amax={self.neg_amax:.4f}, pos_amax={self.pos_amax:.4f})"
            )
        return (
            f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, "
            f"axis={self.axis}, dual_scale=False, "
            f"amax={self.amax:.4f})"
        )


@QuantizerRegistry.register("int16")
def int16_factory():
<<<<<<< HEAD
=======
    """Factory function for creating an IntQuantizer with 16 bits."""
>>>>>>> d81aca6dc1d28d0a78387b9e2c3d3eac34e174c2
    return IntQuantizer(num_bits=16)


@QuantizerRegistry.register("int6")
def int6_factory():
    """Factory function for creating an IntQuantizer with 6 bits."""
    return IntQuantizer(num_bits=6)


@QuantizerRegistry.register("int4")
def int4_factory():
    """Factory function for creating an IntQuantizer with 4 bits."""
    return IntQuantizer(num_bits=4)


@QuantizerRegistry.register("fpe5m2")
def fp8e5m2_factory():
    """Factory function for creating a FloatQuantizer with 5 exponent bits and 2 mantissa bits."""
    return FloatQuantizer(exp_bits=5, mant_bits=2)
