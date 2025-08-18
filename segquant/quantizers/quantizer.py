"""
This module defines quantizers for tensor quantization.

It includes base classes and implementations for integer and floating-point quantizers,
as well as a registry for dynamically registering and retrieving quantizer classes.
"""

from abc import ABC, abstractmethod
import torch
from segquant.utils.extension import (
    load_fake_quant_fp8_ext,
    load_real_quant_fp8_ext,
    load_real_quant_int4_ext,
    load_real_quant_int8_ext,
)


class BaseQuantizer(ABC):
    """Base class for quantizers.
    This class defines the interface for quantizers,
    including methods for calibration and quantization.
    Subclasses should implement the `quantize` method.
    """

    def __init__(
        self,
        axis=None,
        symmetric=True,
        dual_scale=False,
        real_quant=False,
        dynamic=False,
        dummy=False,
    ):
        self.axis = axis
        self.symmetric = symmetric
        self.dual_scale = dual_scale
        self.real_quant = real_quant
        self.dynamic = dynamic
        self.dummy = dummy

        # parameters for quantization, can be tensor or scalar
        self.amax = None
        self.amin = None
        self.neg_amax = None
        self.pos_amax = None

        self.zero_point = None
        self.scale = 1.0
        self.neg_scale = None
        self.pos_scale = None

    def reset(self):
        self.amax = None
        self.amin = None
        self.neg_amax = None
        self.pos_amax = None

    def _move_to_device(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    def _process_dim(self, x):
        ndim = x.dim()
        axis = self.axis

        if axis is None:
            return tuple(range(ndim))

        if isinstance(axis, int):
            axis = (axis,)
        elif not isinstance(axis, (tuple, list)):
            raise TypeError(f"axis must be int or tuple of int, got {type(axis)}")

        axis = tuple(a + ndim if a < 0 else a for a in axis)
        for a in axis:
            if not (0 <= a < ndim):
                raise ValueError(f"axis {a} out of range for tensor with dim {ndim}")
        return tuple(i for i in range(ndim) if i not in axis)

    @abstractmethod
    def quantize(self, x):
        """Quantize the input tensor `x`."""

    def to(self, device):
        self.zero_point = self._move_to_device(self.zero_point, device)
        self.scale = self._move_to_device(self.scale, device)
        self.neg_scale = self._move_to_device(self.neg_scale, device)
        self.pos_scale = self._move_to_device(self.pos_scale, device)
        self.amax = self._move_to_device(self.amax, device)
        self.amin = self._move_to_device(self.amin, device)
        self.neg_amax = self._move_to_device(self.neg_amax, device)
        self.pos_amax = self._move_to_device(self.pos_amax, device)
        return self

    def calibrate(self, x):
        """Calibrate the quantizer using the input tensor `x`.
        This method can be used to compute scale and zero-point values based on the input data.
        """

    def repr_amax(self, t):
        if isinstance(t, torch.Tensor):
            amin = t.min().item()
            amax = t.max().item()
            amax_str = f"[{amin:.4f}, {amax:.4f}]"
        else:
            amax_str = f"{t:.4f}"

        return amax_str


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

    def __init__(
        self,
        num_bits=8,
        symmetric=True,
        axis=None,
        dual_scale=False,
        real_quant=False,
        dynamic=False,
        dummy=False,
    ):
        super().__init__(
            axis=axis,
            symmetric=symmetric,
            dual_scale=dual_scale,
            real_quant=real_quant,
            dynamic=dynamic,
            dummy=dummy,
        )

        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1

    def calibrate(self, x: torch.Tensor):
        epsilon = 1.0 / (1 << 24)

        dim = self._process_dim(x)

        if self.symmetric:
            if self.dual_scale:
                if self.axis is None:
                    neg_x = x[x < 0].abs()
                    pos_x = x[x > 0].abs()
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
                    neg_x = torch.where(x < 0, x.abs(), torch.zeros_like(x))
                    pos_x = torch.where(x > 0, x.abs(), torch.zeros_like(x))

                    neg_max = neg_x.amax(dim=dim, keepdim=True)
                    pos_max = pos_x.amax(dim=dim, keepdim=True)

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
                    max_val = x.abs().amax(dim=dim, keepdim=True)
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
                min_val = x.amin(dim=dim, keepdim=True)
                max_val = x.amax(dim=dim, keepdim=True)
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
        ### axis for weight [out, in]
        # axis=0, amax(dim=1), scale shape(out,1)
        ### axis for weight [out, in, kh, kw]
        # axis=0, amax(dim=(1,2,3)), scale shape(out,1,1,1)
        # axis=(0,1), amax(dim=(2,3)), scale shape(out,in,1,1)
        ### axis for input [batch, in]
        # axis=0, amax(dim=1), scale shape(batch,1)
        ### axis for input [batch, in, H, W]
        # axis=0, amax(dim=(1,2,3)), scale shape(batch,1,1,1)
        # axis=(0,1), amax(dim(2,3)), scale shape(batch,in,1,1)

        if self.symmetric and self.dual_scale:
            pos_scale = self.pos_scale
            neg_scale = self.neg_scale

            x_quant = torch.where(
                x >= 0,
                torch.clamp(torch.round(x * pos_scale), 0, self.qmax),
                torch.clamp(torch.round(x * neg_scale), self.qmin, 0),
            )
            x_dequant = torch.where(x >= 0, x_quant / pos_scale, x_quant / neg_scale)
            return x_dequant.to(x.dtype)

        scale = self.scale
        zero_point = self.zero_point

        x_int = torch.round(x * scale) + zero_point
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) / scale
        assert (
            x.shape == x_dequant.shape
        ), f"Shape mismatch: x {x.shape}, x_dequant {x_dequant.shape}"
        return x_dequant.to(x.dtype)

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._fake_quantize(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.dummy:
            return x
        if self.dynamic:
            self.reset()
            self.calibrate(x)

        if self.real_quant:
            # when real quantization is enabled, only weights are quantized
            assert (
                not self.dual_scale
            ), "Weight quantization does not support dual scale."
            ext = None
            if self.num_bits == 8:
                ext = load_real_quant_int8_ext(required=False)[0]
            elif self.num_bits == 4:
                ext = load_real_quant_int4_ext(required=False)[0]

            if ext is not None:
                res = ext.create_quantized_weights(x)

                if self.axis is None:
                    ext.real_quantized_quantize_weights(x.contiguous(), res, self.scale)
                else:
                    ### axis for weight [out, in]
                    # axis=0, amax(dim=1), scale shape(out,1)
                    ### axis for weight [out, in, kh, kw]
                    # axis=0, amax(dim=(1,2,3)), scale shape(out,1,1,1)
                    # axis=(0,1), amax(dim=(2,3)), scale shape(out,in,1,1)

                    assert isinstance(
                        self.scale, torch.Tensor
                    ), f"Expected self.scale to be a torch.Tensor, but got {type(self.scale)}"

                    # call kernel
                    ext.real_quantized_quantize_weights(
                        x.contiguous(),
                        res,
                        self.scale.squeeze()
                        .to(dtype=torch.float32, device=x.device)
                        .contiguous(),
                    )

                return res

        # fake quantization
        return self._fake_quantize(x)

    def __repr__(self):
        if self.symmetric:
            if self.dual_scale:
                return (
                    f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, "
                    f"real_quant={self.real_quant}, enable={not self.fake}, dynamic={self.dynamic}, "
                    f"dual_scale=True, axis={self.axis}, "
                    f"neg_amax={self.repr_amax(self.neg_amax)}, pos_amax={self.repr_amax(self.pos_amax)})"
                )
            return (
                f"IntQuantizer(num_bits={self.num_bits}, symmetric=True, "
                f"real_quant={self.real_quant}, enable={not self.fake}, dynamic={self.dynamic}, "
                f"dual_scale=False, axis={self.axis}, "
                f"amax={self.repr_amax(self.amax)})"
            )
        return (
            f"IntQuantizer(num_bits={self.num_bits}, symmetric=False, axis={self.axis}, "
            f"real_quant={self.real_quant}, enable={not self.fake}, dynamic={self.dynamic}, "
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

    def __init__(
        self,
        exp_bits=4,
        mant_bits=3,
        axis=None,
        dual_scale=False,
        real_quant=False,
        dynamic=False,
        dummy=False,
    ):
        super().__init__(
            axis=axis,
            symmetric=False,
            dual_scale=dual_scale,
            real_quant=real_quant,
            dynamic=dynamic,
            dummy=dummy,
        )

        self.exp_bits = exp_bits
        self.mant_bits = mant_bits

        if exp_bits == 4 and mant_bits == 3:
            # nvidia's fpe4m3
            self.fp_max = 448.0
            self.fp_min = 2 ** (-6)
            self.subnorm_max = 0.875 * (2 ** (-6))
            self.subnorm_min = 2 ** (-9)
        elif exp_bits == 5 and mant_bits == 2:
            # nvidia's fpe5m2
            self.fp_max = 57344.0
            self.fp_min = 2 ** (-14)
            self.subnorm_max = 0.75 * (2 ** (-14))
            self.subnorm_min = 2 ** (-16)
        else:
            self.fp_max = (2 ** (2**exp_bits - 2)) * (2 - 2 ** (-mant_bits))
            self.fp_min = 2 ** (1 - (2 ** (exp_bits - 1)))

        self.scale = None
        self.zero_point = 0

    def calibrate(self, x: torch.Tensor):
        epsilon = 1.0 / (1 << 24)

        dim = self._process_dim(x)

        if self.dual_scale:
            if self.axis is None:
                neg_x = x[x < 0].abs()
                pos_x = x[x > 0].abs()
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
                neg_x = torch.where(x < 0, x.abs(), torch.zeros_like(x))
                pos_x = torch.where(x > 0, x.abs(), torch.zeros_like(x))

                neg_max = neg_x.amax(dim=dim, keepdim=True)
                pos_max = pos_x.amax(dim=dim, keepdim=True)

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
                max_val = x.abs().amax(dim=dim, keepdim=True)
                self.amax = (
                    torch.maximum(self.amax, max_val)
                    if self.amax is not None
                    else max_val
                )
                self.scale = self.fp_max / self.amax.clamp(min=epsilon)
                self.zero_point = 0

    def _simulate_e4m3(self, x: torch.Tensor):
        ext = load_fake_quant_fp8_ext(required=False)
        if ext is not None:
            return ext.fake_e4m3fy(x)
        return None

    def _simulate_fp(self, x: torch.Tensor) -> torch.Tensor:
        if self.exp_bits == 4 and self.mant_bits == 3:
            res = self._simulate_e4m3(x)
            if res is not None:
                return res

        x_abs = x.abs()
        sign = x.sign()

        x_clamped = torch.clamp(x_abs, min=0.0, max=self.fp_max)

        x_clamped = torch.where(
            x_clamped < self.fp_min, torch.zeros_like(x_clamped), x_clamped
        )

        exponent = torch.floor(torch.log2(torch.clamp(x_clamped, min=self.fp_min)))
        mantissa = x_clamped / (2**exponent)
        mantissa_rounded = torch.round((mantissa - 1) * (2**self.mant_bits)) / (
            2**self.mant_bits
        )
        simulated = (1 + mantissa_rounded) * (2**exponent)

        return simulated * sign

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        ### axis for weight [out, in]
        # axis=0, amax(dim=1), scale shape(out,1)
        ### axis for weight [out, in, kh, kw]
        # axis=0, amax(dim=(1,2,3)), scale shape(out,1,1,1)
        # axis=(0,1), amax(dim=(2,3)), scale shape(out,in,1,1)
        ### axis for input [batch, in]
        # axis=0, amax(dim=1), scale shape(batch,1)
        ### axis for input [batch, in, H, W]
        # axis=0, amax(dim=(1,2,3)), scale shape(batch,1,1,1)
        # axis=(0,1), amax(dim(2,3)), scale shape(batch,in,1,1)

        zero_mask = x.abs() < 1.0 / (1 << 24)

        if self.dual_scale:
            pos_scale = self.pos_scale
            neg_scale = self.neg_scale
            x_scaled = torch.where(x >= 0, x * pos_scale, x * neg_scale)
            x_quant = self._simulate_fp(x_scaled)
            x_dequant = torch.where(x >= 0, x_quant / pos_scale, x_quant / neg_scale)
        else:
            scale = self.scale
            x_scaled = x * scale
            x_quant = self._simulate_fp(x_scaled)
            x_dequant = x_quant / scale

        x_dequant[zero_mask] = 0.0
        assert (
            x.shape == x_dequant.shape
        ), f"Shape mismatch: x {x.shape}, x_dequant {x_dequant.shape}"
        return x_dequant.to(x.dtype)

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._fake_quantize(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.dummy:
            return x
        if self.dynamic:
            self.reset()
            self.calibrate(x)

        if self.real_quant:
            # when real quantization is enabled, only weights are quantized
            assert (
                not self.dual_scale
            ), "Weight quantization does not support dual scale."
            ext = load_real_quant_fp8_ext(required=False)[0]
            if ext is not None:
                res = ext.create_quantized_weights(x)

                if self.axis is None:
                    ext.real_quantized_quantize_weights(x.contiguous(), res, self.scale)
                else:
                    ### axis for weight [out, in]
                    # axis=0, amax(dim=1), scale shape(out,1)
                    ### axis for weight [out, in, kh, kw]
                    # axis=0, amax(dim=(1,2,3)), scale shape(out,1,1,1)
                    # axis=(0,1), amax(dim=(2,3)), scale shape(out,in,1,1)
                    assert isinstance(
                        self.scale, torch.Tensor
                    ), f"Expected self.scale to be a torch.Tensor, but got {type(self.scale)}"

                    # call kernel
                    ext.real_quantized_quantize_weights(
                        x.contiguous(),
                        res,
                        self.scale.squeeze()
                        .to(dtype=torch.float32, device=x.device)
                        .contiguous(),
                    )

                return res

        # fake quantization
        return self._fake_quantize(x)

    def __repr__(self):
        if self.dual_scale:
            return (
                f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, "
                f"real_quant={self.real_quant}, enable={not self.fake}, dynamic={self.dynamic}, "
                f"axis={self.axis}, dual_scale=True, "
                f"neg_amax={self.repr_amax(self.neg_amax)}, pos_amax={self.repr_amax(self.pos_amax)})"
            )
        return (
            f"FloatQuantizer(exp_bits={self.exp_bits}, mant_bits={self.mant_bits}, "
            f"real_quant={self.real_quant}, enable={not self.fake}, dynamic={self.dynamic}, "
            f"axis={self.axis}, dual_scale=False, "
            f"amax={self.repr_amax(self.amax)})"
        )


@QuantizerRegistry.register("int16")
def int16_factory(**kwargs):
    return IntQuantizer(num_bits=16, **kwargs)


@QuantizerRegistry.register("int6")
def int6_factory(**kwargs):
    return IntQuantizer(num_bits=6, **kwargs)


@QuantizerRegistry.register("int4")
def int4_factory(**kwargs):
    return IntQuantizer(num_bits=4, **kwargs)


@QuantizerRegistry.register("fpe5m2")
def fp8e5m2_factory(**kwargs):
    return FloatQuantizer(exp_bits=5, mant_bits=2, **kwargs)


@QuantizerRegistry.register("fp16")
def fp16_factory(**kwargs):
    return FloatQuantizer(exp_bits=5, mant_bits=10, dummy=True, **kwargs)


@QuantizerRegistry.register("bf16")
def bf16_factory(**kwargs):
    return FloatQuantizer(exp_bits=8, mant_bits=7, dummy=True, **kwargs)
