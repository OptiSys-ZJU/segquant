import copy
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from segquant.calibrator.calibrator import DefaultCalibrator, SmoothQuantCalibrator
from segquant.config import DType, Optimum
from segquant.layers.splitter import BaseSplitter
from segquant.quantizers.quantizer import QuantizerRegistry
from segquant.utils.extension import load_real_quant_fp8_ext

class BaseSegmentLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight_tensor=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_calibrated = False

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if custom_weight_tensor is not None:
            self.linear.weight = nn.Parameter(custom_weight_tensor)

        if seg_mode == "input":
            target_features = in_features
        elif seg_mode == "weight":
            target_features = out_features
        else:
            raise ValueError("seg_mode not found")
        self.seg_mode = seg_mode
        self.chunks = chunks
        if chunksizes is None:
            chunk_size = target_features // self.chunks
            remainder = target_features % self.chunks
            self.chunksizes = [
                chunk_size + (1 if i < remainder else 0) for i in range(self.chunks)
            ]
        else:
            assert len(chunksizes) == self.chunks and sum(chunksizes) == target_features
            self.chunksizes = chunksizes

        self.splitter = BaseSplitter(self.chunksizes, seg_mode)

    def __repr__(self):
        return f"BaseSegmentLinear(in={self.in_features}, out={self.out_features}, seg_mode={self.seg_mode}, chunks={self.chunks}, chunksize={self.chunksizes})"

    def forward(self, _x):
        raise NotImplementedError("Forward method should be implemented in subclasses")

class DefaultSegmentLinear(BaseSegmentLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight_tensor=None,
        input_quant_type=None,
        weight_quant_type=None,
        input_quant_args=None,
        weight_quant_args=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            seg_mode,
            chunks,
            chunksizes,
            custom_weight_tensor,
        )

        if input_quant_type is not None:
            if 'real_quant' in input_quant_args:
                self.real_quant = input_quant_args['real_quant']
            else:
                self.real_quant = False
            if 'dual_scale' in input_quant_args:
                self.dual_scale = input_quant_args['dual_scale']
            else:
                self.dual_scale = False
            input_gen = lambda: QuantizerRegistry.create(
                input_quant_type, **(input_quant_args or {})
            )
        else:
            input_gen = lambda: None
        if weight_quant_type is not None:
            if 'real_quant' in weight_quant_args:
                assert self.real_quant == weight_quant_args['real_quant'], \
                    f"Mismatch: self.real_quant={self.real_quant}, " \
                    f"weight_quant_args['real_quant']={weight_quant_args['real_quant']}"

            if 'dual_scale' in weight_quant_args:
                raise ValueError(
                    "Dual scale is not supported for weight quantizer in DefaultSegmentLinear."
                )

            weight_gen = lambda: QuantizerRegistry.create(
                weight_quant_type, **(weight_quant_args or {})
            )
        else:
            weight_gen = lambda: None

        if seg_mode == "input":
            self.input_quantizers = [input_gen() for _ in range(chunks)]
            self.weight_quantizers = [weight_gen()]
        elif seg_mode == "weight":
            self.input_quantizers = [input_gen()]
            self.weight_quantizers = [weight_gen() for _ in range(chunks)]

        self.calibrator = DefaultCalibrator(
            self.input_quantizers, self.weight_quantizers
        )

        if self.real_quant:
            # todo type select: fp8 int8 int4 ...
            ext = load_real_quant_fp8_ext(required=False)

            if ext is None:
                self.real_quant = False
            else:
                self.gemm_scaled = ext.real_quantized_e4m3fy_gemm_scaled
                self.gemm_dual_scaled = ext.real_quantized_e4m3fy_gemm_dual_scaled

    def __repr__(self):
        base = (
            f"DefaultSegmentLinear(\n"
            f"  in_features={self.in_features},\n"
            f"  out_features={self.out_features},\n"
            f"  seg_mode={self.seg_mode},\n"
            f"  chunks={self.chunks},\n"
            f"  chunksize={self.chunksizes},\n"
        )
        if self.has_calibrated:
            input_q = ",\n    ".join(repr(i) for i in self.input_quantizers)
            weight_q = ",\n    ".join(repr(w) for w in self.weight_quantizers)

            if self.chunks == 1:
                base = f"DefaultSegmentLinear(in_features={self.in_features}, out_features={self.out_features},\n"
                return (
                    base + f"  input_quantizer=({input_q}),\n"
                    f"  weight_quantizer=({weight_q})\n"
                    f")"
                )
            return (
                base + f"  input_quantizers=[\n    {input_q}\n  ],\n"
                f"  weight_quantizers=[\n    {weight_q}\n  ]\n"
                f")"
            )
        return base + "  calib=False\n)"

    def calibrate(self, input_data):
        if self.seg_mode == "input":
            input = self.splitter.split_input(input_data)
            weight = [self.linear.weight]
        elif self.seg_mode == "weight":
            input = [input_data]
            weight = self.splitter.split_weight(self.linear.weight)
        self.calibrator.calibrate(input, weight)

    def finish_calibrate(self):
        quantized_weight = self.calibrator.quantize_weight()
        if self.seg_mode == "input":
            assert len(quantized_weight) == 1
            # input mode, weights should also be split
            quantized_weight = quantized_weight[0].split(self.chunksizes, dim=1)
        bias = self.linear.bias.clone() if self.linear.bias is not None else None
        del self.linear
        self.linear = (quantized_weight, bias)
        self.has_calibrated = True

    def forward(self, x):
        if self.seg_mode == "input":
            input_chunks = self.splitter.split_input(x)
        elif self.seg_mode == "weight":
            input_chunks = [x]
        else:
            raise ValueError("seg_mode not found")

        quantized_weights, bias = self.linear
        if self.seg_mode == "weight":
            if self.real_quant:
                if self.dual_scale:
                    output_chunks = [self.gemm_dual_scaled(
                        input_chunks[0], quantized_weights[i],
                        self.input_quantizers[0].pos_scale, self.input_quantizers[0].neg_scale,
                        self.weight_quantizers[i].scale) for i in range(self.chunks)]
                else:
                    output_chunks = [self.gemm_scaled(
                        input_chunks[0], quantized_weights[i],
                        self.input_quantizers[0].scale, self.weight_quantizers[i].scale)
                        for i in range(self.chunks)]
            else:
                quantized_input = self.calibrator.quantize(input_chunks)
                output_chunks = [F.linear(quantized_input[0],quantized_weights[i])
                                 for i in range(self.chunks)]

            res = self.splitter.concat_output(output_chunks)
            return (res + bias if bias is not None else res)

        if self.seg_mode == "input":
            if self.real_quant:
                if self.dual_scale:
                    quantized_output_chunks = [self.gemm_dual_scaled(
                        input_chunks[i], quantized_weights[i], 
                        self.input_quantizers[i].pos_scale, self.input_quantizers[i].neg_scale,
                        self.weight_quantizers[0].scale)
                        for i in range(self.chunks)]
                else:
                    quantized_output_chunks = [self.gemm_scaled(
                        input_chunks[i], quantized_weights[i],
                        self.input_quantizers[i].scale, self.weight_quantizers[0].scale)
                        for i in range(self.chunks)]
            else:
                quantized_input = self.calibrator.quantize(input_chunks)
                quantized_output_chunks = [F.linear(quantized_input[i],quantized_weights[i])
                                        for i in range(self.chunks)]
            res = sum(quantized_output_chunks)
            return res + bias if bias is not None else res


class SmoothQuantSegmentLinear(BaseSegmentLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight_tensor=None,
        input_quant_type=None,
        weight_quant_type=None,
        input_quant_args=None,
        weight_quant_args=None,
        alpha=1.0,
        dual_s=False,
    ):

        super().__init__(
            in_features,
            out_features,
            bias,
            seg_mode,
            chunks,
            chunksizes,
            custom_weight_tensor,
        )

        if input_quant_type is not None:
            if 'real_quant' in input_quant_args:
                self.real_quant = input_quant_args['real_quant']
            else:
                self.real_quant = False
            if 'dual_scale' in input_quant_args:
                self.dual_scale = input_quant_args['dual_scale']
            else:
                self.dual_scale = False

            input_gen = lambda: QuantizerRegistry.create(
                input_quant_type, **(input_quant_args or {})
            )
        else:
            input_gen = lambda: None
        if weight_quant_type is not None:
            if 'real_quant' in weight_quant_args:
                assert self.real_quant == weight_quant_args['real_quant'], \
                    f"Mismatch: self.real_quant={self.real_quant}, " \
                    f"weight_quant_args['real_quant']={weight_quant_args['real_quant']}"
            if 'dual_scale' in weight_quant_args:
                raise ValueError(
                    "Dual scale is not supported for weight quantizer in SmoothQuantSegmentLinear."
                )

            weight_gen = lambda: QuantizerRegistry.create(
                weight_quant_type, **(weight_quant_args or {})
            )
        else:
            weight_gen = lambda: None

        self.input_quantizers = [input_gen() for _ in range(chunks)]
        self.weight_quantizers = [weight_gen() for _ in range(chunks)]

        assert not dual_s, 'dual_s not work now'
        self.dual_s = dual_s
        self.calibrator = SmoothQuantCalibrator(
            self.input_quantizers, self.weight_quantizers, alpha=alpha, dual_s=dual_s
        )

        if self.real_quant:
            # todo type select: fp8 int8 int4 ...
            ext = None #todo

            if ext is None:
                self.real_quant = False
            else:
                self.gemm_scaled = ext.real_quantized_e4m3fy_gemm_scaled
                self.gemm_dual_scaled = ext.real_quantized_e4m3fy_gemm_dual_scaled

    def __repr__(self):
        base = (
            f"SmoothQuantSegmentLinear(\n"
            f"  in_features={self.in_features},\n"
            f"  out_features={self.out_features},\n"
            f"  seg_mode={self.seg_mode},\n"
            f"  chunks={self.chunks},\n"
            f"  chunksize={self.chunksizes},\n"
            f"  alpha={self.calibrator.alpha},\n"
        )
        if self.has_calibrated:
            input_q = ",\n    ".join(repr(i) for i in self.input_quantizers)
            weight_q = ",\n    ".join(repr(w) for w in self.weight_quantizers)

            if self.chunks == 1:
                base = f"SmoothQuantSegmentLinear(in_features={self.in_features}, out_features={self.out_features}, alpha={self.calibrator.alpha},\n"
                return (
                    base + f"  input_quantizer=({input_q}),\n"
                    f"  weight_quantizer=({weight_q})\n"
                    f")"
                )
            return (
                base + f"  input_quantizers=[\n    {input_q}\n  ],\n"
                f"  weight_quantizers=[\n    {weight_q}\n  ]\n"
                f")"
            )
        return base + f"  calib=False\n)"

    def trace(self, input_data):
        if self.seg_mode == "input":
            input = self.splitter.split_input(input_data)
            weight = self.linear.weight.split(self.chunksizes, dim=1)
        elif self.seg_mode == "weight":
            input = [input_data]
            weight = self.splitter.split_weight(self.linear.weight)

        self.calibrator.trace(input, weight)

    def smooth(self):
        self.calibrator.smooth()

    def calibrate(self, input_data):
        if self.seg_mode == "input":
            input = self.splitter.split_input(input_data)
            weight = self.linear.weight.split(self.chunksizes, dim=1)
        elif self.seg_mode == "weight":
            input = [input_data]
            weight = self.splitter.split_weight(self.linear.weight)

        self.calibrator.calibrate(input, weight)

    def finish_calibrate(self):
        quantized_weight = self.calibrator.quantize_weight()
        bias = self.linear.bias.clone() if self.linear.bias is not None else None
        del self.linear
        self.linear = (quantized_weight, bias)
        self.has_calibrated = True

    def forward(self, x):
        if self.seg_mode == "input":
            input_chunks = self.splitter.split_input(x)
        elif self.seg_mode == "weight":
            input_chunks = [x]
        else:
            raise ValueError("seg_mode not found")

        quantized_weights, bias = self.linear
        if self.seg_mode == "weight":
            if self.real_quant:
                if self.dual_scale:
                    output_chunks = [self.gemm_dual_scaled(
                        input_chunks[0], quantized_weights[i],
                        self.input_quantizers[i].pos_scale, self.input_quantizers[i].neg_scale,
                        self.weight_quantizers[i].scale)
                        for i in range(self.chunks)]
                else:
                    output_chunks = [self.gemm_scaled(
                        input_chunks[0], quantized_weights[i],
                        self.input_quantizers[i].scale,
                        self.weight_quantizers[i].scale)
                        for i in range(self.chunks)]
            else:
                quantized_input = self.calibrator.quantize(input_chunks)
                output_chunks = [F.linear(quantized_input[i],quantized_weights[i])
                                 for i in range(self.chunks)]

            res = self.splitter.concat_output(output_chunks)
            return (res + bias if bias is not None else res)

        if self.seg_mode == "input":
            if self.real_quant:
                if self.dual_scale:
                    quantized_output_chunks = [self.gemm_dual_scaled(
                        input_chunks[i], quantized_weights[i],
                        self.input_quantizers[i].pos_scale, self.input_quantizers[i].neg_scale,
                        self.weight_quantizers[i].scale)
                        for i in range(self.chunks)]
                else:
                    quantized_output_chunks = [self.gemm_scaled(
                        input_chunks[i], quantized_weights[i],
                        self.input_quantizers[i].scale,
                        self.weight_quantizers[i].scale)
                        for i in range(self.chunks)]
            else:
                quantized_input = self.calibrator.quantize(input_chunks)
                quantized_output_chunks = [F.linear(quantized_input[i],quantized_weights[i])
                                        for i in range(self.chunks)]
            res = sum(quantized_output_chunks)
            return res + bias if bias is not None else res

def create_segment_linear(
    input_dtype: DType,
    weight_dtype: DType,
    opt: Optimum,
    in_features,
    out_features,
    **kwargs,
):
    opt_map = {
        Optimum.DEFAULT: DefaultSegmentLinear,
        Optimum.SMOOTH: SmoothQuantSegmentLinear,
    }

    if opt not in opt_map:
        raise ValueError(f"Algorithm '{opt}' not supported")

    if opt == Optimum.DEFAULT:
        kwargs.pop("alpha", None)

    return opt_map[opt](
        in_features,
        out_features,
        input_quant_type=input_dtype.value,
        weight_quant_type=weight_dtype.value,
        **kwargs,
    )
