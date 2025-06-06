import types
from typing import Literal
import torch
import torch.nn as nn
from segquant.calibrator.calibrator import CalibratorRegistry
from segquant.config import Calibrate, DType, Optimum
from segquant.layers.splitter import BaseSplitter
from segquant.optimum.optimizer import OptimizerRegistry
from segquant.layers import ext_dict

class SegmentLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight_tensor=None,
        custom_bias_tensor=None,
        input_quant_type=None,
        weight_quant_type=None,
        input_quant_args=None,
        weight_quant_args=None,
        opt_type: Literal["default", "smooth", "svd"] = "default",
        opt_kwargs=None,
        calib_type: Literal["amax", "gptq"] = "amax",
        calib_kwargs=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if custom_weight_tensor is not None:
            assert (
                custom_weight_tensor.shape[0] == out_features
                and custom_weight_tensor.shape[1] == in_features
            ), (
                f"Mismatched custom_weight_tensor shape! "
                f"Expected ({out_features}, {in_features}), "
                f"but got {custom_weight_tensor.shape}."
            )
        else:
            custom_weight_tensor = torch.randn([self.out_features, self.in_features])
        self.bias = bias
        self.bias_data = 0
        if custom_bias_tensor is not None:
            self.bias_data = custom_bias_tensor

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

        real_quant = False
        dual_scale = False
        if input_quant_type is not None:
            if 'real_quant' in input_quant_args:
                real_quant = input_quant_args['real_quant']
            if 'dual_scale' in input_quant_args:
                dual_scale = input_quant_args['dual_scale']

        if weight_quant_type is not None:
            if 'real_quant' in weight_quant_args:
                assert real_quant == weight_quant_args['real_quant'], \
                    f"Mismatch: real_quant={real_quant}, " \
                    f"weight_quant_args['real_quant']={weight_quant_args['real_quant']}"

            if 'dual_scale' in weight_quant_args:
                raise ValueError(
                    "Dual scale is not supported for weight quantizer in DefaultSegmentLinear."
                )

        kernel_type = f'W{weight_quant_type}A{input_quant_type}'

        if real_quant:
            if kernel_type not in ext_dict:
                real_quant = False
            else:
                if ext_dict[kernel_type]['gemm_scaled_fn'] is None or \
                    ext_dict[kernel_type]['gemm_dual_scaled_fn'] is None:
                    real_quant = False

        input_quant_args['real_quant'] = real_quant
        weight_quant_args['real_quant'] = real_quant

        input_len = self.chunks
        weight_len = self.chunks
        if opt_type == 'default':
            if seg_mode == 'input':
                weight_len = 1
            elif seg_mode == 'weight':
                input_len = 1
            else:
                raise ValueError("seg_mode not found")
        elif opt_type == 'smooth' or opt_type == 'svd':
            def trace(self, input_data):
                input_chunks = self._chunk_x(input_data)
                self.optimizer.trace(input_chunks)

            def smooth(self):
                self.optimizer.smooth()

            self.trace = types.MethodType(trace, self)
            self.smooth = types.MethodType(smooth, self)
        else:
            raise ValueError("opt_type not found")

        if self.seg_mode == "input":
            weight_chunks = [custom_weight_tensor]
        elif self.seg_mode == "weight":
            weight_chunks = self.splitter.split_weight(custom_weight_tensor)
        else:
            raise ValueError("seg_mode not found")

        self.optimizer = OptimizerRegistry.create(
            opt_type,
            seg_mode=seg_mode,
            chunks=self.chunks,
            chunksizes=self.chunksizes,
            weight_chunks=weight_chunks,
            input_calibrators=[
                CalibratorRegistry.create(
                    "amax",
                    data_type="input",
                    quant_type=input_quant_type,
                    quant_args=input_quant_args,
                    **calib_kwargs,
                )
                for _ in range(input_len)
            ],
            weight_calibrators=[
                CalibratorRegistry.create(
                    calib_type,
                    data_type="weight",
                    quant_type=weight_quant_type,
                    quant_args=weight_quant_args,
                    **calib_kwargs,
                )
                for _ in range(weight_len)
            ],
            real_quant=real_quant,
            dual_scale=dual_scale,
            kernel_type=kernel_type,
            **opt_kwargs,
        )

    def __repr__(self):
        lines = [
            f"  in_features={self.in_features}, out_features={self.out_features}, seg_mode={self.seg_mode}",
            f"  chunks={self.chunks}, chunksize={self.chunksizes}",
        ]
        lines.append(f"  opt={repr(self.optimizer)}")
        inner_content = ",\n".join(lines)
        base = f"SegmentLinear(\n{inner_content}\n)"
        return base

    def _chunk_x(self, x):
        if self.seg_mode == "input":
            input_chunks = self.splitter.split_input(x)
        elif self.seg_mode == "weight":
            input_chunks = [x]
        else:
            raise ValueError("seg_mode not found")

        return input_chunks

    def calibrate(self, input_data):
        input_chunks = self._chunk_x(input_data)
        self.optimizer.calibrate(input_chunks)

    def finish_calibrate(self):
        self.optimizer.finish_calibrate()

    def forward(self, x):
        input_chunks = self._chunk_x(x)
        quantized_output_chunks = self.optimizer.forward(input_chunks)
        if self.seg_mode == "weight":
            res = self.splitter.concat_output(quantized_output_chunks)
        elif self.seg_mode == "input":
            res = sum(quantized_output_chunks)
        else:
            raise ValueError("seg_mode not found")
        return (res + self.bias_data if self.bias else res)

def create_segment_linear(
    input_dtype: DType,
    weight_dtype: DType,
    opt: Optimum,
    calib: Calibrate,
    in_features,
    out_features,
    opt_kwargs,
    calib_kwargs,
    input_quant_args,
    weight_quant_args,
    **kwargs,
):

    return SegmentLinear(
        in_features,
        out_features,
        input_quant_type=input_dtype.value,
        weight_quant_type=weight_dtype.value,
        opt_type=opt.value,
        calib_type=calib.value,
        opt_kwargs=opt_kwargs,
        calib_kwargs=calib_kwargs,
        input_quant_args=input_quant_args,
        weight_quant_args=weight_quant_args,
        **kwargs
    )
