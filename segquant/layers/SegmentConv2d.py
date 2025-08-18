from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from segquant.calibrator.calibrator import CalibratorRegistry
from segquant.layers.splitter import BCHWSplitter
from segquant.layers import ext_dict
from segquant.optimum.optimizer import OptimizerRegistry

class SegmentConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple | str = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight_tensor: Optional[torch.Tensor] = None,
        custom_bias_tensor: Optional[torch.Tensor] = None,
        input_quant_type=None,
        weight_quant_type=None,
        input_quant_args=None,
        weight_quant_args=None,
        opt_type: Literal["default", "smooth"] = "default",
        opt_kwargs=None,
        calib_type: Literal["amax", "gptq"] = "amax",
        calib_kwargs=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        if self.padding_mode != 'zeros':
            # F.conv2d only supports 'zeros' padding mode
            # todo: add F.pad support for other modes
            raise ValueError("Only 'zeros' padding mode is supported for SegmentConv2d.")

        assert self.in_channels % self.groups == 0, "in_channels must be divisible by groups"
        assert self.out_channels % self.groups == 0, "out_channels must be divisible by groups"

        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        if custom_weight_tensor is not None:
            assert custom_weight_tensor.shape == weight_shape, (
                f"Mismatched custom_weight_tensor shape! "
                f"Expected {weight_shape}, but got {custom_weight_tensor.shape}."
            )

            custom_weight_tensor = custom_weight_tensor.to(device, dtype=dtype).clone()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
        else:
            custom_weight_tensor = torch.randn(weight_shape, device=device, dtype=dtype)
        
        self.bias_data = None
        if self.bias:
            bias_shape = (self.out_channels,)
            if custom_bias_tensor is not None:
                assert custom_bias_tensor.shape == bias_shape, (
                    f"Mismatched custom_bias_tensor shape! "
                    f"Expected {bias_shape}, but got {custom_bias_tensor.shape}."
                )
                self.bias_data = custom_bias_tensor.to(device, dtype=dtype).clone()
            else:
                self.bias_data = torch.zeros(bias_shape, device=device, dtype=dtype)
        
        self.conv_kwargs = {
            "kernel_size": self.kernel_size,
            "bias": self.bias_data,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        
        if seg_mode == "input":
            target_features = in_channels
        elif seg_mode == "weight":
            target_features = out_channels
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
        
        self.splitter = BCHWSplitter(self.chunksizes, seg_mode)

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
                print(f"[Warning] Seglinear need [{kernel_type}] but not found")
                real_quant = False
            else:
                if ext_dict[kernel_type]['conv2d_scaled_fn'] is None or \
                    ext_dict[kernel_type]['conv2d_dual_scaled_fn'] is None:
                    print(f"[Warning] Seglinear need [{kernel_type}] but function is None")
                    real_quant = False

        input_quant_args['real_quant'] = real_quant
        weight_quant_args['real_quant'] = real_quant

        input_len = self.chunks
        weight_len = self.chunks
        self.opt_type = opt_type
        if opt_type == 'default':
            if seg_mode == 'input':
                weight_len = 1
            elif seg_mode == 'weight':
                input_len = 1
            else:
                raise ValueError("seg_mode not found")
        elif opt_type in ('smooth', 'svd'):
            pass
        else:
            raise ValueError("opt_type not found")

        weight_chunks = self._chunk_w(custom_weight_tensor)

        self.optimizer = OptimizerRegistry.create(
            opt_type,
            layer_mode='conv2d',
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
            layer_kwargs=self.conv_kwargs,
            **opt_kwargs,
        )
    
    def _chunk_x(self, x):
        if self.seg_mode == "input":
            input_chunks = self.splitter.split_input(x)
        elif self.seg_mode == "weight":
            input_chunks = [x]
        else:
            raise ValueError("seg_mode not found")

        return input_chunks
    
    def _chunk_w(self, w):
        if self.seg_mode == "input":
            weight_chunks = [w]
        elif self.seg_mode == "weight":
            weight_chunks = self.splitter.split_weight(w)
        else:
            raise ValueError("seg_mode not found")
        return weight_chunks
    
    def trace(self, input_data):
        input_chunks = self._chunk_x(input_data)
        self.optimizer.trace(input_chunks)

    def smooth(self):
        self.optimizer.smooth()

    def calibrate(self, input_data):
        input_chunks = self._chunk_x(input_data)
        self.optimizer.calibrate(input_chunks)

    def finish_calibrate(self):
        self.optimizer.finish_calibrate()
    
    def segment_forward(self, x, weight):
        # only for search
        input_chunks = self._chunk_x(x)
        weight_chunks = self._chunk_w(weight)

        output_chunks = []
        if self.seg_mode == 'weight':
            for w in weight_chunks:
                output_chunks.append(F.conv2d(input_chunks[0], w, **self.conv_kwargs))
        elif self.seg_mode == 'input':
            this_weight_chunks = weight_chunks[0].split(self.chunksizes, dim=1)
            for i, inp in enumerate(input_chunks):
                output_chunks.append(F.conv2d(inp, this_weight_chunks[i], **self.conv_kwargs))

        return output_chunks

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.optimizer.to(device)
        if self.bias_data is not None:
            self.bias_data = self.bias_data.to(device)
        return self

    def forward(self, x, chunked=False):
        input_chunks = self._chunk_x(x)
        quantized_output_chunks = self.optimizer.forward(input_chunks)
        if chunked:
            return quantized_output_chunks
        if self.seg_mode == "weight":
            res = self.splitter.concat_output(quantized_output_chunks)
        elif self.seg_mode == "input":
            res = sum(quantized_output_chunks)
        else:
            raise ValueError("seg_mode not found")
        return (res + self.bias_data if self.bias else res)