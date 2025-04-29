import copy
from typing import Literal
import torch
import torch.nn as nn
from segquant.calibrator.calibrator import DefaultCalibrator, SmoothQuantCalibrator
from segquant.config import DType
from segquant.layers.splitter import BaseSplitter
from segquant.quantizers.quantizer import FakeQuantizer, QuantizerRegistry

class BaseSegmentLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None, 
                 custom_weight_tensor=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_calibrated = False

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if custom_weight_tensor is not None:
            self.linear.weight = nn.Parameter(custom_weight_tensor)

        if seg_mode == 'input':
            target_features = in_features
        elif seg_mode == 'weight':
            target_features = out_features
        else:
            raise ValueError('seg_mode not found')
        self.seg_mode = seg_mode
        self.chunks = chunks
        if chunksizes is None:
            chunk_size = target_features // self.chunks
            remainder = target_features % self.chunks
            self.chunksizes = [chunk_size + (1 if i < remainder else 0) for i in range(self.chunks)]
        else:
            assert len(chunksizes) == self.chunks and sum(chunksizes) == target_features
            self.chunksizes = chunksizes

        self.splitter = BaseSplitter(self.chunksizes, seg_mode)
    
    def __repr__(self):
        return f"BaseSegmentLinear(in={self.in_features}, out={self.out_features}, seg_mode={self.seg_mode}, chunks={self.chunks}, chunksize={self.chunksizes})"

    @staticmethod
    def split_linear(linear, chunksizes, dim=1):
        weight = linear.weight
        bias = linear.bias
        in_features = linear.in_features
        out_features = linear.out_features
        device = weight.device
        dtype = weight.dtype

        layers = []
        start = 0
        for size in chunksizes:
            end = start + size
            if dim == 1:
                sub_weight = weight[:, start:end].clone()
                new_linear = nn.Linear(size, out_features, bias=False)
                new_linear.weight.data.copy_(sub_weight)
            else:
                sub_weight = weight[start:end, :].clone()
                new_linear = nn.Linear(in_features, size, bias=False)
                new_linear.weight.data.copy_(sub_weight)

            new_linear = new_linear.to(device).to(dtype)
            layers.append(new_linear)
            start = end

        return nn.ModuleList(layers), bias

class DefaultSegmentLinear(BaseSegmentLinear):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None,
                 custom_weight_tensor=None,
                 quant_type=None, input_quant_args=None, weight_quant_args=None):
        super().__init__(in_features, out_features, bias, seg_mode, chunks, chunksizes, custom_weight_tensor)

        if quant_type is not None:
            input_gen = lambda: QuantizerRegistry.create(quant_type, **(input_quant_args or {}))
            weight_gen = lambda: QuantizerRegistry.create(quant_type, **(weight_quant_args or {}))
        else:
            input_gen = weight_gen = lambda: None
        
        if seg_mode == 'input':
            self.input_quantizers = [input_gen() for _ in range(chunks)]
            self.weight_quantizers = [weight_gen()]
        elif seg_mode == 'weight':
            self.input_quantizers = [input_gen()]
            self.weight_quantizers = [weight_gen() for _ in range(chunks)]

        self.calibrator = DefaultCalibrator(self.input_quantizers, self.weight_quantizers)

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
                base = (
                    f"DefaultSegmentLinear(in_features={self.in_features}, out_features={self.out_features},\n"
                )
                return (
                    base +
                    f"  input_quantizer=({input_q}),\n"
                    f"  weight_quantizer=({weight_q})\n"
                    f")"
                )
            else:
                return (
                    base +
                    f"  input_quantizers=[\n    {input_q}\n  ],\n"
                    f"  weight_quantizers=[\n    {weight_q}\n  ]\n"
                    f")"
                )
        else:
            return base + f"  calib=False\n)"

    def calibrate(self, input_datas):
        if self.seg_mode == 'input':
            inputs = [self.splitter.split_input(input_data) for input_data in input_datas]
            weight = [self.linear.weight]
        elif self.seg_mode == 'weight':
            inputs = [[input_data] for input_data in input_datas]
            weight = self.splitter.split_weight(self.linear.weight)
        
        self.calibrator.calibrate(inputs, weight)

        quantized_weight = self.calibrator.quantize_weight()
        if self.seg_mode == 'weight':
            self.linear.weight = nn.Parameter(self.splitter.concat_weight(quantized_weight))
            layers, bias = BaseSegmentLinear.split_linear(self.linear, [self.linear.in_features])
            del self.linear
            self.linear = (layers, bias)
        else:
            self.linear.weight = nn.Parameter(quantized_weight[0])
            layers, bias = BaseSegmentLinear.split_linear(self.linear, self.chunksizes)
            del self.linear
            self.linear = (layers, bias)
        
        self.has_calibrated = True

    def forward(self, x):
        if self.seg_mode == 'input':
            input = self.splitter.split_input(x)
        elif self.seg_mode == 'weight':
            input = [x]
        
        quantized_input = self.calibrator.quantize(input)
        layers, bias = self.linear
        if self.seg_mode == 'weight':
            assert len(layers) == 1
            quantized_output = layers[0](quantized_input[0])
            quantized_output_chunks = self.splitter.split_output(quantized_output)

            output_chunks = []
            if False:
                for weight_q, quantized_output_chunk in zip(self.weight_quantizers, quantized_output_chunks):
                    output_chunks.append(FakeQuantizer.dequantize(quantized_output_chunk, self.input_quantizers[0], weight_q, None, None))
            else:
                output_chunks = quantized_output_chunks

            return self.splitter.concat_output(output_chunks) + bias if bias is not None else self.splitter.concat_output(output_chunks)
        
        elif self.seg_mode == 'input':
            quantized_output_chunks = []
            for quantized_input_chunk, layer in zip(quantized_input, layers):
                quantized_output_chunk = layer(quantized_input_chunk)
                quantized_output_chunks.append(quantized_output_chunk)
            
            res = torch.zeros_like(quantized_output_chunks[0])
            for input_q, quantized_output_chunk in zip(self.input_quantizers, quantized_output_chunks):
                if False:
                    res += FakeQuantizer.dequantize(quantized_output_chunk, input_q, self.weight_quantizers[0], None, None)
                else:
                    res += quantized_output_chunk
            return res + bias if bias is not None else res

class SmoothQuantSegmentLinear(BaseSegmentLinear):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None,
                 custom_weight_tensor=None,
                 quant_type=None, input_quant_args=None, weight_quant_args=None,
                 alpha=1.0, dual_s=False):
        
        super().__init__(in_features, out_features, bias, seg_mode, chunks, chunksizes, custom_weight_tensor)

        if quant_type is not None:
            input_gen = lambda: QuantizerRegistry.create(quant_type, **(input_quant_args or {}))
            weight_gen = lambda: QuantizerRegistry.create(quant_type, **(weight_quant_args or {}))
        else:
            input_gen = weight_gen = lambda: None
        
        self.input_quantizers = [input_gen() for _ in range(chunks)]
        self.weight_quantizers = [weight_gen() for _ in range(chunks)]
        
        # dual_s not work now
        assert not dual_s
        self.dual_s = dual_s
        self.calibrator = SmoothQuantCalibrator(self.input_quantizers, self.weight_quantizers, alpha=alpha, dual_s=dual_s)

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
                base = (
                    f"SmoothQuantSegmentLinear(in_features={self.in_features}, out_features={self.out_features}, alpha={self.calibrator.alpha},\n"
                )
                return (
                    base +
                    f"  input_quantizer=({input_q}),\n"
                    f"  weight_quantizer=({weight_q})\n"
                    f")"
                )
            else:
                return (
                    base +
                    f"  input_quantizers=[\n    {input_q}\n  ],\n"
                    f"  weight_quantizers=[\n    {weight_q}\n  ]\n"
                    f")"
                )
        else:
            return base + f"  calib=False\n)"

    def calibrate(self, input_datas):
        if self.seg_mode == 'input':
            inputs = [self.splitter.split_input(input_data) for input_data in input_datas]
            weight = self.linear.weight.split(self.chunksizes, dim=1)
        elif self.seg_mode == 'weight':
            inputs = [[input_data] for input_data in input_datas]
            weight = self.splitter.split_weight(self.linear.weight)
        
        self.calibrator.calibrate(inputs, weight)

        quantized_weight = self.calibrator.quantize_weight()
        if self.seg_mode == 'weight':
            self.linear.weight = nn.Parameter(self.splitter.concat_weight(quantized_weight))
            layers, bias = BaseSegmentLinear.split_linear(self.linear, self.chunksizes, dim=0)
            del self.linear
            self.linear = (layers, bias)
        else:
            self.linear.weight = nn.Parameter(torch.concat(quantized_weight, dim=1))
            layers, bias = BaseSegmentLinear.split_linear(self.linear, self.chunksizes, dim=1)
            del self.linear
            self.linear = (layers, bias)
        
        self.has_calibrated = True

    def forward(self, x):
        if self.seg_mode == 'input':
            input = self.splitter.split_input(x)
        elif self.seg_mode == 'weight':
            input = [x]
        
        quantized_input = self.calibrator.quantize(input)
        layers, bias = self.linear
        if self.seg_mode == 'weight':
            quantized_output_chunks = []
            for quantized_input_chunk, layer in zip(quantized_input, layers):
                quantized_output_chunk = layer(quantized_input_chunk)
                quantized_output_chunks.append(quantized_output_chunk)

            output_chunks = []
            if False:
                for input_q, weight_q, quantized_output_chunk in zip(self.input_quantizers, self.weight_quantizers, quantized_output_chunks):
                    output_chunks.append(FakeQuantizer.dequantize(quantized_output_chunk, input_q, weight_q, None, None))
            else:
                output_chunks = quantized_output_chunks

            c = torch.concat(output_chunks, dim=-1)

            return c + bias if bias is not None else c
        
        elif self.seg_mode == 'input':
            quantized_output_chunks = []
            for quantized_input_chunk, layer in zip(quantized_input, layers):
                quantized_output_chunk = layer(quantized_input_chunk)
                quantized_output_chunks.append(quantized_output_chunk)
            
            res = torch.zeros_like(quantized_output_chunks[0], dtype=x.dtype, device=x.device)
            for input_q, weight_q, quantized_output_chunk in zip(self.input_quantizers, self.weight_quantizers, quantized_output_chunks):
                if False:
                    res += FakeQuantizer.dequantize(quantized_output_chunk, input_q, weight_q, None, None)
                else:
                    res += quantized_output_chunk
            return res + bias if bias is not None else res

def create_segment_linear(dtype: DType, in_features, out_features, **kwargs):
    dtype_map = {
        DType.INT8: DefaultSegmentLinear,
        DType.INT8SMOOTH: SmoothQuantSegmentLinear,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Dtype '{dtype}' not supported")

    if dtype == DType.INT8:
        kwargs.pop('alpha', None)

    return dtype_map[dtype](in_features, out_features, quant_type='int8', **kwargs)

if __name__ == '__main__':
    weight = torch.randn(4, 6)
    print(weight)
    linear = nn.Linear(in_features=6, out_features=4, bias=False)
    linear.weight = nn.Parameter(weight)
    model1 = DefaultSegmentLinear(in_features=6, out_features=4, bias=False, seg_mode='input', chunks=3, chunksizes=[3,2,1], quant_type="int8", custom_weight_tensor=copy.deepcopy(weight))
    model2 = SmoothQuantSegmentLinear(in_features=6, out_features=4, bias=False, seg_mode='input', chunks=3, chunksizes=[3,2,1], quant_type="int8", custom_weight_tensor=copy.deepcopy(weight))
    x = torch.randn(2, 6)
    print("Input:\n", x)
    y = linear(x)
    print("Real:\n", y)
    model1.calibrate([x, torch.randn(2, 6)])
    z = model1(x)
    print("Quant:\n", z)
    model2.calibrate([x, torch.randn(2, 6)])
    s = model2(x)
    print("Smooth:\n", s)