from collections import namedtuple
import copy
from typing import Literal
import torch
import torch.nn as nn
from segquant.calibrator.calibrator import DefaultCalibrator, SmoothQuantCalibrator
from segquant.layers.splitter import BaseSplitter
from segquant.quantizers.quantizer import FakeQuantizer, QuantizerRegistry

class BaseSegmentLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None, 
                 custom_weight_tensor=None):
        super().__init__()

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
    
    @staticmethod
    def split_linear(linear, chunksizes, dim=1):
        weight = linear.weight
        bias = linear.bias
        in_features = linear.in_features
        out_features = linear.out_features

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

            layers.append(new_linear)
            start = end

        return layers, bias

    
class DefaultSegmentLinear(BaseSegmentLinear):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None,
                 custom_weight_tensor=None,
                 quant_type=None, quant_args=None):
        super().__init__(in_features, out_features, bias, seg_mode, chunks, chunksizes, custom_weight_tensor)

        if quant_type is not None:
            gen = lambda: QuantizerRegistry.create(quant_type, **(quant_args or {}))
        else:
            gen = lambda: None
        
        if seg_mode == 'input':
            self.input_quantizers = [gen() for _ in range(chunks)]
            self.weight_quantizers = [gen()]
        elif seg_mode == 'weight':
            self.input_quantizers = [gen()]
            self.weight_quantizers = [gen() for _ in range(chunks)]

        self.calibrator = DefaultCalibrator(self.input_quantizers, self.weight_quantizers)

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

            if bias is not None:
                return self.splitter.concat_output(output_chunks) + bias
            else:
                return self.splitter.concat_output(output_chunks)
        
        elif self.seg_mode == 'input':
            quantized_output_chunks = []
            for quantized_input_chunk, layer in zip(quantized_input, layers):
                quantized_output_chunk = layer(quantized_input_chunk)
                quantized_output_chunks.append(quantized_output_chunk)
            
            if bias is None:
                res = torch.zeros_like(quantized_output_chunks[0])
            else:
                res = bias

            for input_q, quantized_output_chunk in zip(self.input_quantizers, quantized_output_chunks):
                if False:
                    res += FakeQuantizer.dequantize(quantized_output_chunk, input_q, self.weight_quantizers[0], None, None)
                else:
                    res += quantized_output_chunk
            return res


class SmoothQuantSegmentLinear(BaseSegmentLinear):
    def __init__(self, in_features, out_features, bias=True, 
                 seg_mode: Literal['input', 'weight'] = 'weight', chunks=1, chunksizes=None,
                 custom_weight_tensor=None,
                 quant_type=None, quant_args=None,
                 alpha=0.5):
        
        super().__init__(in_features, out_features, bias, seg_mode, chunks, chunksizes, custom_weight_tensor)

        if quant_type is not None:
            gen = lambda: QuantizerRegistry.create(quant_type, **(quant_args or {}))
        else:
            gen = lambda: None
        
        self.input_quantizers = [gen() for _ in range(chunks)]
        self.weight_quantizers = [gen() for _ in range(chunks)]

        self.calibrator = SmoothQuantCalibrator(self.input_quantizers, self.weight_quantizers, alpha=alpha)

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

            if bias is not None:
                return torch.concat(output_chunks, dim=1) + bias
            else:
                return torch.concat(output_chunks, dim=1)
        
        elif self.seg_mode == 'input':
            quantized_output_chunks = []
            for quantized_input_chunk, layer in zip(quantized_input, layers):
                quantized_output_chunk = layer(quantized_input_chunk)
                quantized_output_chunks.append(quantized_output_chunk)
            
            if bias is None:
                res = torch.zeros_like(quantized_output_chunks[0])
            else:
                res = bias

            for input_q, weight_q, quantized_output_chunk in zip(self.input_quantizers, self.weight_quantizers, quantized_output_chunks):
                if False:
                    res += FakeQuantizer.dequantize(quantized_output_chunk, input_q, weight_q, None, None)
                else:
                    res += quantized_output_chunk
            return res


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


    # weight = torch.randn(6, 4)
    # print(weight)
    # linear = nn.Linear(in_features=4, out_features=6, bias=False)
    # linear.weight = nn.Parameter(weight)
    # model1 = DefaultSegmentLinear(in_features=4, out_features=6, bias=False, seg_mode='weight', chunks=3, chunksizes=[3,2,1], quant_type="int8", custom_weight_tensor=copy.deepcopy(weight))
    # model2 = SmoothQuantSegmentLinear(in_features=4, out_features=6, bias=False, seg_mode='weight', chunks=3, chunksizes=[3,2,1], quant_type="int8", custom_weight_tensor=copy.deepcopy(weight))
    # x = torch.randn(2, 4)
    # print("Input:\n", x)
    # y = linear(x)
    # print("Real:\n", y)
    # model1.calibrate([x])
    # z = model1(x)
    # print("Quant:\n", z)
    # model2.calibrate([x])
    # s = model2(x)
    # print("Smooth:\n", s)