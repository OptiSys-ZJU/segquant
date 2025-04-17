from segquant.quantizers.quantizer import BaseQuantizer
from typing import List
import torch

class BaseCalibrator:
    def __init__(self, input_quantizers: List[BaseQuantizer], weight_quantizers: List[BaseQuantizer]):
        self.input_quantizers = input_quantizers
        self.weight_quantizers = weight_quantizers

class DefaultCalibrator(BaseCalibrator):
    def __init__(self, input_quantizers: List[BaseQuantizer], weight_quantizers: List[BaseQuantizer]):
        super().__init__(input_quantizers, weight_quantizers)

        assert len(self.input_quantizers) == 1 or len(self.weight_quantizers) == 1
        self.chunks = len(self.weight_quantizers) if len(self.input_quantizers) == 1 else len(self.input_quantizers)

    def _calibrate_weight(self, weight: List[torch.Tensor]):
        assert len(weight) == len(self.weight_quantizers)
        
        quantized_weights = []
        for q, weight_chunk in zip(self.weight_quantizers, weight):
            q.calibrate(weight_chunk)
            quantized_weights.append(q.quantize(weight_chunk))
        
        return quantized_weights
    
    def _calibrate_input(self, input: List[torch.Tensor]):
        assert len(input) == len(self.input_quantizers)

        for q, input_chunk in zip(self.input_quantizers, input):
            q.calibrate(input_chunk)

    def calibrate(self, inputs: List[List[torch.Tensor]], weight: List[torch.Tensor]):
        '''
        input segment: 
            input: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        
        weight segment:
            input: [tensor(batch_size * in_features)]
            weight: [tensor(out_features, split_size1), tensor(out_features, split_size2), ...]
        '''
        
        for input in inputs:
            self._calibrate_input(input)
        self.quantized_weight = self._calibrate_weight(weight)
    
    def quantize_weight(self):
        '''
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                when input-segment:
                    - [Tensor(out_features, in_features)]

                when weight-segment:
                    - [Tensor(out_features, split_size) * n]
        '''
        return self.quantized_weight

    def quantize(self, input: List[torch.Tensor]):
        '''
        Args:
            input(when input-segment): [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            input(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input-segment:
                    - [Tensor(batch_size * split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * in_features)]
        '''
        quantized_input = []
        for q, input_chunk in zip(self.input_quantizers, input):
            quantized_input.append(q.quantize(input_chunk))
        
        return quantized_input

class SmoothQuantCalibrator(BaseCalibrator):
    def __init__(self, input_quantizers, weight_quantizers, alpha=0.5):
        super().__init__(input_quantizers, weight_quantizers)

        assert len(input_quantizers) == len(weight_quantizers)
        self.chunks = len(input_quantizers)

        self.alpha = alpha
        self.max_w = []
        self.max_x = []
        self.s = [None] * self.chunks

    def _trace_max_w(self, weight: List[torch.Tensor]):
        for i, weight_chunk in enumerate(weight):
            self.max_w.append(weight_chunk.abs().max(dim=0).values.unsqueeze(0))
    
    def _trace_max_x(self, input: List[torch.Tensor]):
        for i, input_chunk in enumerate(input):
            input_chunk_max = input_chunk.abs().max(dim=0).values.unsqueeze(0)
            if len(self.max_x) < self.chunks:
                self.max_x.append(input_chunk_max)
            else:
                self.max_x[i] = torch.maximum(self.max_x[i], input_chunk_max)

    def _broadcast_list(self, l):
        if len(l) == 1:
            return l * self.chunks
        else:
            return l

    def _broadcast_lists(self, a, b):
        if len(a) == 1 and len(b) > 1:
            return a * len(b), b
        elif len(b) == 1 and len(a) > 1:
            return a, b * len(a)
        elif len(a) == len(b):
            return a, b
        else:
            raise ValueError("List length not 1")

    def _calibrate_weight(self, weight: List[torch.Tensor]):
        quantized_weights = []
        weight_broadcast = self._broadcast_list(weight)
        for q, weight_chunk, s in zip(self.weight_quantizers, weight_broadcast, self.s):
            weight_smooth = weight_chunk * s
            q.calibrate(weight_smooth)
            quantized_weights.append(q.quantize(weight_smooth))
        
        return quantized_weights
    
    def _calibrate_input(self, input: List[torch.Tensor]):
        input_broadcast = self._broadcast_list(input)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            input_smooth = input_chunk / s
            q.calibrate(input_smooth)

    def calibrate(self, inputs: List[List[torch.Tensor]], weight: List[torch.Tensor]):
        '''
        input segment: 
            input: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, split_size1), tensor(out_features, split_size2), ...]
        
        weight segment:
            input: [tensor(batch_size * in_features)]
            weight: [tensor(out_features, split_size1), tensor(out_features, split_size2), ...]
        '''
        
        for input in inputs:
            self._trace_max_x(input)
        self._trace_max_w(weight)

        max_x, max_w = self._broadcast_lists(self.max_x, self.max_w)
        for i in range(self.chunks):
            self.s[i] = (max_x[i] ** self.alpha) / (max_w[i] ** (1 - self.alpha))

        for input in inputs:
            self._calibrate_input(input)
        self.quantized_weight = self._calibrate_weight(weight)

    def quantize_weight(self):
        '''
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                - [Tensor(out_features, split_size) * n]
        '''
        return self.quantized_weight

    def quantize(self, input: List[torch.Tensor]):
        '''
        Args:
            input(when input-segment): [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            input(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]
        '''

        quantized_input = []
        input_broadcast = self._broadcast_list(input)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            input_smooth = input_chunk / s
            quantized_input.append(q.quantize(input_smooth))
        
        return quantized_input
