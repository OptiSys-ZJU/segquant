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
        self.quantized_weight = None

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

    def calibrate(self, input: List[torch.Tensor], weight: List[torch.Tensor]):
        '''
        input segment: 
            input: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        
        weight segment:
            input: [tensor(batch_size * in_features)]
            weight: [tensor(out_features, split_size1), tensor(out_features, split_size2), ...]
        '''
        
        self._calibrate_input(input)
        if self.quantized_weight is None:
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
    def __init__(self, input_quantizers, weight_quantizers, alpha=0.5, dual_s=False):
        super().__init__(input_quantizers, weight_quantizers)

        assert len(input_quantizers) == len(weight_quantizers)
        self.chunks = len(input_quantizers)

        self.alpha = alpha
        self.dual_s = dual_s

        self.max_w = []
        self.max_x = []  # if dual_s: [(neg_max, pos_max), ...] else: [max, ...]
        self.s = [None] * self.chunks  # if dual_s: [(neg_s, pos_s), ...] else: [s, ...]

        self.quantized_weights = None

    def _trace_max_w(self, weight: List[torch.Tensor]):
        if len(self.max_w) < self.chunks:
            for i, weight_chunk in enumerate(weight):
                weight_chunk_max = weight_chunk.abs().amax(dim=tuple(range(weight_chunk.ndim - 1)), keepdim=True).squeeze()
                self.max_w.append(weight_chunk_max)
    
    def _trace_max_x(self, input: List[torch.Tensor]):
        for i, input_chunk in enumerate(input):
            if self.dual_s:
                neg_mask = input_chunk < 0
                pos_mask = input_chunk > 0
                neg_max = input_chunk[neg_mask].abs().amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True).squeeze() if neg_mask.any() else torch.tensor(0.0, device=input_chunk.device)
                pos_max = input_chunk[pos_mask].abs().amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True).squeeze() if pos_mask.any() else torch.tensor(0.0, device=input_chunk.device)

                if len(self.max_x) < self.chunks:
                    self.max_x.append((neg_max, pos_max))
                else:
                    old_neg, old_pos = self.max_x[i]
                    self.max_x[i] = (torch.maximum(old_neg, neg_max), torch.maximum(old_pos, pos_max))
            else:
                input_chunk_max = input_chunk.abs().amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True).squeeze()
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
            if self.dual_s:
                assert False
            else:
                weight_smooth = (weight_chunk * s).to(dtype=weight_chunk.dtype, device=weight_chunk.device)
                q.calibrate(weight_smooth)
                quantized_weights.append(q.quantize(weight_smooth))
        
        return quantized_weights


    def _calibrate_input(self, input: List[torch.Tensor]):
        input_broadcast = self._broadcast_list(input)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            if self.dual_s:
                assert False
            else:
                q.calibrate(input_chunk / s)

    def trace(self, input: List[torch.Tensor], weight: List[torch.Tensor]):
        self._trace_max_x(input)
        self._trace_max_w(weight)

    def smooth(self):
        max_x, max_w = self._broadcast_lists(self.max_x, self.max_w)
        for i in range(self.chunks):
            epsilon = 1.0 / (1 << 31)
            if self.dual_s:
                neg_x, pos_x = max_x[i]
                s_neg = (neg_x ** self.alpha) / (max_w[i] ** (1 - self.alpha))
                s_pos = (pos_x ** self.alpha) / (max_w[i] ** (1 - self.alpha))
                s_neg = torch.where(s_neg <= epsilon, torch.ones_like(s_neg), s_neg)
                s_pos = torch.where(s_pos <= epsilon, torch.ones_like(s_pos), s_pos)
                self.s[i] = (
                    torch.clamp(s_neg.to(dtype=torch.float32), min=1e-4, max=1e4),
                    torch.clamp(s_pos.to(dtype=torch.float32), min=1e-4, max=1e4),
                )
            else:
                s = (max_x[i] ** self.alpha) / (max_w[i] ** (1 - self.alpha))
                s = torch.where(s <= epsilon, torch.ones_like(s), s)
                self.s[i] = torch.clamp(s.to(dtype=torch.float32), min=1e-4, max=1e4)

    def calibrate(self, input: List[torch.Tensor], weight: List[torch.Tensor]):
        self._calibrate_input(input)
        if self.quantized_weights is None:
            self.quantized_weights = self._calibrate_weight(weight)

    def quantize_weight(self):
        '''
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                - [Tensor(out_features, split_size) * n]
        '''        
        return self.quantized_weights

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
            if self.dual_s:
                input_smooth = torch.where(input_chunk >= 0, input_chunk / s[1], input_chunk / s[0])
            else:
                input_smooth = input_chunk / s
            quantized_input.append(q.quantize(input_smooth.to(input_chunk.dtype)))
        return quantized_input
