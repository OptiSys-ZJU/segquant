"""
This module provides classes for calibrating and quantizing tensors for neural network models.
The module includes:
- `BaseCalibrator`: A base class for implementing custom calibration strategies.
- `DefaultCalibrator`: A default implementation for calibrating input and weight quantizers.
- `SmoothQuantCalibrator`: An advanced calibrator that uses a smoothing factor.
These classes are designed to work with quantizers to 
optimize the representation of input and weight tensors
for efficient computation, particularly in scenarios like model compression 
or deployment on resource-constrained devices.
Classes:
    - BaseCalibrator: Abstract base class for calibrators.
    - DefaultCalibrator: Implements a default calibration strategy.
    - SmoothQuantCalibrator: Implements a smoothing-based calibration strategy.
Dependencies:
    - torch: PyTorch library for tensor operations.
    - BaseQuantizer: A base class for quantizers, imported from `segquant.quantizers.quantizer`.
Usage:
    These calibrators are intended to be used in conjunction with 
    quantizers to process input and weight tensors for neural network models. 
    They provide methods for tracing, calibrating, and quantizing tensors.
"""

from typing import List
import torch
from segquant.quantizers.quantizer import BaseQuantizer

class BaseCalibrator:
    """
    Base class for calibrators.
    This class is used to calibrate quantizers for input and weight tensors.
    It is designed to be subclassed for specific calibration strategies.
    """
    def __init__(
        self,
        input_quantizers: List[BaseQuantizer],
        weight_quantizers: List[BaseQuantizer],
    ):
        self.input_quantizers = input_quantizers
        self.weight_quantizers = weight_quantizers


class DefaultCalibrator(BaseCalibrator):
    """
    Default calibrator that calibrates input and weight quantizers.
    It supports two modes: input_t-segment and weight-segment.
    In input_t-segment mode, it calibrates input quantizers with input tensors
    and weight quantizers with weight tensors.
    In weight-segment mode, it calibrates input quantizers with weight tensors
    and weight quantizers with input tensors.
    The class assumes that either input_quantizers or weight_quantizers has only one quantizer.
    """
    def __init__(
        self,
        input_quantizers: List[BaseQuantizer],
        weight_quantizers: List[BaseQuantizer],
    ):
        super().__init__(input_quantizers, weight_quantizers)

        assert len(self.input_quantizers) == 1 or len(self.weight_quantizers) == 1
        self.chunks = (
            len(self.weight_quantizers)
            if len(self.input_quantizers) == 1
            else len(self.input_quantizers)
        )
        self.quantized_weight = None

    def _calibrate_weight(self, weight: List[torch.Tensor]):
        assert len(weight) == len(self.weight_quantizers)

        quantized_weights = []
        for q, weight_chunk in zip(self.weight_quantizers, weight):
            q.calibrate(weight_chunk)
            quantized_weights.append(q.quantize(weight_chunk))

        return quantized_weights

    def _calibrate_input(self, input_t: List[torch.Tensor]):
        assert len(input_t) == len(self.input_quantizers)

        for q, input_chunk in zip(self.input_quantizers, input_t):
            q.calibrate(input_chunk)

    def calibrate(self, input_t: List[torch.Tensor], weight: List[torch.Tensor]):
        """
        input_t segment: 
            input_t: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        
        weight segment:
            input_t: [tensor(batch_size * in_features)]
            weight: [tensor(out_features, split_size1), tensor(out_features, split_size2), ...]
        """

        self._calibrate_input(input_t)
        if self.quantized_weight is None:
            self.quantized_weight = self._calibrate_weight(weight)

    def quantize_weight(self):
        """
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(out_features, in_features)]

                when weight-segment:
                    - [Tensor(out_features, split_size) * n]
        """
        return self.quantized_weight

    def quantize(self, input_t: List[torch.Tensor]):
        """
        Args:
            input_t(when input_t-segment): [tensor(batch_size * split_size1), 
                                            tensor(batch_size * split_size2), ...]
            input_t(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(batch_size * split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * in_features)]
        """
        quantized_input = []
        for q, input_chunk in zip(self.input_quantizers, input_t):
            quantized_input.append(q.quantize(input_chunk))

        return quantized_input


class SmoothQuantCalibrator(BaseCalibrator):
    """
    SmoothQuant calibrator that calibrates input and weight quantizers
    using a smoothing factor (alpha) and supports dual scaling (dual_s).

    Pipeline: Trace -> Smooth -> Calibrate -> Quantize
    """
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
            for _, weight_chunk in enumerate(weight):
                weight_chunk_max = (
                    weight_chunk.abs()
                    .amax(dim=tuple(range(weight_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
                self.max_w.append(weight_chunk_max)

    def _trace_max_x(self, input_t: List[torch.Tensor]):
        for i, input_chunk in enumerate(input_t):
            if self.dual_s:
                neg_mask = input_chunk < 0
                pos_mask = input_chunk > 0
                neg_max = (
                    input_chunk[neg_mask]
                    .abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                    if neg_mask.any()
                    else torch.tensor(0.0, device=input_chunk.device)
                )
                pos_max = (
                    input_chunk[pos_mask]
                    .abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                    if pos_mask.any()
                    else torch.tensor(0.0, device=input_chunk.device)
                )

                if len(self.max_x) < self.chunks:
                    self.max_x.append((neg_max, pos_max))
                else:
                    old_neg, old_pos = self.max_x[i]
                    self.max_x[i] = (
                        torch.maximum(old_neg, neg_max),
                        torch.maximum(old_pos, pos_max),
                    )
            else:
                input_chunk_max = (
                    input_chunk.abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
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
                weight_smooth = (weight_chunk * s).to(
                    dtype=weight_chunk.dtype, device=weight_chunk.device
                )
                q.calibrate(weight_smooth)
                quantized_weights.append(q.quantize(weight_smooth))

        return quantized_weights

    def _calibrate_input(self, input_t: List[torch.Tensor]):
        input_broadcast = self._broadcast_list(input_t)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            if self.dual_s:
                assert False
            else:
                q.calibrate(input_chunk / s)

    def trace(self, input_t: List[torch.Tensor], weight: List[torch.Tensor]):
        """
        Trace the maximum values of input and weight tensors for calibration.
        This method computes the maximum values of the input tensors and weight tensors
        and stores them for later use in the smoothing and quantization process.
        It assumes that the input tensors are split into chunks and that the weight tensors
        are also split into chunks.
        Args:
            input_t: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        """
        input_broadcast = self._broadcast_list(input_t)
        self._trace_max_x(input_broadcast)
        self._trace_max_w(weight)

    def smooth(self):
        """
        Smooth the input tensors based on the traced max values.

        This method computes the scaling factors (s) for each input tensor chunk
        based on the maximum values of the input and weight tensors.

        It uses the formula:

            s = (max_x ** alpha) / (max_w ** (1 - alpha))
        where max_x is the maximum value of the input tensor chunk and max_w is
        the maximum value of the corresponding weight tensor chunk.

        If dual_s is True (not implemented now), it computes separate 
        scaling factors for negative and positive
        input values.
        """
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

    def calibrate(self, input_t: List[torch.Tensor], weight: List[torch.Tensor]):
        """
        Calibrate the input and weight quantizers using the provided input and weight tensors.
        This method first traces the maximum values of the input and weight tensors,
        then smooths the input tensors based on the traced values, and finally calibrates
        the input and weight quantizers.
        Args:
            input_t: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        """
        self._calibrate_input(input_t)
        if self.quantized_weights is None:
            self.quantized_weights = self._calibrate_weight(weight)

    def quantize_weight(self):
        """
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                - [Tensor(out_features, split_size) * n]
        """
        return self.quantized_weights

    def smooth_input(self, input_t: List[torch.Tensor]):
        """
        Args:
            input_t(when input_t-segment): [tensor(batch_size * split_size1), 
                                            tensor(batch_size * split_size2), ...]
            input_t(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]
        """

        smoothed_input = []
        input_broadcast = self._broadcast_list(input_t)
        for input_chunk, s in zip(input_broadcast, self.s):
            if self.dual_s:
                input_smooth = torch.where(
                    input_chunk >= 0, input_chunk / s[1], input_chunk / s[0]
                )
            else:
                input_smooth = input_chunk / s
            smoothed_input.append(input_smooth.to(input_chunk.dtype))
        return smoothed_input

    def quantize(self, input_t: List[torch.Tensor]):
        """
        Args:
            input_t(when input_t-segment): [tensor(batch_size * split_size1), 
                                            tensor(batch_size * split_size2), ...]
            input_t(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]
        """

        quantized_input = []
        input_broadcast = self._broadcast_list(input_t)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            if self.dual_s:
                input_smooth = torch.where(
                    input_chunk >= 0, input_chunk / s[1], input_chunk / s[0]
                )
            else:
                input_smooth = input_chunk / s
            quantized_input.append(q.quantize(input_smooth.to(input_chunk.dtype)))
        return quantized_input


class SVDQuantCalibrator(BaseCalibrator):
    """
    SVDQuant calibrator that calibrates input and weight quantizers
    using a smoothing factor (alpha), low-rank branch (low_rank) and supports dual scaling (dual_s).

    Pipeline: Trace -> Smooth -> Calibrate -> Quantize
    """
    def __init__(self, input_quantizers, weight_quantizers, alpha=0.5, low_rank=32, dual_s=False):
        super().__init__(input_quantizers, weight_quantizers)

        assert len(input_quantizers) == len(weight_quantizers)
        self.chunks = len(input_quantizers)

        self.alpha = alpha
        self.low_rank = low_rank
        self.dual_s = dual_s

        self.max_w = []
        self.max_x = []  # if dual_s: [(neg_max, pos_max), ...] else: [max, ...]
        self.s = [None] * self.chunks  # if dual_s: [(neg_s, pos_s), ...] else: [s, ...]

        self.quantized_weights = None

    def _trace_max_w(self, weight: List[torch.Tensor]):
        if len(self.max_w) < self.chunks:
            for _, weight_chunk in enumerate(weight):
                weight_chunk_max = (
                    weight_chunk.abs()
                    .amax(dim=tuple(range(weight_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
                self.max_w.append(weight_chunk_max)

    def _trace_max_x(self, input_t: List[torch.Tensor]):
        for i, input_chunk in enumerate(input_t):
            if self.dual_s:
                neg_mask = input_chunk < 0
                pos_mask = input_chunk > 0
                neg_max = (
                    input_chunk[neg_mask]
                    .abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                    if neg_mask.any()
                    else torch.tensor(0.0, device=input_chunk.device)
                )
                pos_max = (
                    input_chunk[pos_mask]
                    .abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                    if pos_mask.any()
                    else torch.tensor(0.0, device=input_chunk.device)
                )

                if len(self.max_x) < self.chunks:
                    self.max_x.append((neg_max, pos_max))
                else:
                    old_neg, old_pos = self.max_x[i]
                    self.max_x[i] = (
                        torch.maximum(old_neg, neg_max),
                        torch.maximum(old_pos, pos_max),
                    )
            else:
                input_chunk_max = (
                    input_chunk.abs()
                    .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
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
        l1s = [None] * self.chunks
        l2s = [None] * self.chunks
        weight_broadcast = self._broadcast_list(weight)
        for idx, (q, weight_chunk, s) in enumerate(zip(self.weight_quantizers, weight_broadcast, self.s)):
            if self.dual_s:
                assert False
            else:
                weight_smooth = (weight_chunk * s).to(
                    dtype=weight_chunk.dtype, device=weight_chunk.device
                )

                # svd
                print(f'torch.linalg.svd: calibrating weight {idx} with shape {weight_smooth.shape}')
                u, s, vt = torch.linalg.svd(weight_smooth.t().double())
                if u.shape[1] < self.low_rank or vt.shape[0] < self.low_rank:
                    raise ValueError(
                        f'Low-rank dimension {self.low_rank} exceeds layer dimensions {u.shape[1]} and {vt.shape[0]}.'
                    )
                else:
                    # low-rank approximation
                    us = u[:, :self.low_rank] * s[:self.low_rank] # U[:, :r] * s[:r] → (m, r)
                    vt = vt[:self.low_rank] # Vt[:r, :] → (r, n)
                    device = weight_smooth.device
                    dtype = weight_smooth.dtype
                    l1 = us.to(device).to(dtype) # (m, r)
                    l2 = vt.to(device).to(dtype) # (r, n)
                    weight_svd = weight_smooth.t() - l1 @ l2
                    weight_svd = weight_svd.t()
                    l1s[idx] = l1
                    l2s[idx] = l2

                q.calibrate(weight_svd)
                quantized_weights.append(q.quantize(weight_svd))

        return quantized_weights, l1s, l2s

    def _calibrate_input(self, input_t: List[torch.Tensor]):
        input_broadcast = self._broadcast_list(input_t)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            if self.dual_s:
                assert False
            else:
                q.calibrate(input_chunk / s)

    def trace(self, input_t: List[torch.Tensor], weight: List[torch.Tensor]):
        """
        Trace the maximum values of input and weight tensors for calibration.
        This method computes the maximum values of the input tensors and weight tensors
        and stores them for later use in the smoothing and quantization process.
        It assumes that the input tensors are split into chunks and that the weight tensors
        are also split into chunks.
        Args:
            input_t: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        """
        input_broadcast = self._broadcast_list(input_t)
        self._trace_max_x(input_broadcast)
        self._trace_max_w(weight)

    def smooth(self):
        """
        Smooth the input tensors based on the traced max values.

        This method computes the scaling factors (s) for each input tensor chunk
        based on the maximum values of the input and weight tensors.

        It uses the formula:

            s = (max_x ** alpha) / (max_w ** (1 - alpha))
        where max_x is the maximum value of the input tensor chunk and max_w is
        the maximum value of the corresponding weight tensor chunk.

        If dual_s is True (not implemented now), it computes separate 
        scaling factors for negative and positive
        input values.
        """
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

    def calibrate(self, input_t: List[torch.Tensor], weight: List[torch.Tensor]):
        """
        Calibrate the input and weight quantizers using the provided input and weight tensors.
        This method first traces the maximum values of the input and weight tensors,
        then smooths the input tensors based on the traced values, and finally calibrates
        the input and weight quantizers.
        Args:
            input_t: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
            weight: [tensor(out_features, in_features)]
        """
        self._calibrate_input(input_t)
        if self.quantized_weights is None:
            self.quantized_weights = self._calibrate_weight(weight)

    def quantize_weight(self):
        """
        Returns:
            Union[list[Tensor]], list[Tensor]]]: 
                - ([Tensor(out_features, split_size) * n], l1s, l2s)
        """
        return self.quantized_weights

    def smooth_input(self, input_t: List[torch.Tensor]):
        """
        Args:
            input_t(when input_t-segment): [tensor(batch_size * split_size1), 
                                            tensor(batch_size * split_size2), ...]
            input_t(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]
        """

        smoothed_input = []
        input_broadcast = self._broadcast_list(input_t)
        for input_chunk, s in zip(input_broadcast, self.s):
            if self.dual_s:
                input_smooth = torch.where(
                    input_chunk >= 0, input_chunk / s[1], input_chunk / s[0]
                )
            else:
                input_smooth = input_chunk / s
            smoothed_input.append(input_smooth.to(input_chunk.dtype))
        return smoothed_input

    def quantize(self, input_t: List[torch.Tensor]):
        """
        Args:
            input_t(when input_t-segment): [tensor(batch_size * split_size1), 
                                            tensor(batch_size * split_size2), ...]
            input_t(when weight-segment): [tensor(batch_size * in_features)]
        
        Returns:
            Union[Tuple[list[Tensor], list[Tensor]], Tuple[list[Tensor], list[Tensor]]]: 
                when input_t-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]

                when weight-segment:
                    - [Tensor(batch_size * split_size) * n]
                    - [Tensor(out_features, split_size) * n]
        """

        quantized_input = []
        input_broadcast = self._broadcast_list(input_t)
        for q, input_chunk, s in zip(self.input_quantizers, input_broadcast, self.s):
            if self.dual_s:
                input_smooth = torch.where(
                    input_chunk >= 0, input_chunk / s[1], input_chunk / s[0]
                )
            else:
                input_smooth = input_chunk / s
            quantized_input.append(q.quantize(input_smooth.to(input_chunk.dtype)))
        return quantized_input
