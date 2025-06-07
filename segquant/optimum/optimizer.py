from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from segquant.layers import ext_dict


class BaseOptimizer:
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
    ):
        super().__init__()
        self.seg_mode = seg_mode
        self.chunks = chunks
        self.chunksizes = chunksizes
        self.weight_chunks = weight_chunks
        self.input_calibrators = input_calibrators
        self.weight_calibrators = weight_calibrators
        self.real_quant = real_quant
        self.kernel_type = kernel_type

        self.dual_scale = dual_scale
        if self.real_quant and dual_scale:
            self.func_name = 'gemm_dual_scaled_fn'
        else:
            self.func_name = 'gemm_scaled_fn'

        self.has_calibrated = False

    def _get_funcs(self, input_quantized_indices, weight_quantized_indices):
        if self.real_quant:
            if self.dual_scale:
                funcs = [
                    partial(
                        ext_dict[self.kernel_type][self.func_name],
                        pos_scale_x=self.input_calibrators[input_quantized_indices[i]].pos_scale,
                        neg_scale_x=self.input_calibrators[input_quantized_indices[i]].neg_scale,
                        scale_w=self.weight_calibrators[weight_quantized_indices[i]].scale,
                    )
                    for i in range(self.chunks)
                ]
            else:
                funcs = [
                    partial(
                        ext_dict[self.kernel_type][self.func_name],
                        scale_x=self.input_calibrators[input_quantized_indices[i]].scale,
                        scale_w=self.weight_calibrators[weight_quantized_indices[i]].scale,
                    )
                    for i in range(self.chunks)
                ]
        else:
            def quantize_input_decorator(quantizer):
                def decorator(func):
                    from functools import wraps
                    @wraps(func)
                    def wrapper(input_tensor, weight_tensor, *args, **kwargs):
                        processed_input = quantizer.quantize(input_tensor)
                        return func(processed_input, weight_tensor, *args, **kwargs)
                    return wrapper
                return decorator

            def _create_wrapped_linear_func(q):
                @quantize_input_decorator(q)
                def wrapped_linear(input, weight, bias=None):
                    return F.linear(input, weight, bias)
                return wrapped_linear

            funcs = [
                _create_wrapped_linear_func(
                    self.input_calibrators[input_quantized_indices[i]]
                )
                for i in range(self.chunks)
            ]

        return funcs

    def _calibrate_weights(self, input_chunks):
        for weight_calibrator, weight_chunk, input_copy_or_chunk in zip(
            self.weight_calibrators, self.weight_chunks, input_chunks
        ):
            weight_calibrator.calibrate(weight_chunk, input_data=input_copy_or_chunk)

    def calibrate(self, input_chunks):
        for input_calibrator, input_chunk in zip(self.input_calibrators, input_chunks):
            input_calibrator.calibrate(input_chunk)

        # weight
        self._calibrate_weights(input_chunks)

    def finish_calibrate(self):
        for i, input_calibrator in enumerate(self.input_calibrators):
            input_calibrator.finish_calibrate()
            self.input_calibrators[i] = input_calibrator.quantizer

        for i, weight_calibrator in enumerate(self.weight_calibrators):
            self.weight_chunks[i] = weight_calibrator.finish_calibrate(self.weight_chunks[i])
            self.weight_calibrators[i] = weight_calibrator.quantizer

        self.has_calibrated = True

    def _fake_forward(self, input_chunks):
        pass

class OptimizerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(optimizer_cls):
            cls._registry[name] = optimizer_cls
            return optimizer_cls

        return wrapper

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def create(cls, name, **kwargs):
        optimizer_cls = cls.get(name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{name}' not found in registry.")
        return optimizer_cls(**kwargs)

@OptimizerRegistry.register("default")
class DefaultOptimizer(BaseOptimizer):
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        **kwargs,
    ):
        assert len(input_calibrators) == 1 or len(weight_calibrators) == 1, \
            "Either input_calibrators or weight_calibrators must have a length of 1."
        super().__init__(seg_mode, chunks, chunksizes, weight_chunks, input_calibrators, weight_calibrators, real_quant, dual_scale, kernel_type)

        if seg_mode == 'weight':
            self.input_quantized_indices = [0] * self.chunks
            self.weight_quantized_indices = range(self.chunks)
        elif seg_mode == 'input':
            self.input_quantized_indices = range(self.chunks)
            self.weight_quantized_indices = [0] * self.chunks
        else:
            raise ValueError("seg_mode not found")

        if seg_mode == 'input':
            self.input_chunk_indices = range(self.chunks)
            self.weight_chunk_indices = range(self.chunks)
        elif seg_mode == 'weight':
            self.input_chunk_indices = [0] * self.chunks
            self.weight_chunk_indices = range(self.chunks)
        else:
            raise ValueError("seg_mode not found")

    def __repr__(self):
        if self.real_quant:
            base = (
                f"DefaultOptimizer(type={self.kernel_type}\n"
            )
        else:
            base = (
                "DefaultOptimizer(\n"
            )
        input_q = ",\n    ".join(repr(i) for i in self.input_calibrators)
        weight_q = ",\n    ".join(repr(w) for w in self.weight_calibrators)

        if self.chunks == 1:
            return (
                base + f"    input_quantizer=({input_q}),\n"
                f"    weight_quantizer=({weight_q})\n"
                f")"
            )
        return (
            base + f"    input_quantizers=[\n    {input_q}\n  ],\n"
            f"    weight_quantizers=[\n    {weight_q}\n  ]\n"
            f")"
        )

    def _inv_input_list(self, l):
        if len(l) == 1:
            return l * self.chunks
        return [torch.cat(l, dim=-1)]

    def _calibrate_weights(self, input_chunks):
        inv_input_chunks = self._inv_input_list(input_chunks)
        assert len(inv_input_chunks) == len(self.weight_chunks), "inv_input_chunks failed"

        for weight_calibrator, weight_chunk, input_copy_or_chunk in zip(
            self.weight_calibrators, self.weight_chunks, inv_input_chunks
        ):
            weight_calibrator.calibrate(weight_chunk, input_data=input_copy_or_chunk)

    def finish_calibrate(self):
        super().finish_calibrate()
        if self.seg_mode == 'input':
            # weights need to be splited when input enabled
            assert len(self.weight_chunks) == 1, "weight_chunks size not 1"
            self.weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
            self.weight_chunk_indices = range(self.chunks)

    def _fake_forward(self, input_chunks):
        if self.seg_mode == 'input':
            weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
        elif self.seg_mode == 'weight':
            weight_chunks = self.weight_chunks
        else:
            raise ValueError("seg_mode not found")

        quantized_output_chunks = [
            F.linear(
                input_chunks[self.input_chunk_indices[i]],
                weight_chunks[self.weight_chunk_indices[i]],
            )
            for i in range(self.chunks)
        ]
        return quantized_output_chunks

    def forward(self, input_chunks):
        if not self.has_calibrated:
            return self._fake_forward(input_chunks)

        funcs = self._get_funcs(self.input_quantized_indices, self.weight_quantized_indices)
        quantized_output_chunks = [
            funcs[i](
                input_chunks[self.input_chunk_indices[i]].contiguous(),
                self.weight_chunks[self.weight_chunk_indices[i]].contiguous(),
            )
            for i in range(self.chunks)
        ]

        return quantized_output_chunks

@OptimizerRegistry.register("smooth")
class SmoothOptimizer(BaseOptimizer):
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        alpha=0.5,
        **kwargs,
    ):
        """
        SmoothOptimizer that calibrates input and weight quantizers
        using a smoothing factor (alpha).

        Pipeline: Trace -> Smooth -> Calibrate -> Quantize
        """
        assert len(input_calibrators) == len(weight_calibrators) and \
            len(weight_calibrators) == len(weight_chunks), \
            f"Lengths mismatch: input_calibrators={len(input_calibrators)}, " \
            f"weight_calibrators={len(weight_calibrators)}, " \
            f"weight_chunks={len(weight_chunks)}"
        super().__init__(seg_mode, chunks, chunksizes, weight_chunks, input_calibrators, weight_calibrators, real_quant, dual_scale, kernel_type)

        self.alpha = alpha
        self.max_w = []
        self.max_x = []
        self.s = [None] * self.chunks
        self.has_smoothed = False

        self.input_quantized_indices = range(self.chunks)
        self.weight_quantized_indices = range(self.chunks)

        if seg_mode == 'input':
            self.input_chunk_indices = range(self.chunks)
            self.weight_chunk_indices = range(self.chunks)
        elif seg_mode == 'weight':
            self.input_chunk_indices = [0] * self.chunks
            self.weight_chunk_indices = range(self.chunks)
        else:
            raise ValueError("seg_mode not found")



    def __repr__(self):
        if self.real_quant:
            base = (
                f"SmoothOptimizer(alpha={self.alpha},kernel={self.kernel_type}\n"
            )
        else:
            base = (
                f"SmoothOptimizer(alpha={self.alpha}\n"
            )
        input_q = ",\n      ".join(repr(i) for i in self.input_calibrators)
        weight_q = ",\n      ".join(repr(w) for w in self.weight_calibrators)

        if self.chunks == 1:
            return (
                base + f"    input_quantizer=({input_q}),\n"
                f"    weight_quantizer=({weight_q})\n"
                f")"
            )
        return (
            base + f"    input_quantizers=[\n      {input_q}\n  ],\n"
            f"    weight_quantizers=[\n      {weight_q}\n  ]\n"
            f")"
        )

    def _broadcast_list(self, l):
        if len(l) == 1:
            return l * self.chunks
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

    def _trace_max_w(self, weight_chunks: List[torch.Tensor]):
        if len(self.max_w) < self.chunks:
            for weight_chunk in weight_chunks:
                weight_chunk_max = (
                    weight_chunk.abs()
                    .amax(dim=tuple(range(weight_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
                self.max_w.append(weight_chunk_max)

    def _trace_max_x(self, input_chunks: List[torch.Tensor]):
        for i, input_chunk in enumerate(input_chunks):
            input_chunk_max = (
                input_chunk.abs()
                .amax(dim=tuple(range(input_chunk.ndim - 1)), keepdim=True)
                .squeeze()
            )
            if len(self.max_x) < self.chunks:
                self.max_x.append(input_chunk_max)
            else:
                self.max_x[i] = torch.maximum(self.max_x[i], input_chunk_max)

    def trace(self, input_chunks: List[torch.Tensor]):
        """
        Trace the maximum values of input tensors for calibration.
        This method computes the maximum values of the input tensors and weight tensors
        and stores them for later use in the smoothing and quantization process.
        It assumes that the input tensors are split into chunks and that the weight tensors
        are also split into chunks.
        Args:
            input_chunks: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
        """
        input_broadcast = self._broadcast_list(input_chunks)
        self._trace_max_x(input_broadcast)

    def smooth(self):
        """
        Smooth the input tensors based on the traced max values.

        This method computes the scaling factors (s) for each input tensor chunk
        based on the maximum values of the input and weight tensors.

        It uses the formula:

            s = (max_x ** alpha) / (max_w ** (1 - alpha))
        where max_x is the maximum value of the input tensor chunk and max_w is
        the maximum value of the corresponding weight tensor chunk.
        """
        self._trace_max_w(self.weight_chunks)
        max_x, max_w = self._broadcast_lists(self.max_x, self.max_w)
        for i in range(self.chunks):
            epsilon = 1.0 / (1 << 31)
            s = (max_x[i] ** self.alpha) / (max_w[i] ** (1 - self.alpha))
            s = torch.where(s <= epsilon, torch.ones_like(s), s)
            self.s[i] = torch.clamp(s.to(dtype=torch.float32), min=1e-4, max=1e4)
        del self.max_w
        del self.max_x

        if self.seg_mode == 'input':
            # weights need to be splitted when input enabled
            assert len(self.weight_chunks) == 1, "weight_chunks size not 1"
            self.weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
        elif self.seg_mode == 'weight':
            # input chunks must be splitted first
            self.input_chunk_indices = range(self.chunks)
        else:
            raise ValueError("seg_mode not found")

        # smooth weight
        self._smooth_w(self.weight_chunks)
        self.has_smoothed = True

    def _smooth_w(self, weight_chunks: List[torch.Tensor]):
        for i, (weight_chunk, s) in enumerate(zip(weight_chunks, self.s)):
            weight_chunk.mul_(s.to(weight_chunk.device))
            weight_chunks[i] = weight_chunk

    def _smooth_x(self, input_chunks: List[torch.Tensor]):
        smoothed_input_chunks = []
        input_broadcast = self._broadcast_list(input_chunks)
        for input_chunk, s in zip(input_broadcast, self.s):
            input_smooth = input_chunk / s
            smoothed_input_chunks.append(
                input_smooth.to(dtype=input_chunk.dtype, device=input_chunk.device)
            )
        return smoothed_input_chunks

    def calibrate(self, input_chunks):
        assert self.has_smoothed, 'SmoothOptimizer: linear is not smoothed'
        input_chunks = self._smooth_x(input_chunks) # len == chunks
        super().calibrate(input_chunks)

    def _fake_forward(self, input_chunks):
        if self.has_smoothed:
            # when calibrating, self.weight_chunks must be multiple
            input_chunks = self._smooth_x(input_chunks)
            weight_chunks = self.weight_chunks
        else:
            # when tracing
            if self.seg_mode == 'input':
                weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
            elif self.seg_mode == 'weight':
                weight_chunks = self.weight_chunks
            else:
                raise ValueError("seg_mode not found")

        quantized_output_chunks = [
            F.linear(
                input_chunks[self.input_chunk_indices[i]],
                weight_chunks[self.weight_chunk_indices[i]],
            )
            for i in range(self.chunks)
        ]
        return quantized_output_chunks

    def forward(self, input_chunks):
        if not self.has_calibrated:
            return self._fake_forward(input_chunks)

        funcs = self._get_funcs(self.input_quantized_indices, self.weight_quantized_indices)
        input_chunks = self._smooth_x(input_chunks)
        quantized_output_chunks = [
            funcs[i](
                input_chunks[self.input_chunk_indices[i]].contiguous(),
                self.weight_chunks[self.weight_chunk_indices[i]].contiguous(),
            )
            for i in range(self.chunks)
        ]

        return quantized_output_chunks

@OptimizerRegistry.register("svd")
class SVDOptimizer(SmoothOptimizer):
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        alpha=0.5,
        low_rank=32,
        cpu_storage=False,
        **kwargs,
    ):
        super().__init__(
            seg_mode,
            chunks,
            chunksizes,
            weight_chunks,
            input_calibrators,
            weight_calibrators,
            real_quant,
            dual_scale,
            kernel_type,
            alpha,
        )

        self.low_rank = low_rank
        self.l1s = [None] * self.chunks
        self.l2s = [None] * self.chunks
        self.has_svd = False
        self.device = self.weight_chunks[0].device
        self.cpu_storage = cpu_storage

    def __repr__(self):
        if self.real_quant:
            base = (
                f"SVDOptimizer(alpha={self.alpha}, low_rank={self.low_rank}, kernel={self.kernel_type}\n"
            )
        else:
            base = (
                f"SVDOptimizer(alpha={self.alpha}, low_rank={self.low_rank}\n"
            )
        input_q = ",\n      ".join(repr(i) for i in self.input_calibrators)
        weight_q = ",\n      ".join(repr(w) for w in self.weight_calibrators)

        if self.chunks == 1:
            return (
                base + f"    input_quantizer=({input_q}),\n"
                f"    weight_quantizer=({weight_q})\n"
                f")"
            )
        return (
            base + f"    input_quantizers=[\n      {input_q}\n  ],\n"
            f"    weight_quantizers=[\n      {weight_q}\n  ]\n"
            f")"
        )

    def _svd_w(self, smooth_weight_chunks: List[torch.Tensor]):
        assert self.has_smoothed, 'SVDOptimizer: linear is not smoothed'
        assert len(smooth_weight_chunks) == self.chunks, f'SVDOptimizer: weight chunks must equal to chunks but get [{len(smooth_weight_chunks)}]'

        svd_weight_chunk = []
        for idx, smooth_weight_chunk in enumerate(smooth_weight_chunks):
            print(
                f"torch.linalg.svd: calibrating weight {idx} with shape {smooth_weight_chunk.shape}"
            )
            u, s, vt = torch.linalg.svd(smooth_weight_chunk.t().double())
            if u.shape[1] < self.low_rank or vt.shape[0] < self.low_rank:
                raise ValueError(
                    f"Low-rank dimension {self.low_rank} exceeds layer "
                    f"dimensions {u.shape[1]} and {vt.shape[0]}."
                )
            else:
                # low-rank approximation
                us = u[:, :self.low_rank] * s[:self.low_rank] # U[:, :r] * s[:r] → (m, r)
                vt = vt[:self.low_rank] # Vt[:r, :] → (r, n)
                device = smooth_weight_chunk.device
                dtype = smooth_weight_chunk.dtype
                l1 = us.to(device).to(dtype) # (in, rank)
                l2 = vt.to(device).to(dtype) # (rank, out)
                weight_svd = smooth_weight_chunk.t() - l1 @ l2 # (in, out)
                weight_svd = weight_svd.t() # (out, in)

                if self.cpu_storage:
                    self.l1s[idx] = l1.cpu()
                    self.l2s[idx] = l2.cpu()
                else:
                    self.l1s[idx] = l1
                    self.l2s[idx] = l2

            svd_weight_chunk.append(weight_svd)

        return svd_weight_chunk

    def smooth(self):
        super().smooth()
        self.weight_chunks = self._svd_w(self.weight_chunks)
        self.has_svd = True

    def calibrate(self, input_chunks):
        assert self.has_svd, 'SVDOptimizer: linear is not svd'
        super().calibrate(input_chunks)

    def finish_calibrate(self):
        super().finish_calibrate()
        if self.cpu_storage:
            for idx in self.l1s:
                self.l1s[idx] = self.l1s[idx].to(self.device)
                self.l2s[idx] = self.l2s[idx].to(self.device)


    def _fake_forward(self, input_chunks):
        if self.has_svd:
            # when calibrating, self.weight_chunks must be multiple
            input_chunks = self._smooth_x(input_chunks)
            quantized_output_chunks = [
                input_chunks[self.input_chunk_indices[i]] @ self.l1s[i] @ self.l2s[i] +
                F.linear(
                    input_chunks[self.input_chunk_indices[i]],
                    self.weight_chunks[self.weight_chunk_indices[i]],
                )
                for i in range(self.chunks)
            ]
        else:
            # when tracing
            if self.seg_mode == 'input':
                weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
            elif self.seg_mode == 'weight':
                weight_chunks = self.weight_chunks
            else:
                raise ValueError("seg_mode not found")

            quantized_output_chunks = [
                F.linear(
                    input_chunks[self.input_chunk_indices[i]],
                    weight_chunks[self.weight_chunk_indices[i]],
                )
                for i in range(self.chunks)
            ]
        return quantized_output_chunks

    def forward(self, input_chunks):
        if not self.has_calibrated:
            return self._fake_forward(input_chunks)

        funcs = self._get_funcs(self.input_quantized_indices, self.weight_quantized_indices)
        input_chunks = self._smooth_x(input_chunks)
        quantized_output_chunks = [
            input_chunks[self.input_chunk_indices[i]] @ self.l1s[i] @ self.l2s[i] +
            funcs[i](
                input_chunks[self.input_chunk_indices[i]].contiguous(),
                self.weight_chunks[self.weight_chunk_indices[i]].contiguous(),
            )
            for i in range(self.chunks)
        ]

        return quantized_output_chunks
