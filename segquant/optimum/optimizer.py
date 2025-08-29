from functools import partial
import math
from typing import List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segquant.layers import ext_dict
from segquant.utils.GivensOrthLayer import CayleyRegistry
from segquant.utils.im2col import im2col_input, im2col_weight


class BaseOptimizer:
    def __init__(
        self,
        layer_mode,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        layer_kwargs={},
    ):
        super().__init__()
        self.layer_mode = layer_mode
        self.seg_mode = seg_mode
        self.chunks = chunks
        self.chunksizes = chunksizes
        self.weight_chunks = weight_chunks
        self.input_calibrators = input_calibrators
        self.weight_calibrators = weight_calibrators
        self.real_quant = real_quant
        self.kernel_type = kernel_type

        if "kernel_size" in layer_kwargs:
            self.kernel_size = layer_kwargs["kernel_size"]
            layer_kwargs.pop("kernel_size")
        else:
            self.kernel_size = None
        self.layer_kwargs = layer_kwargs

        self.dual_scale = dual_scale
        if self.layer_mode == 'linear':
            if self.real_quant and dual_scale:
                self.func_name = 'gemm_dual_scaled_fn'
            else:
                self.func_name = 'gemm_scaled_fn'
        elif self.layer_mode == 'conv2d':
            if self.real_quant and dual_scale:
                self.func_name = 'conv2d_dual_scaled_fn'
            else:
                self.func_name = 'conv2d_scaled_fn'
        else:
            raise ValueError(f"Unsupported layer mode: {self.layer_mode}")

        self.has_calibrated = False
    
    def _broadcast_list(self, l):
        if len(l) == 1:
            return l * self.chunks
        return l

    def _im2col_weight(self, x):
        assert x.dim() == 4, "Weight tensor x must be 4D for im2col_weight."
        return im2col_weight(x, groups=self.layer_kwargs['groups']) # [groups, out_per_group, in_per_group*kh*kw]

    def _im2col_input(self, x):
        return im2col_input(
            x,
            kernel_size=self.kernel_size,
            stride=self.layer_kwargs["stride"],
            padding=self.layer_kwargs["padding"],
            dilation=self.layer_kwargs["dilation"],
            groups=self.layer_kwargs["groups"],
        )  # [groups, batch*Hout*Wout, in_per_group*kh*kw]

    def _get_funcs(self, input_quantized_indices, weight_quantized_indices):
        if self.real_quant:
            def tensor_wrapper(x):
                if isinstance(x, torch.Tensor):
                    return x.to(dtype=torch.float32).contiguous()
                else:
                    return x

            if self.dual_scale:
                funcs = [
                    partial(
                        ext_dict[self.kernel_type][self.func_name],
                        pos_scale_x=tensor_wrapper(self.input_calibrators[input_quantized_indices[i]].pos_scale),
                        neg_scale_x=tensor_wrapper(self.input_calibrators[input_quantized_indices[i]].neg_scale),
                        scale_w=tensor_wrapper(self.weight_calibrators[weight_quantized_indices[i]].scale),
                    )
                    for i in range(self.chunks)
                ]
            else:
                funcs = [
                    partial(
                        ext_dict[self.kernel_type][self.func_name],
                        scale_x=tensor_wrapper(self.input_calibrators[input_quantized_indices[i]].scale),
                        scale_w=tensor_wrapper(self.weight_calibrators[weight_quantized_indices[i]].scale),
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

            def _create_wrapped_layer_func(q):
                @quantize_input_decorator(q)
                def wrapped_layer(input, weight):
                    if self.layer_mode == 'conv2d':
                        return F.conv2d(input, weight, **self.layer_kwargs)
                    elif self.layer_mode == 'linear':
                        return F.linear(input, weight, **self.layer_kwargs)
                return wrapped_layer

            funcs = [
                _create_wrapped_layer_func(
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
    
    def to(self, device):
        self.weight_chunks = [chunk.to(device) for chunk in self.weight_chunks]
        self.input_calibrators = [calibrator.to(device) for calibrator in self.input_calibrators]
        self.weight_calibrators = [calibrator.to(device) for calibrator in self.weight_calibrators]

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
        layer_mode,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        layer_kwargs={},
        **kwargs,
    ):
        assert len(input_calibrators) == 1 or len(weight_calibrators) == 1, \
            "Either input_calibrators or weight_calibrators must have a length of 1."
        super().__init__(
            layer_mode,
            seg_mode,
            chunks,
            chunksizes,
            weight_chunks,
            input_calibrators,
            weight_calibrators,
            real_quant,
            dual_scale,
            kernel_type,
            layer_kwargs,
        )

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
            self.weight_chunk_indices = [0] * self.chunks
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

    def forward(self, input_chunks):
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
        search_alpha_config=None,
        verbose=False,
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
        )

        self.max_w = []
        self.max_x = []
        self.s = [None] * self.chunks
        self.alpha = [alpha] * self.chunks
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

        self.has_traced_w = False

        self.verbose = verbose

        if search_alpha_config is None or not search_alpha_config['enable']:
            self.search_alpha = False
        else:
            # alpha search
            self.search_alpha = True
            alpha_range = np.arange(
                search_alpha_config["min"],
                search_alpha_config["max"] + 1e-8,
                search_alpha_config["step"],
            )
            self.candidate_alphas = [deque(alpha_range) for _ in range(self.chunks)]
            self.alpha = [alphas.popleft() for alphas in self.candidate_alphas]
            self.opt_err = [float('inf')] * self.chunks
            self.opt_alpha = [a for a in self.alpha]

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

        if not self.has_traced_w:
            self._trace_max_w(self.weight_chunks)
            self.has_traced_w = True
    
    def to(self, device):
        super().to(device)
        self.s = [s.to(device) for s in self.s]

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
        max_x, max_w = self._broadcast_lists(self.max_x, self.max_w)
        for i in range(self.chunks):
            epsilon = 1.0 / (1 << 31)
            s = (max_x[i] ** self.alpha[i]) / (max_w[i] ** (1 - self.alpha[i]))
            s = torch.where(s <= epsilon, torch.ones_like(s), s)
            self.s[i] = torch.clamp(s.to(dtype=torch.float32), min=1e-4, max=1e4)

        if self.seg_mode == 'input':
            # weights need to be splitted when input enabled
            assert len(self.weight_chunks) == 1, "weight_chunks size not 1"
            self.weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
        elif self.seg_mode == 'weight':
            pass
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

    def _clean(self):
        for i, input_calibrator in enumerate(self.input_calibrators):
            self.input_calibrators[i] = input_calibrator.quantizer
        for i, weight_calibrator in enumerate(self.weight_calibrators):
            self.weight_calibrators[i] = weight_calibrator.quantizer
        
        self.max_w = None
        self.max_x = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finish_calibrate(self):
        for i, input_calibrator in enumerate(self.input_calibrators):
            input_calibrator.finish_calibrate()

        for i, weight_calibrator in enumerate(self.weight_calibrators):
            self.weight_chunks[i] = weight_calibrator.finish_calibrate(self.weight_chunks[i])

        if self.seg_mode == 'weight':
            # input chunks must be splitted first
            self.input_chunk_indices = range(self.chunks)
        self.has_calibrated = True

        if not self.search_alpha:
            self._clean()

    def forward(self, input_chunks):
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

    def search_step(self, errs, origin_weight):
        assert len(errs) == self.chunks, "errlist length must be chunks"

        for i in range(self.chunks):
            this_err = errs[i]
            if this_err < self.opt_err[i]:
                self.opt_err[i] = this_err
                self.opt_alpha[i] = self.alpha[i]
            
            if self.verbose:
                print(
                    f"Chunk {i}: current err={this_err:.4f}, best err={self.opt_err[i]:.4f}, best alpha={self.opt_alpha[i]:.4f}"
                )

        # iterate alpha
        self.alpha = [alphas.popleft() for alphas in self.candidate_alphas]
        if self.verbose:
            print(f"Next alphas to try: {self.alpha[0]:.4f}")

        # reset optimizer
        self._reset(origin_weight=origin_weight)

        if all(len(alphas) == 0 for alphas in self.candidate_alphas):
            self._finish_search()
            return True
        return False

    def _finish_search(self):
        if self.verbose:
            print(f"Best alphas found: {self.opt_alpha}")
        self.alpha = self.opt_alpha
        del self.opt_alpha
        del self.opt_err
        del self.candidate_alphas
        self.search_alpha = False

    def _reset(self, origin_weight):
        if self.seg_mode == 'weight':
            self.input_chunk_indices = [0] * self.chunks
            weight_chunks = origin_weight.split(self.chunksizes, dim=0)
            for i in range(len(self.weight_chunks)):
                self.weight_chunks[i] = weight_chunks[i].clone()
        elif self.seg_mode == 'input':
            self.weight_chunks = [origin_weight.clone()]

        for input_calibrator in self.input_calibrators:
            input_calibrator.reset()
        for weight_calibrator in self.weight_calibrators:
            weight_calibrator.reset()
        self.has_smoothed = False
        self.has_calibrated = False


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
        search_alpha_config=None,
        verbose=False,
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
            search_alpha_config,
            verbose,
        )

        self.low_rank = low_rank
        self.l1s = [None] * self.chunks
        self.l2s = [None] * self.chunks
        self.has_svd = False
        self.precision = kwargs.get('precision', 'float64')
        if self.precision not in ['float32', 'float64']:
            raise ValueError(f"Unsupported precision: {self.precision}. Use 'float32' or 'float64'.")

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
    
    def to(self, device):
        super().to(device)
        self.l1s = [l1.to(device) for l1 in self.l1s]
        self.l2s = [l2.to(device) for l2 in self.l2s]

    def _svd_w(self, smooth_weight_chunks: List[torch.Tensor]):
        assert self.has_smoothed, 'SVDOptimizer: linear is not smoothed'
        assert len(smooth_weight_chunks) == self.chunks, \
            f'SVDOptimizer: weight chunks must equal to chunks but get [{len(smooth_weight_chunks)}]'

        svd_weight_chunk = []

        for idx, smooth_weight_chunk in enumerate(smooth_weight_chunks):
            chunk = smooth_weight_chunk.t()
            u, s, vt = torch.linalg.svd(chunk.to(getattr(torch, self.precision)), full_matrices=False)

            if u.shape[1] < self.low_rank or vt.shape[0] < self.low_rank:
                raise ValueError(
                    f"Low-rank dimension {self.low_rank} exceeds layer "
                    f"dimensions {u.shape[1]} and {vt.shape[0]}."
                )

            us = u[:, :self.low_rank] * s[:self.low_rank] # (m, r)
            vt = vt[:self.low_rank, :]                    # (r, n)

            device, dtype = smooth_weight_chunk.device, smooth_weight_chunk.dtype
            l1 = us.to(device=device, dtype=dtype)
            l2 = vt.to(device=device, dtype=dtype)
            self.l1s[idx] = l1
            self.l2s[idx] = l2

            weight_svd = (chunk.to(torch.float64) - us @ vt).t().to(device=device, dtype=dtype)
            svd_weight_chunk.append(weight_svd)
            del u, s, vt, us, chunk
        return svd_weight_chunk

    def smooth(self):
        super().smooth()
        self.weight_chunks = self._svd_w(self.weight_chunks)
        self.has_svd = True

    def calibrate(self, input_chunks):
        assert self.has_svd, 'SVDOptimizer: linear is not svd'
        super().calibrate(input_chunks)

    def forward(self, input_chunks):
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


@OptimizerRegistry.register("ortho")
class OrthoOptimizer:
    def __init__(
        self,
        layer_mode,
        seg_mode,
        chunks,
        chunksizes,
        weight_chunks,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        layer_kwargs={},
        verbose=False,
        givens_m=None,
        cpu_storage=False,
        val_sample_batch=0,
        sub_optimizer_type="default",
        sub_optimizer_kwargs={},
        optimizer_config={},
        stop_criteria={},
        **kwargs,
    ):
        '''
        pipeline:
            determine all scale params(scale, zero_point keep unchanged) for input and weight
            find opt Q


        '''
        assert len(input_calibrators) == len(weight_calibrators) and \
            len(weight_calibrators) == len(weight_chunks), \
            f"Lengths mismatch: input_calibrators={len(input_calibrators)}, " \
            f"weight_calibrators={len(weight_calibrators)}, " \
            f"weight_chunks={len(weight_chunks)}"

        ### sub optimizer
        self.sub_optimizer = OptimizerRegistry.create(
            sub_optimizer_type,
            layer_mode = layer_mode,
            seg_mode = seg_mode,
            chunks = chunks,
            chunksizes = chunksizes,
            weight_chunks = weight_chunks,
            input_calibrators = input_calibrators,
            weight_calibrators = weight_calibrators,
            real_quant = real_quant,
            dual_scale = dual_scale,
            kernel_type = kernel_type,
            layer_kwargs = layer_kwargs,
            **sub_optimizer_kwargs
        )

        self.cpu_storage = cpu_storage
        if self.cpu_storage:
            print('[Warning] OrthoOptimizer set CPU Storage.')

        self.verbose = verbose
        self.has_descended = False

        if givens_m is None:
            # todo: auto infer
            self.givens_m = self.weight_chunks[-1].shape[1]
        else:
            self.givens_m = givens_m

        ## buffer
        self.nsamples = 0
        self.x_rel_buffers = [None] * len(self.input_calibrators) # (xtx, xtex, extex) dim = 0
        self.weight_rel_buffers = [None] * self.chunks # (wwt, wewt, ewewt) dim = 0

        ## val batch
        self.val_sample_batch = val_sample_batch
        self.val_x_buffer = []

        # todo: init strategy
        if len(self.weight_chunks) != self.chunks:
            # seginput, k = chunksizes
            self.k_s = chunksizes.copy()
        else:
            # segweight, k = weight's in
            self.k_s = [chunk.shape[1] for chunk in self.weight_chunks]
        # todo: pairs
        self.pairs = [self._generate_upper_triangular_pairs(k) for k in self.k_s]
        # self.thetas = [
        #     nn.Parameter(
        #         (torch.rand(self.givens_m, dtype=torch.float32, device=self.weight_chunks[i].device) * 2 - 1) * math.pi
        #     )
        #     for i in range(self.chunks)
        # ]
        self.thetas = [
            nn.Parameter(torch.zeros(self.givens_m, dtype=torch.float32, device=self.weight_chunks[i].device)) 
            for i in range(self.chunks)]
        self.Qs = [None] * self.chunks

        ## opt
        self.stop_criteria = stop_criteria
        this_optimizer_config = optimizer_config.copy()
        optimize_type = this_optimizer_config.pop('type')  # 'Adam' or 'SGD'
        if optimize_type.lower() == 'adam':
            OptimClass = torch.optim.Adam
        elif optimize_type.lower() == 'sgd':
            OptimClass = torch.optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer type: {optimize_type}")
        self.optimizers = [
            OptimClass(params=[self.thetas[i]], **this_optimizer_config)
            for i in range(self.chunks)
        ]

    @staticmethod
    def _generate_upper_triangular_pairs(k):
        pairs = []
        for i in range(k-1):
            for j in range(i+1, k):
                pairs.append((i, j))
        return pairs

    @staticmethod
    def _givens_rotation_matrix(k, i, j, theta, device=None):
        G = torch.eye(k, device=device)
        c = torch.cos(theta)
        s = torch.sin(theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        return G
    
    @staticmethod
    def _build_Q(k, thetas, pairs, device):
        Q = torch.eye(k, device=device)
        for theta, (p, q) in zip(thetas, pairs):
            G = OrthoOptimizer._givens_rotation_matrix(k, p, q, theta, device)
            Q = Q @ G
        return Q
    
    @staticmethod
    def _grad_thetas(thetas, pq, grad_Q):
        m = thetas.shape[0]
        k = grad_Q.shape[0]

        device = thetas.device
        dtype = thetas.dtype

        # Prefix products
        P = [torch.eye(k, device=device, dtype=dtype)]
        for j in range(m):
            p, q = pq[j]
            Pj = P[-1].clone()
            u = Pj[p, p]
            v = Pj[p, q]
            r = Pj[q, p]
            t = Pj[q, q]
            c, s = torch.cos(thetas[j]), torch.sin(thetas[j])
            Pj[p, p] = c * u + s * v
            Pj[p, q] = -s * u + c * v
            Pj[q, p] = c * r + s * t
            Pj[q, q] = -s * r + c * t
            P.append(Pj)

        # Suffix products
        S = [torch.eye(k, device=device, dtype=dtype)]  # S_{m+1}
        for j in reversed(range(m)):
            p, q = pq[j]
            Sj = S[0].clone()
            u = Sj[p, p]
            v = Sj[p, q]
            r = Sj[q, p]
            t = Sj[q, q]
            c, s = torch.cos(thetas[j]), torch.sin(thetas[j])
            Sj[p, p] = c * u - s * r
            Sj[p, q] = c * v - s * t
            Sj[q, p] = s * u + c * r
            Sj[q, q] = s * v + c * t
            S.insert(0, Sj)

        # Compute all gradients
        g = torch.zeros(m, device=device, dtype=dtype)
        for j in range(m):
            p, q = pq[j]
            cos_theta, sin_theta = torch.cos(thetas[j]), torch.sin(thetas[j])
            dG = torch.tensor([[-sin_theta, -cos_theta],
                            [ cos_theta, -sin_theta]], device=device, dtype=dtype)
            L = P[j][:,[p, q]]          # shape (k, 2)
            R = S[j+1][[p, q],:]        # shape (2, k)
            A = L @ dG @ R               # shape (k, k)
            g[j] = torch.sum(grad_Q * A)

        return g

    def __getattr__(self, name):
        return getattr(self.sub_optimizer, name)

    def __setattr__(self, name, value):
        if name == "sub_optimizer" or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.sub_optimizer, name, value)

    def after_calibrate(self, input_chunks):
        if hasattr(self.sub_optimizer, '_smooth_x'):
            input_chunks = self._smooth_x(input_chunks)

        if self.val_sample_batch > 0:
            input_chunks_cpu = [chunk.detach().cpu() for chunk in input_chunks]
            self.val_x_buffer.extend(input_chunks_cpu)
            self.val_sample_batch -= 1

        ## init weight_rel_buffers
        if self.weight_rel_buffers[-1] is None:
            if self.seg_mode == 'input':
                assert len(self.weight_chunks) == 1, "weight_chunks size not 1"
                weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
            else:
                weight_chunks = self.weight_chunks

            for i, weight in enumerate(weight_chunks):
                ew = self.weight_calibrators[self.weight_quantized_indices[i]].fake_quantize(weight) - weight
                W = weight.to(dtype=torch.float32, device=weight.device).t()   # (in, out)
                EW = ew.to(dtype=torch.float32, device=weight.device).t()      # (in, out)
                buf_gpu = torch.stack([
                    W @ W.t(),    # wwt
                    W @ EW.t(),   # wewt
                    EW @ EW.t()   # ewewt
                ], dim=0)

                if self.cpu_storage:
                    buf_cpu = buf_gpu.cpu()
                    self.weight_rel_buffers[i] = buf_cpu
                else:
                    self.weight_rel_buffers[i] = buf_gpu

        # compute xtx, xtex, extex
        for i, input_calibrator in enumerate(self.input_calibrators):
            x = input_chunks[i]
            ex = input_calibrator.fake_quantize(x) - x

            if len(x.shape) == 2:
                this_batch = 1
            elif len(x.shape) == 1:
                x = x.unsqueeze(0)
                ex = ex.unsqueeze(0)
                this_batch = 1
            else:
                x = x.reshape((-1, x.shape[-1]))
                ex = ex.reshape((-1, ex.shape[-1]))
                this_batch = x.shape[0]

            x = x.to(dtype=torch.float32, device=x.device).t()  # (in, b)
            ex = ex.to(dtype=torch.float32, device=x.device).t()  # (in, b)
            device = x.device

            if self.x_rel_buffers[i] is None:
                self.x_rel_buffers[i] = torch.zeros(
                    (3, x.shape[0], x.shape[0]),
                    device='cpu' if self.cpu_storage else device,
                    dtype=torch.float32,
                    pin_memory=True if self.cpu_storage else False
                )

            if self.cpu_storage:
                buffer = self.x_rel_buffers[i].to(device, non_blocking=True)
            else:
                buffer = self.x_rel_buffers[i]

            buffer.mul_(self.nsamples / (self.nsamples + this_batch))
            self.nsamples += this_batch

            scale = math.sqrt(2 / self.nsamples)
            x.mul_(scale)
            ex.mul_(scale)

            buffer[0].addmm_(x, x.t(), beta=1.0, alpha=1.0)
            buffer[1].addmm_(x, ex.t(), beta=1.0, alpha=1.0)
            buffer[2].addmm_(ex, ex.t(), beta=1.0, alpha=1.0)

            if self.cpu_storage:
                self.x_rel_buffers[i].copy_(buffer, non_blocking=True)

    def _grad_Q(self, Q, i, device):
        xtx, xtex, extex = self.x_rel_buffers[self.input_quantized_indices[i]]
        wwt, wewt, ewewt = self.weight_rel_buffers[i]
        if xtx.device != device:
            xtx, xtex, extex = xtx.to(device), xtex.to(device), extex.to(device)

        if wwt.device != device:    
            wwt, wewt, ewewt = wwt.to(device), wewt.to(device), ewewt.to(device)

        Q_T = Q.T
        grad = (xtx @ Q @ ewewt
                + wwt @ Q @ extex
                + xtex @ Q_T @ wewt
                + wewt @ Q_T @ xtex
                + xtex @ ewewt
                + wewt @ extex)
        return grad
    
    def grad_func(self, i, device):
        thetas = self.thetas[i]
        k = self.k_s[i]
        pairs = self.pairs[i]
        Q = self._build_Q(k, thetas, pairs, device=device)
        grad_Q = self._grad_Q(Q, i, device=device)
        return self._grad_thetas(thetas, pairs, grad_Q)

    def loss_thetas(self, W, i, device):
        Q = self._build_Q(self.k_s[i], self.thetas[i], self.pairs[i], device=device)
        quant_W = self.weight_calibrators[self.weight_quantized_indices[i]].fake_quantize(W @ Q)  # (out, in)
        X_quantizer = self.input_calibrators[self.input_quantized_indices[i]]
        loss_total = 0.0
        for X in self.val_x_buffer:
            X = X.to(device=device, dtype=torch.float32)
            quant_X = X_quantizer.fake_quantize(X @ Q)  # (b, in)
            loss = torch.norm(X @ W.t() - quant_X @ quant_W.t(), p='fro') ** 2
            loss_total += loss.item()
        
        if len(self.val_x_buffer) == 0:
            return 0.0
        return loss_total / len(self.val_x_buffer)

    def step(self):
        criteria = self.stop_criteria
        max_steps = criteria.get('max_steps', 100)
        grad_tol = criteria.get('grad_tol', 1e-4)
        grad_change_tol = criteria.get('grad_change_tol', 1e-5)
        patience = criteria.get('patience', 5)
        ema_decay = criteria.get('ema_grad_decay', 0.9)
        check_every = criteria.get('check_every', 1)
        update_w_every = criteria.get('update_w_every', 1)

        grad_ema = [None] * self.chunks
        prev_grad = [None] * self.chunks
        stop_counter = [0] * self.chunks
        active = [True] * self.chunks

        if self.seg_mode == 'input':
            assert len(self.weight_chunks) == 1, "weight_chunks size not 1"
            weight_chunks = self.weight_chunks[0].split(self.chunksizes, dim=-1)
        else:
            weight_chunks = self.weight_chunks
        
        print("init thetas", self.thetas)

        for step in range(max_steps):
            if not any(active):
                if self.verbose:
                    print(f"All chunks converged by step {step}")
                break

            for i in range(self.chunks):
                if not active[i]:
                    continue
                
                W = weight_chunks[i].to(dtype=torch.float32)
                device = W.device
                loss = self.loss_thetas(W, i=i, device=device)
                self.thetas[i].grad = self.grad_func(i=i, device=device)
                grad = self.thetas[i].grad.detach()
                self.optimizers[i].step()
                self.optimizers[i].zero_grad()
                Q_new = self._build_Q(self.k_s[i], self.thetas[i], self.pairs[i], device=device)

                if grad_ema[i] is None:
                    grad_ema[i] = grad.clone()
                else:
                    grad_ema[i] = ema_decay * grad_ema[i] + (1 - ema_decay) * grad

                grad_norm = torch.norm(grad_ema[i])
                grad_diff = torch.norm(grad - prev_grad[i]) if prev_grad[i] is not None else 0.0
                prev_grad[i] = grad.clone()

                if grad_norm < grad_tol or grad_diff < grad_change_tol:
                    stop_counter[i] += 1
                else:
                    stop_counter[i] = 0

                if stop_counter[i] >= patience:
                    active[i] = False
                    if self.verbose:
                        print(f"Chunk {i} converged at step {step}")

                if self.verbose and step % check_every == 0:
                    print(f"[Chunk {i}] Step {step}: loss={loss:.6f}, ortho={torch.norm(Q_new @ Q_new.t() - torch.eye(Q_new.shape[0], device=Q_new.device)).item():.4e}, grad_norm={grad_norm:.6f}, grad_diff={grad_diff:.6f}, stop_count={stop_counter[i]}")

                ### update w rel
                if step % update_w_every == 0:
                    weight = W @ Q_new  # (out, in)
                    ew = self.weight_calibrators[self.weight_quantized_indices[i]].fake_quantize(weight) - weight
                    W = weight.t()   # (in, out)
                    EW = ew.t()      # (in, out)
                    buf_gpu = torch.stack([
                        W @ W.t(),    # wwt
                        W @ EW.t(),   # wewt
                        EW @ EW.t()   # ewewt
                    ], dim=0)
                    if self.cpu_storage:
                        buf_cpu = buf_gpu.cpu()
                        self.weight_rel_buffers[i] = buf_cpu
                    else:
                        self.weight_rel_buffers[i] = buf_gpu

        print(self.thetas)

        for i in range(self.chunks):
            device = weight_chunks[i].device
            self.Qs[i] = self._build_Q(self.k_s[i], self.thetas[i], self.pairs[i], device=device)
            weight_chunks[i] = (weight_chunks[i].float() @ self.Qs[i]).to(dtype=weight_chunks[i].dtype, device=weight_chunks[i].device)
        if self.seg_mode == 'input':
            self.weight_chunks = [torch.cat(weight_chunks, dim=-1)]
        else:
            self.weight_chunks = weight_chunks

        self.val_x_buffer = None
        self.x_rel_buffers = None
        self.weight_rel_buffers = None
        self.optimizers = None
        self.has_descended = True

    def _ortho_x(self, input_chunks: List[torch.Tensor]):
        input_broadcast = self._broadcast_list(input_chunks)
        ortho_input_chunks = []
        for i in range(self.chunks):
            # todo kernel implementation
            chunk = input_broadcast[i].clone()
            thetas = self.thetas[i]
            pairs = self.pairs[i]
            for theta, (p, q) in zip(thetas, pairs):
                c, s = torch.cos(theta), torch.sin(theta)
                tmp_p = chunk[:, p].clone()
                tmp_q = chunk[:, q].clone()
                chunk[:, p] = c*tmp_p - s*tmp_q
                chunk[:, q] = s*tmp_p + c*tmp_q

            ortho_input_chunks.append(chunk.to(dtype=chunk.dtype, device=chunk.device))

        return ortho_input_chunks

    def to(self, device):
        self.sub_optimizer.to(device)
        self.Qs = [Q.to(device) for Q in self.Qs if Q is not None]

    def forward(self, input_chunks):
        if not self.has_descended:
            raise RuntimeError("OrthoOptimizer: must cayley_descent first")

        if hasattr(self.sub_optimizer, '_smooth_x'):
            input_chunks = self._smooth_x(input_chunks)
        input_chunks = self._ortho_x(input_chunks)
        return self.sub_optimizer.forward(input_chunks)
