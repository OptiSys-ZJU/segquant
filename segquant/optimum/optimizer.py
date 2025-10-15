from collections import deque

import numpy as np
from segquant.layers.segment_tensor_manager import InputSegmentTensorManager, WeightSegmentTensorManager, segmented_matmul
import torch
from segquant.layers import ext_dict


class BaseOptimizer:
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_manager,
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
        self.weight_manager = weight_manager
        self.input_calibrators = input_calibrators
        self.weight_calibrators = weight_calibrators
        self.real_quant = real_quant
        self.kernel_type = kernel_type

        if self.real_quant and 'Wint4' in self.kernel_type:
            self.packed = True
        else:
            self.packed = False

        self.dual_scale = dual_scale
        if self.real_quant and dual_scale:
            self.func_name = 'gemm_dual_scaled_fn'
        else:
            self.func_name = 'gemm_scaled_fn'

        self.has_calibrated = False

        self.scale_tensor = None

    def make_scale_tensor(self):
        weight_scales = [
            (
                calibrator.scale.to(dtype=torch.float32, device=self.weight_manager.device())
                if isinstance(calibrator.scale, torch.Tensor)
                else torch.tensor(calibrator.scale, dtype=torch.float32, device=self.weight_manager.device())
            )
            for calibrator in self.weight_calibrators
        ]
        if len(weight_scales) == 1:
            weight_scales = weight_scales * self.chunks
        if self.dual_scale:
            pos_input_scales = [
                (
                    calibrator.pos_scale.to(dtype=torch.float32, device=self.weight_manager.device())
                    if isinstance(calibrator.pos_scale, torch.Tensor)
                    else torch.tensor(calibrator.pos_scale, dtype=torch.float32, device=self.weight_manager.device())
                )
                for calibrator in self.input_calibrators
            ]
            if len(pos_input_scales) == 1:
                pos_input_scales = pos_input_scales * self.chunks
            neg_input_scales = [
                (
                    calibrator.neg_scale.to(dtype=torch.float32, device=self.weight_manager.device())
                    if isinstance(calibrator.neg_scale, torch.Tensor)
                    else torch.tensor(calibrator.neg_scale, dtype=torch.float32, device=self.weight_manager.device())
                )
                for calibrator in self.input_calibrators
            ]
            if len(neg_input_scales) == 1:
                neg_input_scales = neg_input_scales * self.chunks

            self.scale_tensor = (torch.stack(pos_input_scales), torch.stack(neg_input_scales), torch.stack(weight_scales))
        else:
            input_scales = [
                (
                    calibrator.scale.to(dtype=torch.float32, device=self.weight_manager.device())
                    if isinstance(calibrator.scale, torch.Tensor)
                    else torch.tensor(calibrator.scale, dtype=torch.float32, device=self.weight_manager.device())
                )
                for calibrator in self.input_calibrators
            ]
            if len(input_scales) == 1:
                input_scales = input_scales * self.chunks
            self.scale_tensor = (torch.stack(input_scales), torch.stack(weight_scales))

    def call_func(self, batch_input: torch.Tensor, batch_weight: torch.Tensor):
        if self.real_quant:
            if self.input_calibrators[0].dynamic:
                # slow path
                for input_calibrator, input_chunk in zip(self.input_calibrators, batch_input):
                    input_calibrator.dynamic_calibrate(input_chunk)
                self.make_scale_tensor()

            if self.dual_scale:
                pos_scale_x, neg_scale_x, scale_w = self.scale_tensor
                res = ext_dict[self.kernel_type][self.func_name](
                    batch_input.contiguous(),
                    batch_weight.contiguous(),
                    pos_scale_x.contiguous(),
                    neg_scale_x.contiguous(),
                    scale_w.contiguous(),
                )
            else:
                scale_x, scale_w = self.scale_tensor
                res = ext_dict[self.kernel_type][self.func_name](
                    batch_input.contiguous(),
                    batch_weight.contiguous(),
                    scale_x.contiguous(),
                    scale_w.contiguous(),
                )
        else:
            for i, c in enumerate(self.input_calibrators):
                batch_input[i].copy_(c.quantize(batch_input[i]))        
            res = segmented_matmul(batch_input, batch_weight)
        return res

    def _calibrate_weights(self, i_view, w_view):
        for weight_calibrator, weight_chunk, input_copy_or_chunk in zip(
            self.weight_calibrators, w_view, i_view
        ):
            weight_calibrator.calibrate(weight_chunk, input_data=input_copy_or_chunk)

    def process_step(self, err):
        pass

    def to_cpu(self):
        self.weight_manager = self.weight_manager.to('cpu')
        self.input_calibrators = [calibrator.to('cpu') for calibrator in self.input_calibrators]
        self.weight_calibrators = [calibrator.to('cpu') for calibrator in self.weight_calibrators]

    def to_cuda(self, device):
        self.weight_manager = self.weight_manager.to(device)
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
        seg_mode,
        chunks,
        chunksizes,
        weight_manager,
        input_calibrators,
        weight_calibrators,
        real_quant=False,
        dual_scale=False,
        kernel_type=None,
        **kwargs,
    ):
        assert len(input_calibrators) == 1 or len(weight_calibrators) == 1, \
            "Either input_calibrators or weight_calibrators must have a length of 1."
        super().__init__(
            seg_mode,
            chunks,
            chunksizes,
            weight_manager,
            input_calibrators,
            weight_calibrators,
            real_quant,
            dual_scale,
            kernel_type,
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

    def calibrate(self, input_manager: InputSegmentTensorManager):
        # input-seg, input calib = segment, chunk (segments, ..., segment_size)
        # weight-seg, input calib = 1, chunk (segments (repeated), ..., in)
        for input_calibrator, input_chunk in zip(self.input_calibrators, input_manager.iter_view()):
            input_calibrator.calibrate(input_chunk)

        if self.seg_mode == 'input':
            # input-seg, input calib = segment, weight calib = 1
            # input chunk (1, ..., in), weight chunk (1, out, in)
            i_view = input_manager.total_view()
            w_view = self.weight_manager.total_view()
        elif self.seg_mode == 'weight':
            # weight-seg, input calib = 1, weight calib = segment
            # input chunk (segments (repeated), ..., in), weight chunk (segments, segment_size, in)
            i_view = input_manager.iter_view()
            w_view = self.weight_manager.iter_view()

        self._calibrate_weights(i_view, w_view)

    def finish_calibrate(self):
        for i, input_calibrator in enumerate(self.input_calibrators):
            input_calibrator.finish_calibrate()
            self.input_calibrators[i] = input_calibrator.quantizer

        if self.seg_mode == 'input':
            # input-seg, weight calib = 1, view # (1, out, in)
            assert len(self.weight_calibrators) == 1, "weight_calibrators length must be 1"
            self.weight_manager.weight_tensor = self.weight_calibrators[0].finish_calibrate(self.weight_manager.total_view()[0])
            self.weight_calibrators[0] = self.weight_calibrators[0].quantizer
        elif self.seg_mode == 'weight':
            calibrated_segments = []
            weight_view = self.weight_manager.iter_view() # (segments, segment_size, in)
            for i, weight_calibrator in enumerate(self.weight_calibrators):
                # weight-seg, weight calib = segments
                calibrated = weight_calibrator.finish_calibrate(weight_view[i])
                calibrated_segments.append(calibrated)
                self.weight_calibrators[i] = weight_calibrator.quantizer

            # (segments, segment_size, in)
            new_quantized_weight_tensor = torch.stack(calibrated_segments, dim=0)
            # update weight
            self.weight_manager.replace_with_segments_layout(new_quantized_weight_tensor, packed=self.packed)

        self.has_calibrated = True
        self.make_scale_tensor()

    def forward(self, input_manager: InputSegmentTensorManager):
        # input-seg input chunk (segments, ..., segment_size) input calib = segments
        # weight-seg input chunk (segments (repeated), ..., in) input calib = 1
        input_chunks = input_manager.iter_view()

        # input-seg weight chunk (segments, out, segment_size) --> (segments, ..., out)
        # weight-seg weight chunk (segments, segment_size, in) --> (segments, ..., segment_size)
        return self.call_func(input_chunks, self.weight_manager.iter_view())

    def process_step(self, err):
        pass

@OptimizerRegistry.register("smooth")
class SmoothOptimizer(BaseOptimizer):
    def __init__(
        self,
        seg_mode,
        chunks,
        chunksizes,
        weight_manager,
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
        super().__init__(
            seg_mode,
            chunks,
            chunksizes,
            weight_manager,
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
        self.weight_chunk_indices = range(self.chunks)

        if seg_mode == 'input':
            self.input_chunk_indices = range(self.chunks)
            self.weight_quantized_indices = [0] * self.chunks
        elif seg_mode == 'weight':
            self.input_chunk_indices = [0] * self.chunks
            self.weight_quantized_indices = range(self.chunks)
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

    def _trace_max_w(self, weight_chunks: torch.Tensor):
        if len(self.max_w) < self.chunks:
            for weight_chunk in weight_chunks:
                weight_chunk_max = (
                    weight_chunk.abs()
                    .amax(dim=tuple(range(weight_chunk.ndim - 1)), keepdim=True)
                    .squeeze()
                )
                self.max_w.append(weight_chunk_max)

    def _trace_max_x(self, input_chunks: torch.Tensor):
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

    def trace(self, input_data: InputSegmentTensorManager):
        """
        Trace the maximum values of input tensors for calibration.
        This method computes the maximum values of the input tensors and weight tensors
        and stores them for later use in the smoothing and quantization process.
        It assumes that the input tensors are split into chunks and that the weight tensors
        are also split into chunks.
        Args:
            input_chunks: [tensor(batch_size * split_size1), tensor(batch_size * split_size2), ...]
        """
        self._trace_max_x(input_data.iter_view())

        if not self.has_traced_w:
            self._trace_max_w(self.weight_manager.iter_view())
            self.has_traced_w = True

    def to_cpu(self):
        super().to_cpu()
        self.s = [s.to('cpu') for s in self.s]

    def to_cuda(self, device):
        super().to_cuda(device)
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
        assert len(self.max_x) == self.chunks, 'max_x failed'
        assert len(self.max_w) == self.chunks, 'max_w failed'
        for i in range(self.chunks):
            epsilon = 1.0 / (1 << 31)
            s = (self.max_x[i] ** self.alpha[i]) / (self.max_w[i] ** (1 - self.alpha[i]))
            s = torch.where(s <= epsilon, torch.ones_like(s), s)
            self.s[i] = torch.clamp(s.to(dtype=torch.float32), min=1e-4, max=1e4)

        # smooth weight
        self._smooth_w(self.weight_manager.iter_view())
        self.has_smoothed = True

    def _smooth_w(self, weight_chunks: torch.Tensor):
        for i, (weight_chunk, s) in enumerate(zip(weight_chunks, self.s)):
            weight_chunk.mul_(s.to(weight_chunk.device))

    def _smooth_x(self, input_chunks: torch.Tensor):
        for input_chunk, s in zip(input_chunks, self.s):
            input_chunk.div_(s.to(input_chunk.device, dtype=input_chunk.dtype))
        return input_chunks

    def calibrate(self, input_manager: InputSegmentTensorManager):
        assert self.has_smoothed, "SmoothOptimizer: linear is not smoothed"
        input_chunks = self._smooth_x(input_manager.iter_view().clone())
        for input_calibrator, input_chunk in zip(self.input_calibrators, input_chunks):
            input_calibrator.calibrate(input_chunk)

        # weight
        self._calibrate_weights(input_chunks, self.weight_manager.iter_view())

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

        calibrated_segments = []
        weight_view = self.weight_manager.iter_view()
        for i, weight_calibrator in enumerate(self.weight_calibrators):
            calibrated = weight_calibrator.finish_calibrate(weight_view[i])
            calibrated_segments.append(calibrated)

        # (segments, out, segment_size) or (segments, segment_size, in)
        new_quantized_weight_tensor = torch.stack(calibrated_segments, dim=0)
        # update weight
        self.weight_manager.replace_with_segments_layout(new_quantized_weight_tensor, packed=self.packed)

        self.has_calibrated = True
        self.make_scale_tensor()

        if not self.search_alpha:
            self._clean()

    def forward(self, input_manager: InputSegmentTensorManager):
        # input-seg input chunk (segments, ..., segment_size)
        # weight-seg input chunk (segments, ..., in)
        input_chunks = self._smooth_x(input_manager.iter_view().clone())        

        # input-seg weight chunk (segments, out, segment_size) --> (segments, ..., out)
        # weight-seg weight chunk (segments, segment_size, in) --> (segments, ..., segment_size)
        return self.call_func(input_chunks, self.weight_manager.iter_view())

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
        self.weight_manager = WeightSegmentTensorManager(
            weight_tensor=origin_weight.clone(),
            seg_mode=self.seg_mode,
            segments=self.chunks,
            segment_size=self.chunksizes[0],
        )

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
        weight_manager,
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
            weight_manager,
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

    def to_cpu(self):
        super().to_cpu()
        if isinstance(self.l1s, list):
            self.l1s = [l1.to('cpu') for l1 in self.l1s]
            self.l2s = [l2.to('cpu') for l2 in self.l2s]
        elif isinstance(self.l1s, torch.Tensor):
            self.l1s = self.l1s.to('cpu')
            self.l2s = self.l2s.to('cpu')

    def to_cuda(self, device):
        super().to_cuda(device)
        if isinstance(self.l1s, list):
            self.l1s = [l1.to(device) for l1 in self.l1s]
            self.l2s = [l2.to(device) for l2 in self.l2s]
        elif isinstance(self.l1s, torch.Tensor):
            self.l1s = self.l1s.to(device)
            self.l2s = self.l2s.to(device)

    def _svd_w(self, smooth_weight_chunks: torch.Tensor):
        assert self.has_smoothed, 'SVDOptimizer: linear is not smoothed'
        assert len(smooth_weight_chunks) == self.chunks, \
            f'SVDOptimizer: weight chunks must equal to chunks but get [{len(smooth_weight_chunks)}]'

        for idx, smooth_weight_chunk in enumerate(smooth_weight_chunks):
            # (out, in) -> (in, out)
            chunk_t = smooth_weight_chunk.t()
            chunk_t_f = chunk_t.to(getattr(torch, self.precision))

            # SVD
            u, s, vt = torch.linalg.svd(chunk_t_f, full_matrices=False)
            if u.shape[1] < self.low_rank or vt.shape[0] < self.low_rank:
                raise ValueError(
                    f"Low-rank dimension {self.low_rank} exceeds layer "
                    f"dimensions {u.shape[1]} and {vt.shape[0]}."
                )
            us = u[:, :self.low_rank] * s[:self.low_rank] # (in, low_rank)
            vt = vt[:self.low_rank, :] # (low_rank, out)
            device, dtype = smooth_weight_chunk.device, smooth_weight_chunk.dtype
            self.l1s[idx] = us.to(device=device, dtype=dtype)
            self.l2s[idx] = vt.to(device=device, dtype=dtype)
            residual_t = (chunk_t_f.to(torch.float64) - us @ vt).to(dtype=dtype, device=device)
            smooth_weight_chunk.copy_(residual_t.t())
            del u, s, vt, us, chunk_t, chunk_t_f, residual_t

        if isinstance(self.l1s, list):
            self.l1s = torch.stack(self.l1s) # (segments, in, low_rank)
            self.l2s = torch.stack(self.l2s)  # (segments, low_rank, out)        

    def smooth(self):
        super().smooth()
        self._svd_w(self.weight_manager.iter_view())
        self.has_svd = True

    def calibrate(self, input_manager: InputSegmentTensorManager):
        assert self.has_svd, 'SVDOptimizer: linear is not svd'
        super().calibrate(input_manager)

    def _low_rank_mul(self, x, l1s, l2s):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (segments, 1, M)
        hidden = torch.bmm(x, l1s)
        out = torch.bmm(hidden, l2s)
        return out

    def forward(self, input_manager: InputSegmentTensorManager):
        # input-seg input chunk (segments, ..., segment_size)
        # weight-seg input chunk (segments, ..., in)
        input_chunks = self._smooth_x(input_manager.iter_view().clone())

        return self.call_func(
            input_chunks, self.weight_manager.iter_view()
        ) + self._low_rank_mul(input_chunks, self.l1s, self.l2s)
