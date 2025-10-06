from functools import partial
import math
from typing import List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segquant.layers.givens_orthogonal import GivensOrthogonal
from segquant.utils.anomaly_channel_detector import AnomalyChannelDetector


def identity(x):
    return x


def identity_tuple(x, chunksizes):
    return (x,)


def repeat_tuple(x, chunksizes):
    return tuple(x for _ in chunksizes)


def tuple_wrap(w):
    return (w,)


def chunks_identity(chunks):
    return chunks


def chunks_one(chunks):
    return 1

class OptimizerRegistry:
    _registry = {}

    _segment_config = {
        "default": {
            "input": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_one,
                "split_input_func": partial(torch.Tensor.split, dim=-1),
                "split_weight_func": tuple_wrap,
            },
            "weight": {
                "input_quantizers_len": chunks_one,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": identity_tuple,
                "split_weight_func": partial(torch.Tensor.split, dim=0),
            },
        },
        "smooth": {
            "input": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": partial(torch.Tensor.split, dim=-1),
                "split_weight_func": partial(torch.Tensor.split, dim=-1),
            },
            "weight": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": repeat_tuple,
                "split_weight_func": partial(torch.Tensor.split, dim=0),
            },
        },
        "svd": {
            "input": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": partial(torch.Tensor.split, dim=-1),
                "split_weight_func": partial(torch.Tensor.split, dim=-1),
            },
            "weight": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": repeat_tuple,
                "split_weight_func": partial(torch.Tensor.split, dim=0),
            },
        },
        "givens": {
            "input": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": partial(torch.Tensor.split, dim=-1),
                "split_weight_func": partial(torch.Tensor.split, dim=-1),
            },
            "weight": {
                "input_quantizers_len": chunks_identity,
                "weight_quantizers_len": chunks_identity,
                "split_input_func": repeat_tuple,
                "split_weight_func": partial(torch.Tensor.split, dim=0),
            },
        },
    }

    @classmethod
    def get_segment_config(cls, opt_type, seg_mode, chunks):
        if opt_type not in cls._segment_config:
            raise ValueError(f"Optimizer type '{opt_type}' not found in registry.")
        if seg_mode not in cls._segment_config[opt_type]:
            raise ValueError(f"seg_mode '{seg_mode}' not found in optimizer config.")

        config = cls._segment_config[opt_type][seg_mode]
        input_quantizers_len = config["input_quantizers_len"](chunks)
        weight_quantizers_len = config["weight_quantizers_len"](chunks)
        split_input_func = config["split_input_func"]
        split_weight_func = config["split_weight_func"]

        return (
            input_quantizers_len,
            weight_quantizers_len,
            split_input_func,
            split_weight_func,
        )

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

class BaseOptimizer:
    def __init__(self, upper_module, **kwargs):
        self.upper_module = upper_module

    def preprocess_x(self, x_chunks):
        return x_chunks

    def on_calibrate_prepare_finish(self, Ws, eWs):
        return Ws

    def on_calibrate_finish_end(self):
        return True

    def chunk_forward(self, quantized_input_chunks, quantized_weight_chunks):
        quantized_output_chunks = []
        for quantized_input_chunk, quantized_weight_chunk in zip(
            quantized_input_chunks, quantized_weight_chunks
        ):
            quantized_output_chunks.append(
                F.linear(quantized_input_chunk, quantized_weight_chunk)
            )
        return quantized_output_chunks


@OptimizerRegistry.register("default")
class DefaultOptimizer(BaseOptimizer):
    def __init__(self, upper_module, **kwargs):
        super().__init__(upper_module, **kwargs)
    
    def __repr__(self):
        return "DefaultOptimizer()"


@OptimizerRegistry.register("smooth")
class SmoothOptimizer(BaseOptimizer):
    def __init__(
        self,
        upper_module: nn.Module,
        in_features: int,
        out_features: int,
        seg_mode: str,
        chunks: int,
        chunksizes: List[int],
        device: torch.device,
        ### extra args
        verbose=False,
        alpha=0.5,
        search_alpha_config=None,
        **kwargs,
    ):
        super().__init__(upper_module, **kwargs)

        self.upper_module = upper_module
        self.chunks = chunks
        self.alpha = [alpha] * self.chunks
        if seg_mode == "input":
            for i, sz in enumerate(chunksizes):
                upper_module.register_buffer(
                    f"smooth_max_x_{i}",
                    torch.full(
                        (sz,), float("-inf"), dtype=torch.float32, device=device
                    ),
                    persistent=False,
                )  # (in,)
                upper_module.register_buffer(
                    f"smooth_max_w_{i}",
                    torch.full(
                        (sz,), float("-inf"), dtype=torch.float32, device=device
                    ),
                    persistent=False,
                )  # (in,)
                upper_module.register_buffer(
                    f"smooth_s_{i}",
                    torch.ones((sz,), dtype=torch.float32, device=device),
                )  # (in,)
        elif seg_mode == "weight":
            for i in range(chunks):
                upper_module.register_buffer(
                    f"smooth_max_x_{i}",
                    torch.full(
                        (in_features,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=device,
                    ),
                    persistent=False,
                )  # (in,)
                upper_module.register_buffer(
                    f"smooth_max_w_{i}",
                    torch.full(
                        (in_features,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=device,
                    ),
                    persistent=False,
                )  # (in,)
                upper_module.register_buffer(
                    f"smooth_s_{i}",
                    torch.ones((in_features,), dtype=torch.float32, device=device),
                )  # (in,)

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
        main_info = (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"search_alpha={self.search_alpha}"
        )
        if self.search_alpha:
            main_info += f", candidate_alphas_remaining={[len(a) for a in self.candidate_alphas]}"
        main_info += ")"
        return main_info

    def _trace_max_w(self, weight_chunks: List[torch.Tensor]):
        for i, weight_chunk in enumerate(weight_chunks):
            weight_chunk_max = (
                weight_chunk.abs()
                .amax(dim=tuple(range(weight_chunk.ndim - 1)))
            )
            max_w = getattr(self.upper_module, f"smooth_max_w_{i}")
            max_w[:] = torch.maximum(max_w, weight_chunk_max)

    def _trace_max_x(self, input_chunks: List[torch.Tensor]):
        for i, input_chunk in enumerate(input_chunks):
            input_chunk_max = (
                input_chunk.abs()
                .amax(dim=tuple(range(input_chunk.ndim - 1)))
            )
            max_x = getattr(self.upper_module, f"smooth_max_x_{i}")
            max_x[:] = torch.maximum(max_x, input_chunk_max)

    def trace(self, x_chunks, w_chunks):
        if w_chunks is not None:
            self._trace_max_w(w_chunks)
        self._trace_max_x(x_chunks)

    def on_trace_finish(self, weight_chunks: nn.ParameterList) -> List[torch.Tensor]:
        smoothed_weight_chunks = []
        for i, weight_chunk in enumerate(weight_chunks):
            max_x = getattr(self.upper_module, f"smooth_max_x_{i}")
            max_w = getattr(self.upper_module, f"smooth_max_w_{i}")
            scale = getattr(self.upper_module, f"smooth_s_{i}")
            epsilon = 1.0 / (1 << 31)
            s = (max_w.pow(1 - self.alpha[i])) / (max_x.pow(self.alpha[i]))
            s = torch.where(s <= epsilon, torch.ones_like(s), s)
            scale[:] = torch.clamp(s.to(dtype=torch.float32), min=1e-4, max=1e4)

            inv_scale = 1.0 / scale
            smoothed_weight_chunks.append(
                weight_chunk * inv_scale.to(weight_chunk.device)
            )

        return smoothed_weight_chunks

    def preprocess_x(self, x_chunks):
        smoothed_x_chunks = []
        for i, x_chunk in enumerate(x_chunks):
            s = getattr(self.upper_module, f"smooth_s_{i}")
            smoothed_x_chunks.append(x_chunk * s.to(x_chunk.device))
        return smoothed_x_chunks

    def _clean(self):
        for i in range(self.chunks):
            del self.upper_module._buffers[f'smooth_max_x_{i}']
            del self.upper_module._buffers[f'smooth_max_w_{i}']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_calibrate_finish_end(self):
        if not self.search_alpha:
            self._clean()
            return True
        return False

    def search_step(self, errs):
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

@OptimizerRegistry.register("svd")
class SVDOptimizer(SmoothOptimizer):
    def __init__(
        self,
        upper_module,
        in_features,
        out_features,
        seg_mode,
        chunks,
        chunksizes,
        device,
        ## extra args
        verbose=False,
        alpha=0.5,
        search_alpha_config=None,
        low_rank=32,
        precision='float64',
        **kwargs,
    ):
        super().__init__(
            upper_module,
            in_features,
            out_features,
            seg_mode,
            chunks,
            chunksizes,
            device,
            verbose,
            alpha,
            search_alpha_config,
            **kwargs,
        )

        self.low_rank = low_rank
        if seg_mode == "input":
            for i, sz in enumerate(chunksizes):
                upper_module.register_buffer(
                    f"svd_l1_{i}",
                    torch.empty(
                        (sz, self.low_rank), dtype=torch.float32, device=device
                    ),
                )  # (in, low_rank)
                upper_module.register_buffer(
                    f"svd_l2_{i}",
                    torch.empty(
                        (self.low_rank, out_features),
                        dtype=torch.float32,
                        device=device,
                    ),
                )  # (low_rank, out)
        elif seg_mode == "weight":
            for i, sz in enumerate(chunksizes):
                upper_module.register_buffer(
                    f"svd_l1_{i}",
                    torch.empty(
                        (in_features, self.low_rank), dtype=torch.float32, device=device
                    ),
                )  # (in, low_rank)
                upper_module.register_buffer(
                    f"svd_l2_{i}",
                    torch.empty(
                        (self.low_rank, sz),
                        dtype=torch.float32,
                        device=device,
                    ),
                )  # (low_rank, out)

        self.precision = precision
        if self.precision not in ["float32", "float64"]:
            raise ValueError(
                f"Unsupported precision: {self.precision}. Use 'float32' or 'float64'."
            )

    def __repr__(self):
        main_info = (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"low_rank={self.low_rank}, "
            f"precision={self.precision}, "
            f"search_alpha={self.search_alpha}"
        )
        if self.search_alpha:
            main_info += f", candidate_alphas_remaining={[len(a) for a in self.candidate_alphas]}"
        main_info += ")"
        return main_info

    @torch.no_grad()
    def on_trace_finish(self, weight_chunks: nn.ParameterList):
        smoothed_weight_chunks = super().on_trace_finish(weight_chunks=weight_chunks)

        ## smooth ok, do svd
        svd_weight_chunks = []
        for i, smoothed_weight_chunk in enumerate(smoothed_weight_chunks):
            weight_chunk = smoothed_weight_chunk.t()  # (in, out)
            u, s, vt = torch.linalg.svd(
                weight_chunk.to(getattr(torch, self.precision)), full_matrices=False
            )
            if u.shape[1] < self.low_rank or vt.shape[0] < self.low_rank:
                raise ValueError(
                    f"Low-rank dimension {self.low_rank} exceeds layer "
                    f"dimensions {u.shape[1]} and {vt.shape[0]}."
                )
            us = u[:, :self.low_rank] * s[:self.low_rank] # (m, r)
            vt = vt[:self.low_rank, :]                    # (r, n)

            device, dtype = smoothed_weight_chunk.device, smoothed_weight_chunk.dtype

            l1 = getattr(self.upper_module, f'svd_l1_{i}')
            l2 = getattr(self.upper_module, f'svd_l2_{i}')
            l1[:] = us.to(device=device, dtype=dtype)
            l2[:] = vt.to(device=device, dtype=dtype)

            weight_svd = (
                (weight_chunk.to(torch.float64) - us @ vt)
                .t()
                .to(device=device, dtype=dtype)
            )
            svd_weight_chunks.append(weight_svd)

        return svd_weight_chunks

    def chunk_forward(self, quantized_input_chunks, quantized_weight_chunks):
        quantized_output_chunks = []
        for i, (quantized_input_chunk, quantized_weight_chunk) in enumerate(
            zip(quantized_input_chunks, quantized_weight_chunks)
        ):
            quantized_output_chunks.append(
                quantized_input_chunk
                @ getattr(self.upper_module, f"svd_l1_{i}")
                @ getattr(self.upper_module, f"svd_l2_{i}")
                + F.linear(quantized_input_chunk, quantized_weight_chunk)
            )
        return quantized_output_chunks


@OptimizerRegistry.register("givens")
class GivensOptimizer(BaseOptimizer):
    def __init__(
        self,
        upper_module: nn.Module,
        in_features: int,
        out_features: int,
        seg_mode: str,
        chunks: int,
        chunksizes: List[int],
        device: torch.device,
        ### sub opt
        sub_optimizer_type="default",
        sub_optimizer_kwargs={},
        ### extra args
        verbose=False,
        optim_type="SGD",
        optim_config={},
        givens_num=None,
        sample_mode="rand",
        init_vecs_mode="identity",
        dtype=torch.float32,
        enable_autograd=False,
        enable_grad_buffer=False,
        enable_low_memory_grad=False,
        cpu_storage=False,
        stop_criteria={},
        mini_batch_size=16,
        **kwargs,
    ):
        super().__init__(upper_module, **kwargs)
        self.sub_optimizer = OptimizerRegistry.create(
            sub_optimizer_type,
            upper_module=upper_module,
            in_features=in_features,
            out_features=out_features,
            seg_mode=seg_mode,
            chunks=chunks,
            chunksizes=chunksizes,
            device=device,
            **sub_optimizer_kwargs
        )
        self.sub_optimizer_type = sub_optimizer_type

        if sample_mode not in ['rand', 'ascending', 'descending', 'custom']:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if sample_mode == 'custom':
            self.enable_anomaly_detection = True
        else:
            self.enable_anomaly_detection = False
        self.sample_mode = sample_mode

        self.verbose = verbose

        if optim_type == 'SGD':
            OptimClass = torch.optim.SGD
        elif optim_type == 'Adam':
            OptimClass = torch.optim.Adam
        elif optim_type == 'AdamW':
            OptimClass = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optim_type: {optim_type}")

        if seg_mode == "input":
            upper_module.givens = nn.ModuleList(
                [
                    GivensOrthogonal(
                        k=sz,
                        givens_num=givens_num,
                        init_vecs_mode=init_vecs_mode,
                        generator=None,
                        dtype=dtype,
                        device=device,
                        requires_grad=True,
                        enable_autograd=enable_autograd,
                        enable_grad_buffer=enable_grad_buffer,
                        enable_low_memory_grad=enable_low_memory_grad,
                    ) for sz in chunksizes
                ]
            )
        elif seg_mode == "weight":
            upper_module.givens = nn.ModuleList(
                [
                    GivensOrthogonal(
                        k=in_features,
                        givens_num=givens_num,
                        init_vecs_mode=init_vecs_mode,
                        generator=None,
                        dtype=dtype,
                        device=device,
                        requires_grad=True,
                        enable_autograd=enable_autograd,
                        enable_grad_buffer=enable_grad_buffer,
                        enable_low_memory_grad=enable_low_memory_grad,
                    )
                    for _ in chunksizes
                ]
            )

        self.optims = [
            OptimClass(params=upper_module.givens[i].parameters(), **optim_config)
            for i in range(chunks)
        ]

        self.detectors = []
        if seg_mode == "input":
            for i, sz in enumerate(chunksizes):
                if self.enable_anomaly_detection:
                    self.detectors.append(
                        AnomalyChannelDetector(k=sz, alpha=0.5, device=device)
                    )
        elif seg_mode == "weight":
            for i, sz in enumerate(chunksizes):
                if self.enable_anomaly_detection:
                    self.detectors.append(
                        AnomalyChannelDetector(k=in_features, alpha=0.5, device=device)
                    )

        self.enable_autograd = enable_autograd
        if enable_autograd:
            raise NotImplementedError("Autograd mode not implemented yet.")

        self.dtype = dtype

        criteria = stop_criteria
        max_steps = criteria.get("max_steps", 100)
        grad_tol = criteria.get("grad_tol", 1e-4)
        grad_diff_rel = criteria.get("grad_diff_rel", 0.1)
        patience = criteria.get("patience", 5)
        ema_decay = criteria.get("ema_decay", 0.9)
        check_every = criteria.get("check_every", 1)

        self.learning_config = {
            "mini_batch_size": mini_batch_size,
            "max_steps": max_steps,
            "grad_tol": grad_tol,
            "grad_diff_rel": grad_diff_rel,
            "patience": patience,
            "ema_decay": ema_decay,
            "check_every": check_every,
        }

        self.learning_state = {
            "loss_ema": [None] * chunks,
            "grad_ema": [None] * chunks,
            "prev_grad": [None] * chunks,
            "stop_counter": [0] * chunks,
            "active": [True] * chunks,
        }

        self.n_samples = [0] * chunks
        self.losses = [None] * chunks
        self.grads = [None] * chunks
        self.best_loss = [None] * chunks
        self.best_state = [None] * chunks

        self.has_givens_optimized = False
        self.has_finished_grads = False

    def __repr__(self):
        optim_type = self.optims[0].__class__.__name__ if self.optims else "N/A"

        main_info = (
            f"{self.__class__.__name__}("
            f"optim_type={optim_type}, "
            f"sample_mode='{self.sample_mode}', "
            f"sub_optimizer={repr(self.sub_optimizer)}, "
            f"givens={repr(self.upper_module.givens)}"
            ")"
        )
        return main_info

    @staticmethod
    def score2pair(k, givens_num, scores):
        assert scores.shape[0] == k
        _, idx_sorted = torch.sort(scores, descending=True)
        idx_sorted = idx_sorted.tolist()
        pairs = []
        indices = list(range(k))
        for delta in range(k - 1):
            this_left_sub_indices = indices[: k - delta]
            this_right_sub_indices = indices[delta:]
            for i in range(len(this_left_sub_indices) // 2):
                pairs.append(
                    (
                        idx_sorted[this_left_sub_indices[i]],
                        idx_sorted[this_left_sub_indices[-(i + 1)]],
                    )
                )
                if len(pairs) >= givens_num:
                    return pairs
            if delta != 0:
                for i in range(len(this_right_sub_indices) // 2):
                    pairs.append(
                        (
                            idx_sorted[this_right_sub_indices[i]],
                            idx_sorted[this_right_sub_indices[-(i + 1)]],
                        )
                    )
                    if len(pairs) >= givens_num:
                        return pairs
        return pairs

    @staticmethod
    @torch.no_grad()
    def loss(manager: GivensOrthogonal, X, W, quantizer_X, quantizer_W):
        Q = manager.forward()
        tmp_x = X @ Q
        tmp_w = Q.t() @ W
        qX = quantizer_X.fake_quantize(tmp_x)
        qW = quantizer_W.fake_quantize(tmp_w)

        term = X @ W - qX @ qW
        return torch.norm(term, p="fro") ** 2

    @staticmethod
    @torch.no_grad()
    def manual_grad_Q(Q: torch.Tensor, X: torch.Tensor, W: torch.Tensor,
                  quantizer_X, quantizer_W):
        """
        Compute manual STE gradient w.r.t. Q for:
            L = || Quant(X @ Q) @ Quant(Q.T @ W) - X @ W ||_F^2

        Args:
            Q: (k, k)
            X: (batch, k)
            W: (k, n)
            quantizer_X, quantizer_W: objects with .fake_quantize(tensor) -> tensor

        Returns:
            grad_Q: tensor of shape (k, k) equal to dL/dQ (or averaged if reduction="mean")
        """
        # forward
        XQ = X @ Q               # (m, k)
        QW = Q.t() @ W           # (k, n)

        A = quantizer_X.fake_quantize(XQ)   # (m, k)
        B = quantizer_W.fake_quantize(QW)   # (k, n)

        Y_pred = A @ B           # (m, n)
        Y_true = X @ W           # (m, n)
        E = Y_pred - Y_true      # (m, n)

        G = 2.0 * E              # (m, n)

        # grad parts
        # part1 = X^T * G * B^T
        # part2 = A^T * G * W^T
        grad1 = X.t().mm(G).mm(B.t())     # (k, k)
        grad2 = A.t().mm(G).mm(W.t())     # (k, k)

        grad_Q = (grad1 + grad2)
        return grad_Q

    @staticmethod
    @torch.no_grad()
    def manual_grad(manager: GivensOrthogonal, X, W, quantizer_X, quantizer_W):
        Q = manager.forward()
        gQ = GivensOptimizer.manual_grad_Q(Q, X, W, quantizer_X, quantizer_W)
        g = manager.try_grad(chain_grad=gQ)
        return g

    def __getstate__(self):
        return {
            "sub_optimizer": self.sub_optimizer,
            "upper_module": getattr(self, "upper_module", None),
        }

    def __setstate__(self, state):
        self.sub_optimizer = state["sub_optimizer"]
        self.upper_module = state.get("upper_module", None)

    def __getattr__(self, name):
        return getattr(self.sub_optimizer, name)

    def __setattr__(self, name, value):
        if name in ("sub_optimizer", "upper_module") or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.sub_optimizer, name, value)

    def __delattr__(self, name):
        if name in ("sub_optimizer", "upper_module") or name in self.__dict__:
            super().__delattr__(name)
        else:
            delattr(self.sub_optimizer, name)

    def process_detector(self, Xs, Ws):
        for i, (x, w, detector) in enumerate(zip(Xs, Ws, self.detectors)):
            if len(x.shape) == 2:
                pass
            elif len(x.shape) == 1:
                x = x.unsqueeze(0)
            else:
                x = x.reshape((-1, x.shape[-1]))
            detector.update(x, w)

    def init_pairs(self, meta=None):
        for i, givens_layer in enumerate(self.upper_module.givens):
            if self.enable_anomaly_detection:
                assert (
                    self.sample_mode == "custom"
                ), "Anomaly detection requires sample_mode='custom'"
                scores = self.detectors[i].get_anomaly_scores()
                if self.verbose:
                    print(
                        f"[{meta}] Chunk {i} anomaly scores: max={scores.max():.4f}, min={scores.min():.4f}, mean={scores.mean():.4f}"
                    )
                givens_layer.init_pairs(
                    self.sample_mode,
                    sample_func=partial(GivensOptimizer.score2pair, scores=scores),
                )

                # debug
                anomaly_scores = scores.detach().cpu()
                selected_pairs = givens_layer.pairs.copy()
                torch.save(
                    {
                        "anomaly_scores": anomaly_scores,
                        "selected_pairs": selected_pairs,
                    },
                    f"anomaly/pairs_{27}_chunk_{i}.pt",
                )
            else:
                givens_layer.init_pairs(self.sample_mode)

    @torch.no_grad()
    def step_cal_grads(
        self,
        Xs,
        Ws,
        iqs,
        wqs,
    ):
        if self.has_finished_grads:
            return

        for i, (x, w_, iq, wq, givens_layer) in enumerate(zip(Xs, Ws, iqs, wqs, self.upper_module.givens)):
            if len(x.shape) == 2:
                this_batch = 1
            elif len(x.shape) == 1:
                x = x.unsqueeze(0)
                this_batch = 1
            else:
                x = x.reshape((-1, x.shape[-1]))
                this_batch = x.shape[0]
            w = w_.to(dtype=self.dtype, device=w_.device).t()  # (in, out)

            self.n_samples[i] += this_batch

            this_loss = GivensOptimizer.loss(
                givens_layer,
                x.to(dtype=self.dtype, device=w.device),
                w,
                iq,
                wq,
            )

            this_grad = GivensOptimizer.manual_grad(
                givens_layer,
                x.to(dtype=self.dtype, device=w.device),
                w,
                iq,
                wq,
            )

            if self.losses[i] is None:
                self.losses[i] = this_loss.item()
                self.grads[i] = this_grad
            else:
                self.losses[i] += this_loss.item()
                self.grads[i] += this_grad

    def step_mini_batch(self, current_step, meta=None):
        loss_ema, grad_ema, prev_grad, stop_counter, active = (
            self.learning_state["loss_ema"],
            self.learning_state["grad_ema"],
            self.learning_state["prev_grad"],
            self.learning_state["stop_counter"],
            self.learning_state["active"],
        )

        grad_tol, grad_diff_rel, patience, ema_decay, check_every = (
            self.learning_config["grad_tol"],
            self.learning_config["grad_diff_rel"],
            self.learning_config["patience"],
            self.learning_config["ema_decay"],
            self.learning_config["check_every"],
        )

        if current_step >= self.learning_config["max_steps"]:
            if self.verbose:
                print(f"Reached max steps {current_step}")
            self.has_finished_grads = True
            return

        if not any(self.learning_state["active"]):
            if self.verbose:
                print(f"All chunks converged by step {current_step}")
            self.has_finished_grads = True
            return

        for i, (givens_layer, optim) in enumerate(zip(self.upper_module.givens, self.optims)):
            if not active[i]:
                continue

            this_loss = self.losses[i] / (self.n_samples[i] + 1e-12)
            if loss_ema[i] is None:
                loss_ema[i] = this_loss
            else:
                loss_ema[i] = ema_decay * loss_ema[i] + (1 - ema_decay) * this_loss

            if self.best_loss[i] is None or loss_ema[i] < self.best_loss[i]:
                self.best_loss[i] = loss_ema[i]
                self.best_state[i] = givens_layer.vecs.detach().clone()
                print(
                    f"[{meta}] New best loss for chunk {i}: {loss_ema[i]:.6f} at step {current_step}"
                )

            optim.zero_grad()
            grad = self.grads[i] / (self.n_samples[i] + 1e-12)
            givens_layer.vecs.grad = grad
            optim.step()
            Q_new = givens_layer.forward()

            # update grad
            if grad_ema[i] is None:
                grad_ema[i] = grad.clone()
            else:
                grad_ema[i] = ema_decay * grad_ema[i] + (1 - ema_decay) * grad

            grad_norm = torch.norm(grad_ema[i]) / grad.numel() ** 0.5
            grad_diff = (
                torch.norm(grad - prev_grad[i]) / (torch.norm(prev_grad[i]) + 1e-12)
                if prev_grad[i] is not None
                else 0.0
            )
            prev_grad[i] = grad.clone()

            if grad_norm < grad_tol or grad_diff < grad_diff_rel:
                stop_counter[i] += 1
            else:
                stop_counter[i] = 0

            if stop_counter[i] >= patience:
                active[i] = False
                if self.verbose:
                    print(f"Chunk {i} converged at step {current_step}")

            if self.verbose and current_step % check_every == 0:
                ortho_err = torch.norm(
                    Q_new @ Q_new.t() - torch.eye(Q_new.shape[0], device=Q_new.device)
                ).item()
                print(
                    f"[{meta}][Chunk {i}] Step {current_step}: "
                    f"loss={loss_ema[i]:.6f}, "
                    f"ortho={ortho_err:.4e}, "
                    f"grad_norm={grad_norm:.6f}, "
                    f"grad_diff={grad_diff:.6f}, "
                    f"stop_count={stop_counter[i]}"
                )

            # reset
            self.n_samples[i] = 0
            self.losses[i] = None
            self.grads[i] = None

    def on_calibrate_prepare_finish(self, Ws):
        if self.verbose:
            print("Givens optimization finished.")
            for i, givens_layer in enumerate(self.upper_module.givens):
                Q_final = givens_layer.forward()
                ortho_err = torch.norm(
                    Q_final @ Q_final.t() - torch.eye(Q_final.shape[0], device=Q_final.device)
                ).item()
                print(f"Final ortho check for chunk {i}: {ortho_err:.4e}")
                print(f"Givens rotations: {givens_layer.vecs[:5]} ...")

                # debug
                vecs = givens_layer.vecs.detach().cpu()
                s = getattr(self.upper_module, f"smooth_s_{i}")
                torch.save(
                    {
                        "vecs": vecs,
                        "smooth": s,
                    },
                    f"anomaly/givens_{27}_chunk{i}.pt",
                )

        ## givens weight
        givens_weight_chunks = []
        for i, (w, givens_layer, bs) in enumerate(zip(Ws, self.upper_module.givens, self.best_state)):
            givens_layer.vecs.data.copy_(
                bs.to(device=givens_layer.vecs.device, dtype=givens_layer.vecs.dtype)
            )
            Q = givens_layer.forward()
            givens_weight_chunks.append((w.to(dtype=self.dtype, device=Q.device) @ Q).to(dtype=w.dtype, device=w.device))  # (Q.t @ W.t).t --> (W @ Q)

            # clear buffer
            givens_layer.clear_buffer()

        self.has_givens_optimized = True

        del self.learning_state
        del self.n_samples
        del self.losses
        del self.grads
        del self.best_loss
        del self.best_state
        del self.detectors
        del self.learning_config

        return givens_weight_chunks

    def on_calibrate_finish_end(self):
        return self.sub_optimizer.on_calibrate_finish_end()

    def preprocess_x(self, x_chunks):
        # slow version for fake quant
        x_chunks = self.sub_optimizer.preprocess_x(x_chunks)
        if self.has_givens_optimized:
            givens_x_chunks = []
            for i, (x_chunk, givens_layer) in enumerate(
                zip(x_chunks, self.upper_module.givens)
            ):
                givens_x_chunks.append(x_chunk @ givens_layer.forward().to(dtype=x_chunk.dtype, device=x_chunk.device))

            return givens_x_chunks
        else:
            return x_chunks
