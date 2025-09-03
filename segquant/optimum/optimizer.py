from functools import partial
import math
from typing import List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segquant.layers.givens_orthogonal import GivensOrthogonal


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

    def trace(self, x_chunks, w_chunks):
        raise NotImplementedError("trace method not implemented")

    def preprocess_x(self, x_chunks):
        return x_chunks

    def on_calibrate_prepare_finish(self, Ws, eWs):
        return Ws

    def on_calibrate_finish_end(self):
        pass

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
        val_sample_batch=0,
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

        if sample_mode not in ['rand', 'ascending', 'descending', 'custom']:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if sample_mode == 'custom':
            # todo custom func
            sample_func = None
            raise NotImplementedError("Custom sample_mode not implemented yet.")
        else:
            sample_func = None

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
                        sample_mode=sample_mode,
                        sample_func=sample_func,
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
                        sample_mode=sample_mode,
                        sample_func=sample_func,
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

        self.enable_autograd = enable_autograd
        if not enable_autograd:
            # make buffer
            buffer_device = device if not cpu_storage else torch.device('cpu')
            if seg_mode == "input":
                for i, sz in enumerate(chunksizes):
                    upper_module.register_buffer(
                        f"givens_X_rel_{i}",
                        torch.zeros((3, sz, sz), dtype=dtype, device=buffer_device),
                        persistent=False,
                    )  # (xtx, xtex, extex) dim = 0
            elif seg_mode == "weight":
                for i, sz in enumerate(chunksizes):
                    upper_module.register_buffer(
                        f"givens_X_rel_{i}",
                        torch.zeros(
                            (3, in_features, in_features),
                            dtype=dtype,
                            device=buffer_device,
                        ),
                        persistent=False,
                    )  # (xtx, xtex, extex) dim = 0

        self.dtype = dtype
        self.nsamples = 0
        self.stop_criteria = stop_criteria
        if enable_autograd:
            self.val_sample_batch = 0
            self.losses = [
                torch.tensor(
                    0.0, device=device, dtype=self.dtype, requires_grad=True
                )
                for _ in range(chunks)
            ]
            self.Ws = None
            self.eWs = None
        self.val_sample_batch = val_sample_batch
        self.val_x_buffer = []

        self.has_givens_optimized = False

    @staticmethod
    def loss(manager: GivensOrthogonal, X, W, epsX, epsW):
        Q = manager.forward()
        term = X @ Q @ epsW + epsX @ Q.t() @ W + epsX @ epsW
        return torch.norm(term, p="fro") ** 2, Q

    @staticmethod
    @torch.no_grad()
    def manual_grad_Q_buffer(Q, XtX, XteX, eXteX, W, eW):
        eWeWt = eW @ eW.t()
        WWt = W @ W.t()
        WeWt = W @ eW.t()

        grad = (
            XtX @ Q @ eWeWt
            + WWt @ Q @ eXteX
            + XteX @ Q.t() @ WeWt
            + WeWt @ Q.t() @ XteX
            + XteX @ eWeWt
            + WeWt @ eXteX
        )

        return grad

    @staticmethod
    @torch.no_grad()
    def manual_grad_buffer(manager: GivensOrthogonal, X_buffer, W, epsW):
        Q = manager.forward()
        XtX, XteX, eXteX = X_buffer
        gQ = GivensOptimizer.manual_grad_Q_buffer(Q, XtX, XteX, eXteX, W, epsW)
        g = manager.try_grad(chain_grad=gQ)
        return g

    @staticmethod
    @torch.no_grad()
    def manual_grad_Q(Q, X, W, epsX, epsW):
        XtX = X.t() @ X
        WWT = W @ W.t()
        epsXtX = epsX.t() @ epsX
        epsWDW = epsW @ epsW.t()

        grad = 2 * (XtX @ Q @ epsWDW + WWT @ Q @ epsXtX)
        grad += 2 * (
            X.t() @ epsX @ Q.t() @ W @ epsW.t() + W @ epsW.t() @ Q.t() @ X.t() @ epsX
        )
        grad += 2 * (X.t() @ epsX @ epsWDW + W @ epsW.t() @ epsXtX)
        return grad

    @staticmethod
    @torch.no_grad()
    def manual_grad(manager: GivensOrthogonal, X, W, epsX, epsW):
        Q = manager.forward()
        gQ = GivensOptimizer.manual_grad_Q(Q, X, W, epsX, epsW)
        g = manager.try_grad(chain_grad=gQ)
        return g

    def __getattr__(self, name):
        return getattr(self.sub_optimizer, name)

    def __setattr__(self, name, value):
        if name in ("sub_optimizer", "upper_module") or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.sub_optimizer, name, value)

    @torch.no_grad()
    def stat_error(self, Xs: List[torch.tensor], eXs: List[torch.tensor], Ws: List[torch.tensor], eWs: List[torch.tensor]):
        if self.enable_autograd:
            if Ws is not None and eWs is not None:
                self.Ws = Ws
                self.eWs = eWs

            for i, (x, ex, w, ew) in enumerate(zip(Xs, eXs, self.Ws, self.eWs)):
                if self.val_sample_batch > 0:
                    x_cpu = x.detach().cpu()
                    ex_cpu = ex.detach().cpu()
                    self.val_x_buffer.append((x_cpu, ex_cpu))
                    self.val_sample_batch -= 1

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
                self.nsamples += this_batch

                loss, _ = GivensOptimizer.loss(
                    self.upper_module.givens[i],
                    x.to(w.device),
                    w.to(w.device),
                    ex.to(w.device),
                    ew.to(w.device),
                )
                self.losses[i] = self.losses[i] + loss
        else:
            for i, (x, ex) in enumerate(zip(Xs, eXs)):
                if self.val_sample_batch > 0:
                    x_cpu = x.detach().cpu()
                    ex_cpu = ex.detach().cpu()
                    self.val_x_buffer.append((x_cpu, ex_cpu))
                    self.val_sample_batch -= 1

                # compute xtx, xtex, extex
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

                x = x.to(dtype=self.dtype, device=x.device).t()  # (in, b)
                ex = ex.to(dtype=self.dtype, device=x.device).t()  # (in, b)

                buffer = getattr(
                    self.upper_module, f"givens_X_rel_{i}"
                )

                tmp_buffer = buffer.to(x.device, non_blocking=True)
                tmp_buffer.mul_(self.nsamples / (self.nsamples + this_batch))
                self.nsamples += this_batch

                scale = math.sqrt(2 / self.nsamples)
                x = x * scale
                ex = ex * scale

                tmp_buffer[0].addmm_(x, x.t(), beta=1.0, alpha=1.0)
                tmp_buffer[1].addmm_(x, ex.t(), beta=1.0, alpha=1.0)
                tmp_buffer[2].addmm_(ex, ex.t(), beta=1.0, alpha=1.0)

                buffer.copy_(tmp_buffer.to(buffer.device))

    def on_calibrate_prepare_finish(self, Ws, eWs):
        criteria = self.stop_criteria
        max_steps = criteria.get('max_steps', 100)
        grad_tol = criteria.get('grad_tol', 1e-4)
        grad_change_tol = criteria.get('grad_change_tol', 1e-5)
        patience = criteria.get('patience', 5)
        ema_decay = criteria.get('ema_grad_decay', 0.9)
        check_every = criteria.get('check_every', 1)

        grad_ema = [None] * len(Ws)
        prev_grad = [None] * len(Ws)
        stop_counter = [0] * len(Ws)
        active = [True] * len(Ws)

        if self.enable_autograd:
            for i in range(len(Ws)):
                self.losses[i] = self.losses[i] / self.nsamples

        for step in range(max_steps):
            if not any(active):
                if self.verbose:
                    print(f"All chunks converged by step {step}")
                break

            for i, (w_, ew_, givens_layer, optim) in enumerate(zip(Ws, eWs, self.upper_module.givens, self.optims)):
                if not active[i]:
                    continue

                w = w_.to(dtype=self.dtype, device=w_.device).t()  # (in, out)
                ew = ew_.to(dtype=self.dtype, device=ew_.device).t()  # (in, out)

                ### grad optims
                optim.zero_grad()
                if self.enable_autograd:
                    self.losses[i].backward()
                else:
                    grad = GivensOptimizer.manual_grad_buffer(
                        givens_layer,
                        getattr(self.upper_module, f"givens_X_rel_{i}"),
                        w,
                        ew,
                    )
                    givens_layer.vecs.grad = grad

                optim.step()

                # get loss
                total_loss = 0.0
                Q_new_result = None
                n = len(self.val_x_buffer)
                if n != 0:
                    with torch.no_grad():
                        for idx, (x, ex) in enumerate(self.val_x_buffer):
                            loss, Q_new = GivensOptimizer.loss(givens_layer, x.to(w.device), w, ex.to(w.device), ew)
                            total_loss += loss
                            if Q_new_result is None:
                                Q_new_result = Q_new
                    avg_loss = total_loss / n
                else:
                    Q_new_result = givens_layer.forward()
                    avg_loss = float('nan')

                # update grad
                if grad_ema[i] is None:
                    grad_ema[i] = grad.clone()
                else:
                    grad_ema[i] = ema_decay * grad_ema[i] + (1 - ema_decay) * grad

                grad_norm = torch.norm(grad_ema[i])
                grad_diff = (
                    torch.norm(grad - prev_grad[i]) if prev_grad[i] is not None else 0.0
                )
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
                    ortho_err = torch.norm(
                        Q_new @ Q_new.t() - torch.eye(Q_new.shape[0], device=Q_new.device)
                    ).item()
                    print(
                        f"[Chunk {i}] Step {step}: "
                        f"loss={avg_loss:.6f}, "
                        f"ortho={ortho_err:.4e}, "
                        f"grad_norm={grad_norm:.6f}, "
                        f"grad_diff={grad_diff:.6f}, "
                        f"stop_count={stop_counter[i]}"
                    )

        if self.verbose:
            print("Givens optimization finished.")
            for i, givens_layer in enumerate(self.upper_module.givens):
                Q_final = givens_layer.forward()
                ortho_err = torch.norm(
                    Q_final @ Q_final.t() - torch.eye(Q_final.shape[0], device=Q_final.device)
                ).item()
                print(f"Final ortho check for chunk {i}: {ortho_err:.4e}")
                print(f"Givens rotations: {givens_layer.vecs[:5]} ...")

        ## givens weight
        givens_weight_chunks = []
        for i, (w, givens_layer) in enumerate(zip(Ws, self.upper_module.givens)):
            Q = givens_layer.forward()
            givens_weight_chunks.append(w @ Q)  # (Q.t @ W.t).t --> (W @ Q)
        self.has_givens_optimized = True

        return givens_weight_chunks

    def preprocess_x(self, x_chunks):
        # slow version for fake quant
        x_chunks = self.sub_optimizer.preprocess_x(x_chunks)
        if self.has_givens_optimized:
            givens_x_chunks = []
            for i, (x_chunk, givens_layer) in enumerate(
                zip(x_chunks, self.upper_module.givens)
            ):
                givens_x_chunks.append(x_chunk @ givens_layer.forward())

            return givens_x_chunks
        else:
            return x_chunks
