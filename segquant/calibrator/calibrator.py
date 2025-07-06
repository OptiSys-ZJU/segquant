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

from typing import Literal
import math
import torch
from segquant.quantizers.quantizer import QuantizerRegistry


class BaseCalibrator:
    """
    Base class for calibrators.
    This class is used to calibrate quantizers for input and weight tensors.
    It is designed to be subclassed for specific calibration strategies.
    """
    def __init__(self, data_type: Literal['weight', 'input'], quant_type, quant_args):
        self.data_type = data_type
        self.quantizer = QuantizerRegistry.create(
            quant_type, **(quant_args or {})
        )
    
    def quantize(self, x):
        return self.quantizer.quantize(x)

    @property
    def pos_scale(self):
        return self.quantizer.pos_scale

    @property
    def neg_scale(self):
        return self.quantizer.neg_scale

    @property
    def scale(self):
        return self.quantizer.scale
    
    def to(self, device):
        if hasattr(self.quantizer, 'to'):
            self.quantizer = self.quantizer.to(device)
        return self

    def __repr__(self):
        return repr(self.quantizer)

class CalibratorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(calibrator_cls):
            cls._registry[name] = calibrator_cls
            return calibrator_cls

        return wrapper

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def create(cls, name, **kwargs):
        calibrator_cls = cls.get(name)
        if calibrator_cls is None:
            raise ValueError(f"Calibrator '{name}' not found in registry.")
        return calibrator_cls(**kwargs)

@CalibratorRegistry.register("amax")
class AMaxCalibrator(BaseCalibrator):
    def __init__(self, data_type, quant_type, quant_args, **kwargs,):
        super().__init__(data_type, quant_type, quant_args)
        self.has_calibrated = False

    def __repr__(self):
        base = (
            f"AMaxCalibrator(quantizer={repr(self.quantizer)})"
        )
        return base
    
    def reset(self):
        self.has_calibrated = False
        self.quantizer.reset()

    def calibrate(self, x, **kwargs):
        if self.data_type == 'weight' and self.has_calibrated:
            return
        self.quantizer.calibrate(x)
        self.has_calibrated = True

    def finish_calibrate(self, weight_data=None):
        if self.data_type == 'weight':
            return self.quantizer.quantize(weight_data)
        return None

@CalibratorRegistry.register("gptq")
class GPTQCalibrator(BaseCalibrator):
    def __init__(
        self,
        data_type: Literal["weight", "input"],
        quant_type,
        quant_args,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        cpu_storage=False,
        verbose=False,
        **kwargs,
    ):
        assert data_type == 'weight', 'Only weight calibrator can be used to GPTQ.'
        super().__init__(data_type, quant_type, quant_args)
        self.nsamples = 0
        self.H = None

        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.static_groups = static_groups
        self.err = 0
        self.verbose = verbose

        self.cpu_storage = cpu_storage
        if self.cpu_storage:
            print('[Warning] GPTQCalibrator set CPU Storage.')

    def __repr__(self):
        base = (
            f"GPTQCalibrator(blocksize={self.blocksize}, percdamp={self.percdamp}, groupsize={self.groupsize}, actorder={self.actorder}, static_groups={self.static_groups},\n"
            f"      err={self.err:4f}\n"
            f"      quantizer={repr(self.quantizer)})"
        )
        return base

    def reset(self):
        self.nsamples = 0
        self.H.zero_()
        self.err = 0
        self.quantizer.reset()

    def calibrate(self, x, input_data):
        if len(input_data.shape) == 2:
            this_batch = 1
        elif len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
            this_batch = 1
        else:
            input_data = input_data.reshape((-1, input_data.shape[-1]))
            this_batch = input_data.shape[0]

        input_data = input_data.to(dtype=torch.float32, device=x.device).t()  # (in, b)
        device = x.device

        if self.H is None:
            self.H = torch.zeros(
                (x.shape[1], x.shape[1]),
                device='cpu' if self.cpu_storage else device,
                dtype=torch.float32,
                pin_memory=True if self.cpu_storage else False
            )

        if self.cpu_storage:
            H = self.H.to(device, non_blocking=True)
        else:
            H = self.H

        H.mul_(self.nsamples / (self.nsamples + this_batch))
        self.nsamples += this_batch

        scale = math.sqrt(2 / self.nsamples)
        input_data.mul_(scale)

        H.addmm_(input_data, input_data.t(), beta=1.0, alpha=1.0)

        if self.cpu_storage:
            self.H.copy_(H, non_blocking=True)
    
    def colpacked1d_to_rowpacked1d(self, Q_colpacked_1d: torch.Tensor, out: int, cols: int) -> torch.Tensor:
        assert out % 2 == 0 and cols % 2 == 0, "Output and columns must be even for int4 quantization."

        Q_colpacked = Q_colpacked_1d.view(cols, out // 2).T  # [out//2, cols]

        packed_rows = []
        for row in Q_colpacked:
            low = row & 0x0F
            low0 = low[0::2]
            low1 = low[1::2] << 4
            low_packed = (low1 | low0).to(torch.uint8)  # shape: (col // 2,)

            high = (row >> 4) & 0x0F
            high0 = high[0::2]
            high1 = high[1::2] << 4
            high_packed = (high1 | high0).to(torch.uint8)  # shape: (col // 2,)

            packed_rows.append(low_packed)
            packed_rows.append(high_packed)
        
        Q_repacked = torch.stack(packed_rows, dim=0)  # shape: (2 * rows, col // 2)
        return Q_repacked.reshape(-1)  # shape: (out * cols // 2,)

    def finish_calibrate(self, weight_data=None):
        blocksize = self.blocksize
        percdamp = self.percdamp
        groupsize = self.groupsize
        actorder = self.actorder
        static_groups = self.static_groups
        dtype = weight_data.dtype

        W = weight_data
        W = W.float()
        column = W.shape[1]

        self.quantizer.calibrate(W)

        H = (
            self.H.to(dtype=torch.float32, device=weight_data.device, non_blocking=True)
            if self.cpu_storage
            else self.H
        )

        diag_H = torch.diag(H)
        dead = diag_H < 1e-8
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        if static_groups:
            import copy
            groups = []
            for i in range(0, column, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                self.quantizer.calibrate(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = None

        H = (H + H.T) / 2
        damp = percdamp * torch.mean(torch.diag(H)).item()
        identity = torch.eye(column, device=W.device)

        for i in range(10):
            try:
                H_damped = H + identity * damp
                chol = torch.linalg.cholesky(H_damped)
                break
            except torch._C._LinAlgError:
                damp *= 10
        else:
            print(
                "[Warning] GPTQCalibrator: Cholesky failed after multiple damping attempts. H may be ill-conditioned, disable GPTQ"
            )
            return self.quantizer.quantize(W.to(dtype))
        Hinv = torch.cholesky_inverse(chol)

        for i1 in range(0, column, blocksize):
            i2 = min(i1 + blocksize, column)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i] #(out,)
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.calibrate(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.fake_quantize(w.unsqueeze(1)).squeeze(1) # (out, 1)
                real_q = self.quantizer.quantize(w.unsqueeze(1).to(dtype)) # (out, 1) or (out//2,)
                if Q is None:
                    Q = real_q
                else:
                    Q = torch.hstack((Q, real_q)) # (out, in) or (out//2 * in,)
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]
        
        if hasattr(self.quantizer, 'num_bits') and self.quantizer.num_bits == 4 and self.quantizer.real_quant:
            Q = self.colpacked1d_to_rowpacked1d(Q, W.shape[0], W.shape[1])

        if self.data_type == 'weight':
            self.err = torch.sum(Losses).item()
            if self.verbose:
                print(
                    f"GPTQCalibrator: finish_calibrate [{W.shape[0]}, {W.shape[1]}], error [{self.err:4f}]"
                )
            return Q

        return None
