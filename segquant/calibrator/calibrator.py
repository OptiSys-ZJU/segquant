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
        self.err = None

        self.cpu_storage = cpu_storage

    def __repr__(self):
        base = (
            f"GPTQCalibrator(blocksize={self.blocksize}, percdamp={self.percdamp}, groupsize={self.groupsize}, actorder={self.actorder}, static_groups={self.static_groups},\n"
            f"      err={self.err:4f}\n"
            f"      quantizer={repr(self.quantizer)})"
        )
        return base

    def calibrate(self, x, input_data):
        if len(input_data.shape) == 2:
            this_batch = 1
        elif len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
            this_batch = 1
        else:
            input_data = input_data.reshape((-1, input_data.shape[-1]))
            this_batch = input_data.shape[0]

        input_data = input_data.t()  # (b, in) -> (in, b)

        device = x.device

        if self.H is None:
            self.H = torch.zeros(
                (x.shape[1], x.shape[1]),
                device='cpu' if self.cpu_storage else device,
                dtype=torch.float32,
            )

        H = self.H.to(device=device) if self.cpu_storage else self.H

        if input_data.dtype != torch.float32:
            input_data = input_data.to(dtype=torch.float32)

        input_data = input_data.to(device)
        H.mul_(self.nsamples / (self.nsamples + this_batch))
        self.nsamples += this_batch

        input_data.mul_(math.sqrt(2 / self.nsamples))
        H.addmm_(input_data, input_data.t(), beta=1.0, alpha=1.0)

        if self.cpu_storage:
            self.H.copy_(H.to(device='cpu'))

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

        H = self.H.to(dtype=torch.float32, device=weight_data.device) if self.cpu_storage else self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

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

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(column, device=W.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

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

                q = self.quantizer.fake_quantize(w) # (out,)
                real_q = self.quantizer.quantize(w.unsqueeze(1).to(dtype)) # (out, 1)
                if Q is None:
                    Q = real_q
                else:
                    Q = torch.hstack((Q, real_q))
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        if self.data_type == 'weight':
            self.err = torch.sum(Losses).item()
            print(
                f"GPTQCalibrator: finish_calibrate [{W.shape[0]}, {W.shape[1]}], error [{self.err:4f}]"
            )
            return Q

        return None
