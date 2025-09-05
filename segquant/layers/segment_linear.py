from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from segquant.calibrator.calibrator import CalibratorRegistry
from segquant.config import Calibrate, DType, Optimum
from segquant.optimum.optimizer import OptimizerRegistry
from segquant.layers import ext_dict

class SegmentLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        seg_mode: Literal["input", "weight"] = "weight",
        chunks=1,
        chunksizes=None,
        custom_weight=None,
        custom_bias=None,
        input_quant_type=None,
        weight_quant_type=None,
        input_quant_args=None,
        weight_quant_args=None,
        opt_type: Literal["default", "smooth", "svd", "givens"] = "default",
        opt_kwargs=None,
        calib_type: Literal["amax", "gptq"] = "amax",
        calib_kwargs=None,
        device='cuda:0',
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if custom_weight is not None:
            assert (
                custom_weight.shape[0] == out_features
                and custom_weight.shape[1] == in_features
            ), (
                f"Mismatched custom_weight shape! "
                f"Expected ({out_features}, {in_features}), "
                f"but got {custom_weight.shape}."
            )
            ### use new space
            custom_weight = custom_weight.to(device).clone()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
        else:
            custom_weight = torch.randn(
                [self.out_features, self.in_features], device=device
            )
        self.bias = bias
        self.bias_data = None
        if self.bias:
            if custom_bias is not None:
                self.bias_data = nn.Parameter(custom_bias.to(device).clone())
            else:
                self.bias_data = nn.Parameter(torch.zeros([self.out_features], device=device))

        if seg_mode == "input":
            target_features = in_features
        elif seg_mode == "weight":
            target_features = out_features
        else:
            raise ValueError("seg_mode not found")
        self.seg_mode = seg_mode
        self.chunks = chunks
        if chunksizes is None:
            chunk_size = target_features // self.chunks
            chunksizes = [chunk_size for _ in range(self.chunks)]
            remain_size = target_features - chunk_size * self.chunks
            for i in range(remain_size):
                chunksizes[i] += 1
        else:
            assert len(chunksizes) == self.chunks and sum(chunksizes) == target_features
        self.chunksizes = chunksizes

        real_quant = False
        dual_scale = False
        if input_quant_type is not None:
            if "real_quant" in input_quant_args:
                real_quant = input_quant_args["real_quant"]
            if "dual_scale" in input_quant_args:
                dual_scale = input_quant_args["dual_scale"]

        if weight_quant_type is not None:
            if "real_quant" in weight_quant_args:
                assert real_quant == weight_quant_args["real_quant"], (
                    f"Mismatch: real_quant={real_quant}, "
                    f"weight_quant_args['real_quant']={weight_quant_args['real_quant']}"
                )

            if "dual_scale" in weight_quant_args:
                raise ValueError(
                    "Dual scale is not supported for weight quantizer in DefaultSegmentLinear."
                )

        kernel_type = f"W{weight_quant_type}A{input_quant_type}"

        if real_quant:
            if kernel_type not in ext_dict:
                print(f"[Warning] Seglinear need [{kernel_type}] but not found")
                real_quant = False
            else:
                if (
                    ext_dict[kernel_type]["gemm_scaled_fn"] is None
                    or ext_dict[kernel_type]["gemm_dual_scaled_fn"] is None
                ):
                    print(
                        f"[Warning] Seglinear need [{kernel_type}] but function is None"
                    )
                    real_quant = False

        input_quant_args["real_quant"] = real_quant
        weight_quant_args["real_quant"] = real_quant

        (
            self.input_quantizers_len,
            self.weight_quantizers_len,
            self.split_input_func,
            self.split_weight_func,
        ) = OptimizerRegistry.get_segment_config(
            opt_type, seg_mode, self.chunks
        )

        self.input_quantizers = [
            CalibratorRegistry.create(
                "amax",
                data_type="input",
                quant_type=input_quant_type,
                quant_args=input_quant_args,
                **calib_kwargs,
            )
            for _ in range(self.input_quantizers_len)
        ]

        self.weight_quantizers = [
            CalibratorRegistry.create(
                calib_type,
                data_type="weight",
                quant_type=weight_quant_type,
                quant_args=weight_quant_args,
                **calib_kwargs,
            )
            for _ in range(self.weight_quantizers_len)
        ]

        self.weight_chunks = nn.ParameterList(
            self.split_weight_func(custom_weight, self.chunksizes)
        )

        self.opt_type = opt_type
        self.optimizer = OptimizerRegistry.create(
            opt_type,
            upper_module=self,
            in_features=in_features,
            out_features=out_features,
            seg_mode=seg_mode,
            chunks=self.chunks,
            chunksizes=self.chunksizes,
            device=device,
            **opt_kwargs,
        )

        self.has_traces_w = False
        self.has_calibrated_w = False
        self.has_calculated_epsw = False

        if real_quant:
            ## func == kernel_type + segtype/single + opt_type + dual_scale
            raise NotImplementedError
            if real_quant and dual_scale:
                self.func_name = 'gemm_dual_scaled_fn'
            else:
                self.func_name = 'gemm_scaled_fn'
        else:
            self.func = self._fake_func

    def _fake_func(self, x, chunked=False):
        x_chunks = self.split_input_func(x, self.chunksizes)
        preprocess_x_chunks = self.optimizer.preprocess_x(x_chunks)
        quantized_input_chunks = [
            iq.fake_quantize(x) for iq, x in zip(self.input_quantizers, preprocess_x_chunks)
        ]

        ## weight has quantized in finish_calibrate
        quantized_weight_chunks = SegmentLinear.rearrange_w_chunks_by_x(self.weight_chunks, quantized_input_chunks, self.chunksizes)

        assert len(quantized_input_chunks) == len(quantized_weight_chunks), "Input and weight chunks length mismatch"

        if chunked:
            # no bias
            quantized_output_chunks = self.optimizer.chunk_forward(quantized_input_chunks, quantized_weight_chunks)

            if self.chunks > 1 and len(quantized_output_chunks) == 1:
                assert self.opt_type == 'default', "Only default opt_type can return unchunked output"
                return quantized_output_chunks[0].split(self.chunksizes, dim=1)
            return quantized_output_chunks
        else:
            # add bias
            quantized_output_chunks = self.optimizer.chunk_forward(
                quantized_input_chunks, quantized_weight_chunks
            )
            if self.seg_mode == 'input':
                return torch.sum(torch.stack(quantized_output_chunks), dim=0)
            elif self.seg_mode == 'weight':
                res = torch.cat(quantized_output_chunks, dim=1)
                if self.bias:
                    res = res + self.bias_data
                return res
            else:
                raise ValueError("seg_mode not found")

    def __repr__(self):
        lines = [
            f"  in_features={self.in_features}, out_features={self.out_features}, seg_mode={self.seg_mode}",
            f"  chunks={self.chunks}, chunksize={self.chunksizes}",
            f"  opt={repr(self.optimizer)}",
        ]
        inner_content = ",\n".join(lines)
        base = f"SegmentLinear(\n{inner_content}\n"

        input_q = ",\n    ".join(repr(i) for i in self.input_quantizers)
        weight_q = ",\n    ".join(repr(w) for w in self.weight_quantizers)

        if self.chunks == 1:
            return (
                f"{base}  input_quantizer=({input_q}),\n"
                f"  weight_quantizer=({weight_q})\n"
                f")"
            )
        else:
            return (
                f"{base}    input_quantizers=[\n    {input_q}\n  ],\n"
                f"    weight_quantizers=[\n    {weight_q}\n  ]\n"
                f")"
            )

    @staticmethod
    def rearrange_w_chunks_by_x(weight_chunks, x_chunks, chunksizes):
        if len(x_chunks) > len(weight_chunks):
            # split W
            res = weight_chunks[0].split(chunksizes, dim=-1)
        elif len(x_chunks) < len(weight_chunks):
            # concat W
            res = [torch.cat(list(weight_chunks), dim=0)]
        else:
            res = weight_chunks
        return res  

    @staticmethod
    def rearrange_x_chunks_by_w(x_chunks, weight_chunks):
        if len(x_chunks) > len(weight_chunks):
            # concat X
            res = [torch.cat(x_chunks, dim=-1)]
        elif len(x_chunks) < len(weight_chunks):
            # repeat X
            res = x_chunks * len(weight_chunks)
        else:
            res = x_chunks
        return res

    def trace(self, x):
        x_chunks = self.split_input_func(x, self.chunksizes)
        w_chunks = None
        if not self.has_traces_w:
            w_chunks = self.weight_chunks
            self.has_traces_w = True
        self.optimizer.trace(x_chunks, w_chunks)

    @torch.no_grad()
    def finish_trace(self):
        new_chunks = self.optimizer.on_trace_finish(self.weight_chunks)
        for p, new_p in zip(self.weight_chunks, new_chunks):
            p.copy_(new_p)

    def reset(self, origin_weight):
        new_chunks = self.split_weight_func(origin_weight, self.chunksizes)
        assert len(new_chunks) == len(self.weight_chunks), "Parameter count mismatch"

        for p, new_p in zip(self.weight_chunks, new_chunks):
            p.data = new_p.to(p.device, p.dtype).clone()

        for input_quantizer in self.input_quantizers:
            input_quantizer.reset()
        for weight_quantizer in self.weight_quantizers:
            weight_quantizer.reset()

        self.has_calibrated_w = False

    def calibrate(self, x):
        x_chunks = self.split_input_func(x, self.chunksizes)
        processed_x_chunks = self.optimizer.preprocess_x(x_chunks)
        for iq, processed_x_chunk in zip(self.input_quantizers, processed_x_chunks):
            iq.calibrate(processed_x_chunk)

        # for GPTQ weight
        if any(hasattr(wq, 'stat') for wq in self.weight_quantizers):
            # handle x unbalanced chunks
            fit_x_chunks = SegmentLinear.rearrange_x_chunks_by_w(
                processed_x_chunks, self.weight_chunks
            )
            for i, wq in enumerate(self.weight_quantizers):
                wq.stat(fit_x_chunks[i].clone())

        if not self.has_calibrated_w:
            for wq, processed_weight_chunk in zip(
                self.weight_quantizers, self.weight_chunks
            ):
                wq.calibrate(processed_weight_chunk)
            self.has_calibrated_w = True

    def after_calibrate(self, x):
        x_chunks = self.split_input_func(x, self.chunksizes)

        ### calculate eps for X
        Xs = self.optimizer.preprocess_x(x_chunks)
        eXs = [iq.fake_quantize(x) - x for iq, x in zip(self.input_quantizers, Xs)]
        if len(Xs) < len(self.weight_chunks):
            Xs = Xs * len(self.weight_chunks)
            eXs = eXs * len(self.weight_chunks)

        if not self.has_calculated_epsw:
            ### calculate eps for W
            if len(self.weight_chunks) < self.chunks:
                Ws = self.weight_chunks[0].split(self.chunksizes, dim=1)
            else:
                Ws = list(self.weight_chunks)
            eWs = [wq.fake_quantize(w) - w for wq, w in zip(self.weight_quantizers, Ws)]
        else:
            Ws = [None] * len(self.chunks)
            eWs = [None] * len(self.chunks)

        self.optimizer.stat_error(Xs, eXs, Ws, eWs)

    def finish_calibrate(self):
        for iq in self.input_quantizers:
            iq.finish_calibrate()

        # change weight_chunks to quantized weight_chunks
        Ws = list(self.weight_chunks)
        eWs = [wq.fake_quantize(w) - w for wq, w in zip(self.weight_quantizers, Ws)]
        Ws = self.optimizer.on_calibrate_prepare_finish(Ws, eWs)
        for i, wq in enumerate(self.weight_quantizers):
            self.weight_chunks[i] = nn.Parameter(wq.finish_calibrate(Ws[i]))

        opt_finished = self.optimizer.on_calibrate_finish_end()
        if opt_finished:
            for i, input_quantizer in enumerate(self.input_quantizers):
                self.input_quantizers[i] = input_quantizer.quantizer
            for i, weight_quantizer in enumerate(self.weight_quantizers):
                self.weight_quantizers[i] = weight_quantizer.quantizer

    @torch.no_grad()
    def segment_forward(self, x, weight):
        # only for search
        input_chunks = self.split_input_func(x, self.chunksizes)
        weight_chunks = self.split_weight_func(weight, self.chunksizes)

        output_chunks = []
        if self.seg_mode == "weight":
            for w in weight_chunks:
                output_chunks.append(F.linear(input_chunks[0], w))
        elif self.seg_mode == "input":
            if len(weight_chunks) < self.chunks:
                this_weight_chunks = weight_chunks[0].split(self.chunksizes, dim=1)
            else:
                this_weight_chunks = weight_chunks
            for i, inp in enumerate(input_chunks):
                output_chunks.append(F.linear(inp, this_weight_chunks[i]))

        return output_chunks

    def forward(self, x, chunked=False):
        return self.func(x, chunked)


def create_segment_linear(
    input_dtype: DType,
    weight_dtype: DType,
    opt: Optimum,
    calib: Calibrate,
    in_features,
    out_features,
    opt_kwargs,
    calib_kwargs,
    input_quant_args,
    weight_quant_args,
    **kwargs,
):

    return SegmentLinear(
        in_features,
        out_features,
        input_quant_type=input_dtype.value,
        weight_quant_type=weight_dtype.value,
        opt_type=opt.value,
        calib_type=calib.value,
        opt_kwargs=opt_kwargs,
        calib_kwargs=calib_kwargs,
        input_quant_args=input_quant_args,
        weight_quant_args=weight_quant_args,
        **kwargs,
    )
