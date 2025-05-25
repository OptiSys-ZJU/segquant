"""
This module provides functionality for quantizing PyTorch models, including
linear layer replacement, calibration, and smoothing. It supports segment
linear layers and pattern detection for advanced quantization techniques.
"""

import fnmatch
import torch
from torch import nn
from tqdm import tqdm
from segquant.config import default_quantize_config
from segquant.layers.SegmentLinear import create_segment_linear
from segquant.pattern_detector import SegQuantPatternDetector


def _move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return tuple(_move_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    return batch


def _get_quantization_config(
    final_config,
    name,
    input_dtype=None,
    weight_dtype=None,
    opt=None,
    input_axis=None,
    weight_axis=None,
    alpha=None,
):
    input_dtype = input_dtype or final_config["default"].get("input_dtype")
    weight_dtype = weight_dtype or final_config["default"].get("weight_dtype")
    opt = opt or final_config["default"].get("opt")
    input_axis = input_axis or final_config["default"].get("input_axis")
    weight_axis = weight_axis or final_config["default"].get("weight_axis")
    alpha = alpha or final_config["default"].get("alpha")

    if name in final_config:
        input_dtype = final_config[name].get("input_dtype", input_dtype)
        weight_dtype = final_config[name].get("weight_dtype", weight_dtype)
        opt = final_config[name].get("opt", opt)
        input_axis = final_config[name].get("input_axis", input_axis)
        weight_axis = final_config[name].get("weight_axis", weight_axis)
        alpha = final_config[name].get("alpha", alpha)

    return input_dtype, weight_dtype, opt, input_axis, weight_axis, alpha


def _create_linear(
    layer,
    layer_name,
    seg_linear_config,
    final_config,
    input_dtype=None,
    weight_dtype=None,
    opt=None,
    dual_scale=False,
    input_axis=None,
    weight_axis=None,
    alpha=None,
):
    """
    Create a new segment linear layer based on the provided configuration.
    Args:
        layer: The original linear layer to be replaced.
        layer_name: The name of the layer.
        seg_linear_config: The configuration for the segment linear layer.
        final_config: The final quantization configuration.
        input_dtype: Optional; the input data type.
        weight_dtype: Optional; the weight data type.
        opt: Optional; the optimization type.
        dual_scale: Whether to use dual scale quantization.
        input_axis: Optional; the input axis for quantization.
        weight_axis: Optional; the weight axis for quantization.
        alpha: Optional; the scaling factor for quantization.
    Returns:
        A new segment linear layer with the specified configuration.
    """
    old_linear = layer
    device = old_linear.weight.device
    old_dtype = old_linear.weight.dtype
    has_bias = hasattr(old_linear, "bias") and old_linear.bias is not None

    (
        input_dtype,
        weight_dtype,
        opt,
        input_axis,
        weight_axis,
        alpha,
    ) = _get_quantization_config(
        final_config,
        layer_name,
        input_dtype,
        weight_dtype,
        opt,
        input_axis,
        weight_axis,
        alpha,
    )

    new_linear = create_segment_linear(
        input_dtype,
        weight_dtype,
        opt,
        old_linear.in_features,
        old_linear.out_features,
        bias=has_bias,
        seg_mode=seg_linear_config["seg_mode"],
        chunks=seg_linear_config.get(
            "chunks", len(seg_linear_config.get("chunksizes", []))
        ),
        chunksizes=seg_linear_config.get("chunksizes"),
        custom_weight_tensor=old_linear.weight,
        input_quant_args={"dual_scale": dual_scale, "axis": input_axis},
        weight_quant_args={"axis": weight_axis},
        alpha=alpha,
    )
    if has_bias:
        new_linear.linear.bias.data.copy_(old_linear.bias.data)

    new_linear = new_linear.to(device).to(old_dtype)
    return new_linear


def _replace_linears(model, to_replace_linears: dict):
    for layer_name, new_linear in to_replace_linears.items():
        parts = layer_name.split(".")
        module = model
        for part in parts[:-1]:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        setattr(module, parts[-1], new_linear)


def _get_all_linears(model: nn.Module):
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module
    return linears


def _smooth_linears(
    model: nn.Module,
    to_calib_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_calib_linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    if hasattr(to_calib_linears[n], "trace"):
                        to_calib_linears[n].trace(inp[0])

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Smooth Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for l in to_calib_linears.values():
        if hasattr(l, "smooth"):
            l.smooth()


def _calib_linears(
    model: nn.Module,
    to_calib_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_calib_linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    to_calib_linears[n].calibrate(inp[0])

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Calib Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for l in to_calib_linears.values():
        l.finish_calibrate()


def _filter_disabled(linears, config):
    disable_patterns = [
        k for k, v in config.items() if k != "default" and v.get("enable") is False
    ]
    keys_to_remove = set()
    for key in linears.keys():
        for pattern in disable_patterns:
            if fnmatch.fnmatch(key, pattern):
                keys_to_remove.add(key)
                break
    for key in keys_to_remove:
        del linears[key]


def quantize(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    config=None,
    verbose=False,
    example=None,
):
    """
    Quantize the model using the provided calibration data loader and configuration.
    Args:
        model: The PyTorch model to be quantized.
        calib_data_loader: DataLoader for calibration data.
        config: Optional; configuration for quantization.
        verbose: Whether to print verbose output.
        example: Optional; an example input for pattern detection.
    Returns:
        The quantized model.
    """
    device = next(model.parameters()).device
    final_config = config or default_quantize_config
    final_config["default"] = final_config.get(
        "default", default_quantize_config["default"]
    )

    if not final_config["default"]["enable"]:
        return model

    linears = _get_all_linears(model)
    _filter_disabled(linears, final_config)
    to_calib_linears = {}

    if verbose:
        print(f"get valid linear num [{len(linears)}]")

    dual_scale_linears = set()
    enable_seg = any(cfg.get("seglinear") for cfg in final_config.values())
    if enable_seg:
        example = example if example is not None else calib_data_loader.dataset[0]
        example = _move_to_device(example, device)

        seg_detector = SegQuantPatternDetector(
            model,
            example_inputs=example,
            search_patterns_lst=[
                p.value for p in final_config["default"]["search_patterns"]
            ],
        )
        seg_result = seg_detector.find_all_patterns()

        if "activation_to_linear" in seg_result:
            dual_scale_linears = set(seg_result["activation_to_linear"])
            del seg_result["activation_to_linear"]

        for l in seg_result.values():
            for seg_linear_config in l:
                name = seg_linear_config["linear_name"]
                if name in linears:
                    disabled = False
                    for pattern in final_config:
                        if fnmatch.fnmatch(name, pattern):
                            if not final_config[pattern].get("seglinear", True):
                                print(
                                    f"[INFO] Detect but disabled [{name}] "
                                    f"(matched pattern: {pattern})"
                                )
                                disabled = True
                            break
                    if disabled:
                        continue
                    to_calib_linears[name] = _create_linear(
                        linears[name],
                        name,
                        seg_linear_config,
                        final_config,
                        dual_scale=(name in dual_scale_linears),
                    )
                    print(f"[INFO] Detected [{name}]")
                    del linears[name]

    for name, linear in linears.items():
        to_calib_linears[name] = _create_linear(
            linear,
            name,
            {"chunks": 1, "seg_mode": "weight"},
            final_config,
            dual_scale=(name in dual_scale_linears),
        )

    if verbose:
        print("start smooth ...")
    _smooth_linears(model, to_calib_linears, calib_data_loader, device)
    if verbose:
        print("start calibrate ...")
    _calib_linears(model, to_calib_linears, calib_data_loader, device)
    if verbose:
        print("start replace ...")
    _replace_linears(model, to_calib_linears)

    if verbose:
        print(model)

    return model
