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
from segquant.layers.SegmentLinear import create_segment_linear, SegmentLinear
from segquant.pattern_detector import SegQuantPatternDetector

class _EarlyStopForward(Exception):
    pass

def _move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return tuple(_move_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    return batch

def _get_all_linears(model: nn.Module, default, config):
    disable_patterns = [
        k for k, v in config.items() if k != "default" and v.get("enable") is False
    ]
    enable_patterns = [
        k for k, v in config.items() if k != "default" and v.get("enable") is True
    ]

    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            enable = default
            for pattern in disable_patterns:
                if fnmatch.fnmatch(name, pattern):
                    enable = False
                    break

            for pattern in enable_patterns:
                if fnmatch.fnmatch(name, pattern):
                    enable = True
                    break
            if enable:
                linears[name] = module
    return linears

def _create_linear(
    layer,
    layer_name,
    seg_linear_config,
    final_config,
    dual_scale=False,
):
    old_linear = layer
    device = old_linear.weight.device
    old_dtype = old_linear.weight.dtype
    has_bias = hasattr(old_linear, "bias") and old_linear.bias is not None

    this_config = final_config['default']
    for n in final_config.keys():
        if fnmatch.fnmatch(layer_name, n):
            this_config = final_config[n]
            break

    def get_config(t):
        input_quant_config_copy = this_config[t].copy()
        input_quant_type = input_quant_config_copy.pop('type')
        input_quant_args = input_quant_config_copy
        return input_quant_type, input_quant_args

    real_quant = this_config['real_quant']
    input_quant_type, input_quant_args = get_config('input_quant')
    weight_quant_type, weight_quant_args = get_config('weight_quant')
    opt_quant_type, opt_quant_args = get_config('opt')
    calib_quant_type, calib_quant_args = get_config('calib')

    new_linear = create_segment_linear(
        input_quant_type,
        weight_quant_type,
        opt_quant_type,
        calib_quant_type,
        old_linear.in_features,
        old_linear.out_features,
        opt_kwargs=opt_quant_args,
        calib_kwargs=calib_quant_args,
        input_quant_args={"real_quant": real_quant, "dual_scale": dual_scale, **input_quant_args},
        weight_quant_args={"real_quant": real_quant, **weight_quant_args},
        bias=has_bias,
        custom_bias_tensor=old_linear.bias.data if has_bias else None,
        seg_mode=seg_linear_config["seg_mode"],
        chunks=seg_linear_config.get(
            "chunks", len(seg_linear_config.get("chunksizes", []))
        ),
        chunksizes=seg_linear_config.get("chunksizes"),
        custom_weight_tensor=old_linear.weight.data,
    )

    new_linear = new_linear.to(device).to(old_dtype)
    del old_linear
    return new_linear

def _trace_linears(
    model: nn.Module,
    to_smooth_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_smooth_linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    if hasattr(to_smooth_linears[n], "trace"):
                        to_smooth_linears[n].trace(inp[0])

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Trace Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

def _smooth_linears(to_smooth_linears: dict):
    for l in tqdm(to_smooth_linears.values(), desc="[Smoothing linears]"):
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
    
    for l in tqdm(to_calib_linears.values(), desc="[Finishing Calibrate Linears]"):
        l.finish_calibrate()

def _search_linears(
    model: nn.Module,
    to_search_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    err_map = {k: [] for k in to_search_linears}
    nn_linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_search_linears:
            nn_linears[name] = module
            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    if n in to_search_linears:
                        real = to_search_linears[n].segment_forward(inp[0], weight=_mod.weight.data)
                        cur = to_search_linears[n].forward(inp[0], chunked=True)
                        diff_norms = [((r - c) ** 2).mean().item() for r, c in zip(real, cur)]
                        if not err_map[n]:
                            err_map[n] = diff_norms
                        else:
                            err_map[n] = [a + b for a, b in zip(err_map[n], diff_norms)]

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Search Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for n in list(to_search_linears):
        if to_search_linears[n].optimizer.search_step(err_map[n], origin_weight=nn_linears[n].weight.data):
            del to_search_linears[n]

def _trace_linear(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    linear_name: str,
    linear: nn.Linear,
    seglinear: SegmentLinear,
    device,
):
    def hook_trace_fn(_mod, inp, _out):
        seglinear.trace(inp[0])
        raise _EarlyStopForward()
    hook_trace = linear.register_forward_hook(hook_trace_fn)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc=f"[Trace Linear {linear_name}]"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            try:
                _ = (
                    model(*this_input_tuple)
                    if isinstance(this_input_tuple, tuple)
                    else model(**this_input_tuple)
                )
            except _EarlyStopForward:
                pass
    hook_trace.remove()

def _calibrate_linear(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    linear_name: str,
    linear: nn.Linear,
    seglinear: SegmentLinear,
    device,
):
    def hook_calibrate_fn(_mod, inp, _out):
        seglinear.calibrate(inp[0])
        raise _EarlyStopForward()
    hook_calib = linear.register_forward_hook(hook_calibrate_fn)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc=f"[Calib Linear {linear_name}]"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            try:
                _ = (
                    model(*this_input_tuple)
                    if isinstance(this_input_tuple, tuple)
                    else model(**this_input_tuple)
                )
            except _EarlyStopForward:
                pass
    hook_calib.remove()

def _search_linear(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    linear_name: str,
    linear: nn.Linear,
    seglinear: SegmentLinear,
    device,
):
    err = [None]
    def hook_search_fn(_mod, inp, _out):
        real = seglinear.segment_forward(inp[0], weight=_mod.weight.data)
        cur = seglinear.forward(inp[0], chunked=True)
        diff_norms = [((r.float() - c.float()) ** 2).mean().item() for r, c in zip(real, cur)]
        if err[0] is None:
            err[0] = diff_norms
        else:
            err[0] = [a + b for a, b in zip(err[0], diff_norms)]
        raise _EarlyStopForward()
    hook_search = linear.register_forward_hook(hook_search_fn)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc=f"[Search Linear {linear_name}]"
        ):
            this_input_tuple = _move_to_device(batch[0], device)
            try:
                _ = (
                    model(*this_input_tuple)
                    if isinstance(this_input_tuple, tuple)
                    else model(**this_input_tuple)
                )
            except _EarlyStopForward:
                pass
    hook_search.remove()

    if seglinear.optimizer.search_step(err[0], origin_weight=linear.weight.data):
        return True
    return False

def _replace_linears(model, to_replace_linears: dict):
    for layer_name, new_linear in to_replace_linears.items():
        parts = layer_name.split(".")
        module = model
        for part in parts[:-1]:
            if isinstance(module, nn.ModuleList) and part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        old_linear = getattr(module, parts[-1])
        setattr(module, parts[-1], new_linear)
        del old_linear

def quantize_linear(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    linear_name: str,
    seglinear: SegmentLinear,
    device,
):
    parts = linear_name.split(".")
    module = model
    for part in parts[:-1]:
        if isinstance(module, nn.ModuleList) and part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    module = getattr(module, parts[-1])

    assert isinstance(module, nn.Linear), f"Expected nn.Linear, but got {type(module)}"
    assert (module.in_features, module.out_features) == (
        seglinear.in_features,
        seglinear.out_features,
    ), f"Shape mismatch: Linear({module.in_features}, {module.out_features}) vs SegmentLinear({seglinear.in_features}, {seglinear.out_features})"

    need_smooth = seglinear.opt_type in ('smooth', 'svd')
    need_search = hasattr(seglinear.optimizer, 'search_alpha') and seglinear.optimizer.search_alpha

    if need_smooth:
        # trace
        _trace_linear(model, calib_data_loader, linear_name, module, seglinear, device)

    while need_search:
        if need_smooth:
            seglinear.smooth()
        # calibrate
        _calibrate_linear(model, calib_data_loader, linear_name, module, seglinear, device)
        seglinear.finish_calibrate()

        # search
        finished_search = _search_linear(model, calib_data_loader, linear_name, module, seglinear, device)
        if finished_search:
            break

    if need_smooth:
        seglinear.smooth()
    # calibrate
    _calibrate_linear(model, calib_data_loader, linear_name, module, seglinear, device)
    seglinear.finish_calibrate()

def quantize(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    config=None,
    per_layer_mode=False,
    verbose=False,
    example=None,
):
    """
    Quantize the model using the provided calibration data loader and configuration.
    Args:
        model (nn.Module): The PyTorch model to be quantized.
        calib_data_loader (dataloader): DataLoader for calibration data.
        config (dict): Optional; configuration for quantization.
        verbose (bool): Whether to print verbose output.
        example (any): Optional; an example input for pattern detection.
    Returns:
        The quantized model.
    """
    device = next(model.parameters()).device
    final_config = config or default_quantize_config
    final_config["default"] = final_config.get(
        "default", default_quantize_config["default"]
    )

    if all(not final_config[k]["enable"] for k in final_config):
        return model

    linears = _get_all_linears(model, final_config["default"]['enable'], config)
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
    del linears
    if per_layer_mode:
        for linear_name, seglinear in tqdm(to_calib_linears.items(), desc="[Quantize Linears]"):
            quantize_linear(model, calib_data_loader, linear_name, seglinear, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if verbose:
            print("start replace ...")
        _replace_linears(model, to_calib_linears)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # hook all linears at a time
        to_smooth_linears = {
            k: v for k, v in to_calib_linears.items()
            if v.opt_type in ('smooth', 'svd')
        }
        if to_smooth_linears:
            if verbose:
                print("start trace ...")
            _trace_linears(model, to_smooth_linears, calib_data_loader, device)

        to_search_linears = {
            k: v for k, v in to_calib_linears.items()
            if hasattr(v.optimizer, 'search_alpha') and v.optimizer.search_alpha
        }

        while to_search_linears:
            if to_search_linears:
                if verbose:
                    print("[search] start smooth ...")
                _smooth_linears(to_search_linears)
            if verbose:
                print("[search] start calibrate ...")
            _calib_linears(model, to_search_linears, calib_data_loader, device)
            _search_linears(model, to_search_linears, calib_data_loader, device)

        if to_smooth_linears:
            if verbose:
                print("start smooth ...")
            _smooth_linears(to_smooth_linears)
        if verbose:
            print("start calibrate ...")
        _calib_linears(model, to_calib_linears, calib_data_loader, device)

        if verbose:
            print("start replace ...")
        _replace_linears(model, to_calib_linears)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(model)

    return model
