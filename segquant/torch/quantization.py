"""
This module provides functionality for quantizing PyTorch models, including
linear layer replacement, calibration, and smoothing. It supports segment
linear layers and pattern detection for advanced quantization techniques.
"""

import fnmatch
import torch
import uuid
from torch import nn
from tqdm import tqdm
from segquant.config import default_quantize_config
from segquant.layers.SegmentLinear import create_segment_linear, SegmentLinear
from segquant.pattern_detector import SegQuantPatternDetector

def generate_run_id():
    return str(uuid.uuid4())

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
    device=None,
):
    ### device = target_model_device

    old_linear = layer
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
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return new_linear

def _trace_linears(
    model: nn.Module,
    to_smooth_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    origin_model_device,
    target_model_device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_smooth_linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    if hasattr(to_smooth_linears[n], "trace"):
                        to_smooth_linears[n].trace(_move_to_device(inp[0], target_model_device))

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Trace Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], origin_model_device)
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
    origin_model_device,
    target_model_device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_calib_linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    to_calib_linears[n].calibrate(_move_to_device(inp[0], target_model_device))

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Calib Linears] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], origin_model_device)
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
    origin_model_device,
    target_model_device,
    dump_search=False,
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
                        input_tensor = _move_to_device(inp[0], target_model_device)
                        real = to_search_linears[n].segment_forward(input_tensor, weight=_move_to_device(_mod.weight.data, target_model_device))
                        cur = to_search_linears[n].forward(input_tensor, chunked=True)
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
            this_input_tuple = _move_to_device(batch[0], origin_model_device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for n in list(to_search_linears):
        origin_weight = _move_to_device(nn_linears[n].weight.data, target_model_device)
        if to_search_linears[n].optimizer.search_step(err_map[n], origin_weight=origin_weight):
            del to_search_linears[n]

    if dump_search:
        alpha_map = {} 
        for k, v in to_search_linears.items():
            if hasattr(v.optimizer, 'alpha'):
                alpha_map[k] = {
                    'candidate_alphas': v.optimizer.candidate_alphas,
                    'alpha': v.optimizer.alpha,
                    'opt_err': v.optimizer.opt_err,
                    'opt_alpha': v.optimizer.opt_alpha,
                }
                print(f"[Search Linears] {k} alpha: {v.optimizer.alpha}, opt_err: {v.optimizer.opt_err}, opt_alpha: {v.optimizer.opt_alpha}")
        
        filename = f"search_dump_{generate_run_id()}.pt"
        torch.save(alpha_map, filename)
        print(f"[Search Linears] Dumped search results to {filename}")

def _replace_linears(model, to_replace_linears: dict, device):
    for layer_name, new_linear in to_replace_linears.items():
        parts = layer_name.split(".")
        module = model
        for part in parts[:-1]:
            if isinstance(module, nn.ModuleList) and part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        new_linear = new_linear.to(device)
        old_linear = getattr(module, parts[-1])
        setattr(module, parts[-1], new_linear)
        del old_linear

def quantize(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    config=None,
    tmp_device=None,
    verbose=False,
    example=None,
    dump_search=False,
    search_recovery_file=None,
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
    origin_model_device = next(model.parameters()).device
    target_model_device = origin_model_device
    if tmp_device is not None:
        target_model_device = tmp_device
        print(f"[INFO] Quantizing model Process will be in Target[{target_model_device}], Origin(Final)[{origin_model_device}]")
    
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
        example = _move_to_device(example, origin_model_device)

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
                        device=target_model_device,
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
            device=target_model_device,
        )
    del linears

    # hook all linears at a time
    to_smooth_linears = {
        k: v for k, v in to_calib_linears.items()
        if v.opt_type in ('smooth', 'svd')
    }

    to_search_linears = {
        k: v for k, v in to_calib_linears.items()
        if hasattr(v.optimizer, 'search_alpha') and v.optimizer.search_alpha
    }

    # search_recovery_file
    if search_recovery_file is not None:
        alpha_map = torch.load(search_recovery_file, weights_only=False)
        for k, v in alpha_map.items():
            if k in to_search_linears:
                to_search_linears[k].optimizer.candidate_alphas = v['candidate_alphas']
                to_search_linears[k].optimizer.alpha = v['alpha']
                to_search_linears[k].optimizer.opt_err = v['opt_err']
                to_search_linears[k].optimizer.opt_alpha = v['opt_alpha']
                print(f"[Search Recovery] {k} alpha: {v['alpha']}, opt_err: {v['opt_err']}, opt_alpha: {v['opt_alpha']}")
            else:
                print("[Warning] Search recovery file contains keys not in to_search_linears:", k)

    if to_smooth_linears:
        if verbose:
            print("start trace ...")
        _trace_linears(model, to_smooth_linears, calib_data_loader, origin_model_device, target_model_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    while to_search_linears:
        if to_search_linears:
            if verbose:
                print("[search] start smooth ...")
            _smooth_linears(to_search_linears)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if verbose:
            print("[search] start calibrate ...")
        _calib_linears(model, to_search_linears, calib_data_loader, origin_model_device, target_model_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _search_linears(model, to_search_linears, calib_data_loader, origin_model_device, target_model_device, dump_search=dump_search)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if to_smooth_linears:
        if verbose:
            print("start smooth ...")
        _smooth_linears(to_smooth_linears)
    if verbose:
        print("start calibrate ...")
    _calib_linears(model, to_calib_linears, calib_data_loader, origin_model_device, target_model_device)

    if verbose:
        print("start replace ...")
    _replace_linears(model, to_calib_linears, origin_model_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(model)

    return model
