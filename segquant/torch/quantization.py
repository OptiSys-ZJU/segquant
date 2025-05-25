import torch
import torch.nn as nn
import fnmatch

from tqdm import tqdm
import fnmatch

from segquant.config import DType, Optimum, SegPattern, default_quantize_config
from segquant.layers.SegmentLinear import create_segment_linear
from segquant.pattern_detector import SegQuantPatternDetector


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return tuple(move_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def get_quantization_config(
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


def create_linear(
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
    ) = get_quantization_config(
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


def replace_linears(model, to_replace_linears: dict):
    for layer_name, new_linear in to_replace_linears.items():
        parts = layer_name.split(".")
        module = model
        for part in parts[:-1]:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        setattr(module, parts[-1], new_linear)


def get_all_linears(model: nn.Module):
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module
    return linears


def smooth_linears(
    model: nn.Module,
    to_calib_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_calib_linears:

            def get_hook(n):
                def hook_fn(mod, inp, out, n=n):
                    if hasattr(to_calib_linears[n], "trace"):
                        to_calib_linears[n].trace(inp[0])

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Smooth Linears] Running model on calibration data"
        ):
            this_input_tuple = move_to_device(batch[0], device)
            model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for l in to_calib_linears.values():
        if hasattr(l, "smooth"):
            l.smooth()


def calib_linears(
    model: nn.Module,
    to_calib_linears: dict,
    calib_data_loader: torch.utils.data.DataLoader,
    device,
):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in to_calib_linears:

            def get_hook(n):
                def hook_fn(mod, inp, out, n=n):
                    to_calib_linears[n].calibrate(inp[0])

                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            calib_data_loader, desc="[Calib Linears] Running model on calibration data"
        ):
            this_input_tuple = move_to_device(batch[0], device)
            model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    for l in to_calib_linears.values():
        l.finish_calibrate()


def filter_disabled(linears, config):
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
    device = next(model.parameters()).device
    final_config = config or default_quantize_config
    final_config["default"] = final_config.get(
        "default", default_quantize_config["default"]
    )

    if not final_config["default"]["enable"]:
        return model

    linears = get_all_linears(model)
    filter_disabled(linears, final_config)
    to_calib_linears = {}

    if verbose:
        print(f"get valid linear num [{len(linears)}]")

    dual_scale_linears = set()
    enable_seg = any(cfg.get("seglinear") for cfg in final_config.values())
    if enable_seg:
        example = example if example is not None else calib_data_loader.dataset[0]
        example = move_to_device(example, device)

        seg_detector = SegQuantPatternDetector(
            model,
            example_inputs=example,
            search_patterns=[
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
                                    f"[INFO] Detect but disabled [{name}] (matched pattern: {pattern})"
                                )
                                disabled = True
                            break
                    if disabled:
                        continue
                    to_calib_linears[name] = create_linear(
                        linears[name],
                        name,
                        seg_linear_config,
                        final_config,
                        dual_scale=(name in dual_scale_linears),
                    )
                    print(f"[INFO] Detected [{name}]")
                    del linears[name]

    for name in linears:
        to_calib_linears[name] = create_linear(
            linears[name],
            name,
            {"chunks": 1, "seg_mode": "weight"},
            final_config,
            dual_scale=(name in dual_scale_linears),
        )

    if verbose:
        print("start smooth ...")
    smooth_linears(model, to_calib_linears, calib_data_loader, device)
    if verbose:
        print("start calibrate ...")
    calib_linears(model, to_calib_linears, calib_data_loader, device)
    if verbose:
        print("start replace ...")
    replace_linears(model, to_calib_linears)

    if verbose:
        print(model)

    return model


if __name__ == "__main__":
    config = {
        "default": {
            "enable": True,
            "input_dtype": DType.INT8,
            "weight_dtype": DType.INT8,
            "opt": Optimum.SMOOTH,
            "seglinear": True,
            "search_patterns": SegPattern.all(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 1.0,
        },
    }

    from dataset.coco.coco_dataset import COCODataset
    from segquant.sample.sampler import Q_DiffusionSampler
    from segquant.torch.calibrate_set import generate_calibrate_set
    from backend.torch.models.stable_diffusion_3_controlnet import (
        StableDiffusion3ControlNetModel,
    )
    from backend.torch.models.flux_controlnet import FluxControlNetModel
    from sample.sampler import model_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), device
    )
    # model = FluxControlNetModel.from_repo(('../FLUX.1-dev', '../FLUX.1-dev-Controlnet-Canny'), device)

    quant_layer = "controlnet"

    dataset = COCODataset(
        path="../dataset/controlnet_datasets/controlnet_canny_dataset", cache_size=16
    )
    sampler = Q_DiffusionSampler()
    sample_dataloader = dataset.get_dataloader(batch_size=1)
    calibset = generate_calibrate_set(
        model,
        sampler,
        sample_dataloader,
        quant_layer,
        max_timestep=30,
        sample_size=1,
        timestep_per_sample=30,
        controlnet_conditioning_scale=0.7,
        guidance_scale=3.5,
    )

    calib_loader = calibset.get_dataloader(batch_size=1)

    model.controlnet = quantize(
        model_map[quant_layer](model), calib_loader, config, verbose=True
    )

    ## test
    # latents = torch.load('../latents.pt')
    for batch in dataset.get_dataloader(batch_size=1, shuffle=True):
        prompt, image, control = batch[0]
        print(prompt)
        image = model.forward(
            prompt=prompt,
            control_image=control,
            controlnet_conditioning_scale=0.7,
            num_inference_steps=28,
            guidance_scale=3.5,
            # latents=latents,
        )[0]
        image[0].save(f"pic.jpg")
        break
