import torch
import torch.nn as nn

from segquant.config import DType, SegPattern, default_quantize_config
from segquant.layers.SegmentLinear import BaseSegmentLinear, create_segment_linear
from segquant.pattern_detector import SegQuantPatternDetector

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return tuple(move_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch

def replace_named_linear(model, layer_name, seg_linear_config, dtype: DType, dual_scale=False):
    parts = layer_name.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    old_linear = getattr(module, parts[-1])
    device = old_linear.weight.device
    old_dtype = old_linear.weight.dtype
    has_bias = hasattr(old_linear, "bias") and old_linear.bias is not None

    if 'chunksizes' in seg_linear_config:
        new_linear = create_segment_linear(dtype, old_linear.in_features, old_linear.out_features,
                                        bias=has_bias,
                                        seg_mode=seg_linear_config['seg_mode'],
                                        chunks=len(seg_linear_config['chunksizes']),
                                        chunksizes=seg_linear_config['chunksizes'],
                                        custom_weight_tensor=old_linear.weight,
                                        quant_args= {
                                            'dual_scale':dual_scale,
                                        })
    elif 'chunks' in seg_linear_config:
        new_linear = create_segment_linear(dtype, old_linear.in_features, old_linear.out_features,
                                        bias=has_bias,
                                        seg_mode=seg_linear_config['seg_mode'],
                                        chunks=seg_linear_config['chunks'],
                                        custom_weight_tensor=old_linear.weight,
                                        quant_args= {
                                            'dual_scale':dual_scale,
                                        })
    else:
        raise ValueError('replace_named_linear: chunk keyword not found')

    if has_bias:
        new_linear.linear.bias.data.copy_(old_linear.bias.data)
    
    new_linear = new_linear.to(device).to(old_dtype)
    setattr(module, parts[-1], new_linear)

def trace_all_linears(model: nn.Module, calib_data_loader: torch.utils.data.DataLoader, device):
    seg_linear_inputs = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            seg_linear_inputs[name] = []
            def get_hook(n):
                def hook_fn(mod, inp, out):
                    seg_linear_inputs[n].append(inp[0].detach().cpu())
                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))
    model.eval()
    with torch.no_grad():
        for batch in calib_data_loader:
            this_input_tuple = move_to_device(batch[0], device)
            if isinstance(this_input_tuple, dict):
                model(**this_input_tuple)
            else:
                model(*this_input_tuple)
    for h in hooks:
        h.remove()
    
    return seg_linear_inputs

def quantize(model: nn.Module, calib_data_loader: torch.utils.data.DataLoader, config=None, verbose=False):
    device = next(model.parameters()).device

    final_config = None
    if config is None:
        final_config = default_quantize_config
    else:
        final_config = config
        if 'default' not in config:
            final_config['default'] = default_quantize_config['default']

    ## trace 
    seg_linear_inputs = trace_all_linears(model, calib_data_loader, device)

    seg_names = set()

    dual_scale_linears = set()

    enable_seg = any(cfg.get("seglinear") is True for cfg in final_config.values())
    if enable_seg:
        example = calib_data_loader.dataset[0]
        example = move_to_device(example, device)
        seg_detector = SegQuantPatternDetector(model, 
                                               example_inputs=example, 
                                               search_patterns=[p.value for p in final_config['default']['search_patterns']])
        seg_result = seg_detector.find_all_patterns()

        if 'activation_to_linear' in seg_result:
            dual_scale_linears = set(seg_result['activation_to_linear'])
            del seg_result['activation_to_linear']

        for l in seg_result.values():
            for seg_linear_config in l:
                name = seg_linear_config['linear_name']
                dtype = final_config['default']['dtype']
                if name in final_config:
                    if 'dtype' in final_config[name]:
                        dtype = final_config[name]['dtype']
                replace_named_linear(model, name, seg_linear_config, dtype, dual_scale=(name in dual_scale_linears))
                seg_names.add(name)

    ## replace single linears
    for name in seg_linear_inputs.keys():
        if name not in seg_names:
            dtype = final_config['default']['dtype']
            if name in final_config:
                if 'dtype' in final_config[name]:
                    dtype = final_config[name]['dtype']
            
            single_config = {
                'chunks': 1,
                'seg_mode': 'weight',
            }
            replace_named_linear(model, name, single_config, dtype, dual_scale=(name in dual_scale_linears))

    ## calibration
    for name, module in model.named_modules():
        if isinstance(module, BaseSegmentLinear):
            if verbose:
                print(f"Calibrating {name} with {len(seg_linear_inputs[name])} samples")
            module.calibrate(move_to_device(seg_linear_inputs[name], device))
    
    if verbose:
        print(model)
    
    return model


if __name__ == '__main__':
    config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
        },
    }

    from dataset.coco.coco_dataset import COCODataset
    from sample.sampler import Q_DiffusionSampler
    from segquant.torch.calibrate_set import generate_calibrate_set
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from backend.torch.models.flux_controlnet import FluxControlNetModel
    from backend.torch.modules.controlnet_flux import FluxControlNetModel as FluxControlNet, FluxMultiControlNetModel as FluxMultiControlNet
    from segquant.torch.calibrate_set import BaseCalibSet
    from sample.sampler import model_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)
    # model = FluxControlNetModel.from_repo(('../FLUX.1-dev', '../FLUX.1-dev-Controlnet-Canny'), device)
    
    quant_layer = 'controlnet'
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)
    sampler = Q_DiffusionSampler()
    sample_dataloader = dataset.get_dataloader(batch_size=1)
    calibset = generate_calibrate_set(model, sampler, sample_dataloader, quant_layer, 
                                        max_timestep=30, 
                                        sample_size=1, 
                                        timestep_per_sample=30, 
                                        controlnet_conditioning_scale=0.7,
                                        guidance_scale=3.5)
    
    # calibset.dump('flux-controlnet.pt')
    # calibset = BaseCalibSet.from_file('flux-controlnet.pt')

    calib_loader = calibset.get_dataloader(batch_size=1)

    model.controlnet = quantize(model_map[quant_layer](model), calib_loader, config, True)

    
    ## test
    latents = torch.load('../latents.pt')
    for batch in dataset.get_dataloader(batch_size=1):
        prompt, image, control = batch[0]
        image = model.forward(
            prompt = prompt, 
            control_image=control, 
            controlnet_conditioning_scale=0.7,
            num_inference_steps=28,
            guidance_scale=3.5,
            latents=latents,
        )[0]
        image[0].save(f'pic.jpg')
        break