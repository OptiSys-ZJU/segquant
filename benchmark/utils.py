import argparse
import json
import torch
from backend.torch.models.flux_controlnet import FluxControlNetModel
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from backend.torch.models.stable_diffusion_xl import StableDiffusionXLModel
from backend.torch.modules.controlnet_sd3 import SD3ControlNetModel
from backend.torch.modules.transformer_flux import FluxTransformer2DModel
from backend.torch.modules.transformer_sd3 import SD3Transformer2DModel
from backend.torch.modules.controlnet_flux import FluxControlNetModel as ControlNetFlux
from backend.torch.modules.unet_2d_condition import UNet2DConditionModel
from backend.torch.utils import randn_tensor
import os

from segquant.sample.sampler import QDiffusionSampler
from segquant.torch.calibrate_set import generate_calibrate_set, load_calibrate_set

def get_full_model(model_type, device="cuda:0"):
    """Get full model for the specified type"""
    if model_type == 'sd3':
        return StableDiffusion3ControlNetModel.from_repo(
            ('../stable-diffusion-3-medium-diffusers', "../SD3-Controlnet-Canny"), device
        )
    elif model_type == 'flux':
        return FluxControlNetModel.from_repo(
            ('../FLUX.1-dev', "../FLUX.1-dev-Controlnet-Canny"), device, enable_control=False
        )
    elif model_type == 'sdxl':
        return StableDiffusionXLModel.from_repo('../stable-diffusion-xl-base-1.0', device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def get_part_model(model_type, layer_type, device="cuda:0"):
    """Get specific layer/component of the model"""
    if model_type == 'sd3':
        if layer_type == 'dit':
            return SD3Transformer2DModel.from_config(
                "../stable-diffusion-3-medium-diffusers/transformer/config.json",
                "../stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors",
            ).half().to(device)
        elif layer_type == 'controlnet':
            return SD3ControlNetModel.from_config(
                "../SD3-Controlnet-Canny/config.json",
                "../SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors",
            ).half().to(device)
    elif model_type == 'flux':
        if layer_type == 'dit':
            return FluxTransformer2DModel.from_config(
                "../FLUX.1-dev/transformer/config.json",
                "../FLUX.1-dev/transformer/diffusion_pytorch_model*.safetensors",
                "../FLUX.1-dev/transformer/diffusion_pytorch_model.safetensors.index.json",
            ).half().to(device)
        elif layer_type == 'controlnet':
            return ControlNetFlux.from_config(
                "../FLUX.1-dev-Controlnet-Canny/config.json",
                "../FLUX.1-dev-Controlnet-Canny/diffusion_pytorch_model.safetensors",
            ).half().to(device)
    elif model_type == 'sdxl':
        if layer_type == 'unet':
            return UNet2DConditionModel.from_config(
                "../stable-diffusion-xl-base-1.0/unet/config.json",
                "../stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors",
            ).half().to(device)
    
    raise ValueError(f'Unknown combination: model_type={model_type}, layer_type={layer_type}')

def get_full_model_with_quantized_part(model_type, layer_type, model_part, device="cuda:0"):
    """Get full model with quantized component"""
    if model_type == 'flux':
        if layer_type == 'dit':
            return FluxControlNetModel.from_repo(
                ("../FLUX.1-dev", "../FLUX.1-dev-Controlnet-Canny"), 
                device, enable_control=False, transformer=model_part
            )
        elif layer_type == 'controlnet':
            return FluxControlNetModel.from_repo(
                ("../FLUX.1-dev", "../FLUX.1-dev-Controlnet-Canny"), 
                device, enable_control=True, controlnet=model_part
            )
    elif model_type == 'sd3':
        if layer_type == 'dit':
            return StableDiffusion3ControlNetModel.from_repo(
                ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), 
                device, transformer=model_part
            )
        elif layer_type == 'controlnet':
            return StableDiffusion3ControlNetModel.from_repo(
                ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), 
                device, controlnet=model_part
            )
    elif model_type == 'sdxl':
        if layer_type == 'unet':
            return StableDiffusionXLModel.from_repo(
                "../stable-diffusion-xl-base-1.0", device, unet=model_part
            )
    
    raise ValueError(f'Unknown combination: model_type={model_type}, layer_type={layer_type}')

def get_latents(model_type, device="cuda:0"):
    if model_type == 'flux':
        latents = randn_tensor(
            (1, 4096, 64,), device=torch.device(device), dtype=torch.float16
        )
    elif model_type == 'sd3':
        latents = randn_tensor(
            (1, 16, 128, 128,), device=torch.device(device), dtype=torch.float16
        )
    elif model_type == 'sdxl':
        latents = randn_tensor(
            (1, 4, 128, 128,), device=torch.device(device), dtype=torch.float16
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return latents

def get_dataset(dataset_type, dataset_root_dir):
    if dataset_type == "COCO":
        from dataset.coco.coco_dataset import COCODataset
        dataset = COCODataset(os.path.join(dataset_root_dir, 'COCO-Caption2017-canny'), cache_size=16)
    elif dataset_type == "MJHQ":
        from dataset.mjhq.mjhq_dataset import MJHQDataset
        dataset = MJHQDataset(os.path.join(dataset_root_dir, 'MJHQ-30K-canny'), cache_size=16)
    elif dataset_type == "DCI":
        from dataset.dci.dci_dataset import DCIDataset
        dataset = DCIDataset(os.path.join(dataset_root_dir, 'DCI-canny'), cache_size=16)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return dataset

def get_dataset_prompt_metadata_file(dataset_type, dataset_root_dir):
    if dataset_type == "COCO":
        return os.path.join(dataset_root_dir, 'COCO-Caption2017-canny', 'metadata.json')
    elif dataset_type == "MJHQ":
        return os.path.join(dataset_root_dir, 'MJHQ-30K-canny', 'metadata.json')
    elif dataset_type == "DCI":
        return os.path.join(dataset_root_dir, 'DCI-canny', 'metadata.json')
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_calibrate_data(dataset_type, model_type, layer_type, dataset_root_dir, calibrate_root_dir, calib_args):
    calib_key = (
        f"maxT{calib_args['max_timestep']}_"
        f"sz{calib_args['sample_size']}_"
        f"tps{calib_args['timestep_per_sample']}_"
        f"cond{calib_args['controlnet_conditioning_scale']}_"
        f"gs{calib_args['guidance_scale']}_"
        f"{'shuffle' if calib_args['shuffle'] else 'noshuffle'}"
    )

    calibset_path = os.path.join(
        calibrate_root_dir,
        dataset_type,
        model_type, 
        layer_type, 
        calib_key
    )
    
    calibset = load_calibrate_set(calibset_path)
    if calibset is None:
        sampler = QDiffusionSampler()
        sample_dataloader = get_dataset(dataset_type, dataset_root_dir).get_dataloader(
            batch_size=1, shuffle=calib_args["shuffle"]
        )
        model_real = get_full_model(model_type, device="cuda:0")
        calibset = generate_calibrate_set(
            model_real,
            sampler,
            sample_dataloader,
            layer_type,
            calibset_path,
            max_timestep=calib_args["max_timestep"],
            sample_size=calib_args["sample_size"],
            timestep_per_sample=calib_args["timestep_per_sample"],
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
        )
        del model_real
    
    return calibset

def occupy_gpu_memory(device=0, reserve=512):
    torch.cuda.set_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    reserved_bytes = int(reserve * 1024 * 1024)

    block = torch.empty((total_mem - reserved_bytes) // 4, dtype=torch.float32, device=f'cuda:{device}')
    del block