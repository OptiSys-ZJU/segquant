import os
import torch
from torch import nn
from backend.torch.modules.controlnet_sd3 import SD3ControlNetModel
from backend.torch.modules.transformer_sd3 import SD3Transformer2DModel
from benchmark import trace_pic
from segquant.config import Calibrate, DType, Optimum, SegPattern
from segquant.torch.affiner import process_affiner
from segquant.sample.sampler import QDiffusionSampler
from segquant.torch.calibrate_set import generate_calibrate_set, load_calibrate_set
from segquant.torch.quantization import quantize
from dataset.coco.coco_dataset import COCODataset
from dataset.dci.dci_dataset import DCIDataset
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from backend.torch.models.flux_controlnet import FluxControlNetModel
from backend.torch.modules.controlnet_flux import FluxControlNetModel as ControlNetFlux
from backend.torch.modules.transformer_flux import FluxTransformer2DModel
from backend.torch.modules.unet_2d_condition import UNet2DConditionModel
from backend.torch.models.stable_diffusion_xl import StableDiffusionXLModel
from backend.torch.utils import randn_tensor

calib_args = {
    "max_timestep": 50,
    "sample_size": 1,
    "timestep_per_sample": 50,
    "controlnet_conditioning_scale": 0,
    "guidance_scale": 7,
    "shuffle": False,
}

quant_config = {
    "default": {
        "enable": True,
        "seglinear": True,
        "search_patterns": SegPattern.all(),
        "real_quant": False,
        "opt": {
            "type": Optimum.SMOOTH,
            "alpha": 0.5,
            "low_rank": 64,
            "cpu_storage": False,
            "search_alpha_config": {
                "enable": False,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "verbose": True,
        },
        "calib": {
            "type": Calibrate.AMAX,
            "cpu_storage": False,
            "verbose": False,
        },
        "input_quant": {
            "type": DType.INT8,
            "axis": None,
            # "axis": -1, # per-token, input shape (..., in)
            # "dynamic": True,
        },
        "weight_quant": {
            "type": DType.INT8,
            "axis": None,
            # "axis": 1, # per-channel, weight shape (out, in)
        },
    },
}

def get_randn_latents(model_type):
    if model_type == 'flux':
        latents = randn_tensor(
            (1, 4096, 64,), device=torch.device("cuda:0"), dtype=torch.float16
        )
    elif model_type == 'sd3':
        latents = randn_tensor(
            (1, 16, 128, 128,), device=torch.device("cuda:0"), dtype=torch.float16
        )
    elif model_type == 'sdxl':
        latents = randn_tensor(
            (1, 4, 128, 128,), device=torch.device("cuda:0"), dtype=torch.float16
        )
    else:
        raise ValueError(f'[{model_type} not found]')
    
    return latents

def get_full_model(model_type):
    if model_type == 'sd3':
        model_real = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
            "cuda:0",
        )
    elif model_type == 'flux':
        model_real = FluxControlNetModel.from_repo(
            ("../FLUX.1-dev", "../FLUX.1-dev-Controlnet-Canny"),
            "cuda:0",
            enable_control=False,
        )
    elif model_type == 'sdxl':
        model_real = StableDiffusionXLModel.from_repo(
            "../stable-diffusion-xl-base-1.0", "cuda:0"
        )
    else:
        raise ValueError(f'[{model_type} not found]')

    return model_real

def get_part_model(model_type, part_layer):
    if model_type == 'sd3':
        if part_layer == 'dit':
            model = SD3Transformer2DModel.from_config(
                "../stable-diffusion-3-medium-diffusers/transformer/config.json",
                "../stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors",
            ).half().to('cuda:0')
        elif part_layer == 'ctrlnet':
            model = SD3ControlNetModel.from_config(
                "../SD3-Controlnet-Canny/config.json",
                "../SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors",
            ).half().to('cuda:0')
        else:
            raise ValueError(f'[{part_layer} not found]')
    elif model_type == 'flux':
        if part_layer == 'dit':
            model = FluxTransformer2DModel.from_config(
                "../FLUX.1-dev/transformer/config.json",
                "../FLUX.1-dev/transformer/diffusion_pytorch_model*.safetensors",
                "../FLUX.1-dev/transformer/diffusion_pytorch_model.safetensors.index.json",
            ).half().to('cuda:0')
        elif part_layer == 'ctrlnet':
            model = ControlNetFlux.from_config(
                "../FLUX.1-dev-Controlnet-Canny/config.json",
                "../FLUX.1-dev-Controlnet-Canny/diffusion_pytorch_model.safetensors",
            ).half().to('cuda:0')
        else:
            raise ValueError(f'[{part_layer} not found]')
    elif model_type == 'sdxl':
        if part_layer == 'unet':
            model = UNet2DConditionModel.from_config(
                "../stable-diffusion-xl-base-1.0/unet/config.json",
                "../stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors",
            ).half().to("cuda:0")
        else:
            raise ValueError(f'[{part_layer} not found]')
    else:
        raise ValueError(f'[{model_type} not found]')

    return model

def get_quantized_model(model_type: str, quant_layer: str, config, dataset, calibargs: dict, latents=None):
    calib_key = (
        f"maxT{calibargs['max_timestep']}_"
        f"sz{calibargs['sample_size']}_"
        f"tps{calibargs['timestep_per_sample']}_"
        f"cond{calibargs['controlnet_conditioning_scale']}_"
        f"gs{calibargs['guidance_scale']}_"
        f"{'shuffle' if calibargs['shuffle'] else 'noshuffle'}"
    )

    calibset_path = os.path.join("calibset_record", model_type, quant_layer, calib_key)
    calibset = load_calibrate_set(calibset_path)
    if calibset is None:
        sampler = QDiffusionSampler()
        sample_dataloader = dataset.get_dataloader(
            batch_size=1, shuffle=calibargs["shuffle"]
        )
        model_real = get_full_model(model_type)
        calibset = generate_calibrate_set(
            model_real,
            sampler,
            sample_dataloader,
            quant_layer,
            calibset_path,
            max_timestep=calibargs["max_timestep"],
            sample_size=calibargs["sample_size"],
            timestep_per_sample=calibargs["timestep_per_sample"],
            controlnet_conditioning_scale=calibargs["controlnet_conditioning_scale"],
            guidance_scale=calibargs["guidance_scale"],
            latents=latents,
        )

        del model_real

    calib_loader = calibset.get_dataloader(batch_size=1)
    model = get_part_model(model_type, quant_layer)
    return quantize(model, calib_loader, config, True)

def get_full_model_by_quantized_part(model_type, part_layer, model_part):
    if model_type == 'flux':
        if part_layer == 'dit':
            model = FluxControlNetModel.from_repo(
                ("../FLUX.1-dev", "../FLUX.1-dev-Controlnet-Canny"), "cuda:0", enable_control=False,
                transformer=model_part
            )
        elif part_layer == 'ctrlnet':
            model = FluxControlNetModel.from_repo(
                ("../FLUX.1-dev", "../FLUX.1-dev-Controlnet-Canny"), "cuda:0", enable_control=True,
                controlnet=model_part
            )
        else:
            raise ValueError(f'[{part_layer} not found]')
    elif model_type == 'sd3':
        if part_layer == 'dit':
            model = StableDiffusion3ControlNetModel.from_repo(
                ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cuda:0",
                transformer=model_part
            )
        elif part_layer == 'ctrlnet':
            model = StableDiffusion3ControlNetModel.from_repo(
                ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cuda:0",
                controlnet=model_part
            )
        else:
            raise ValueError(f'[{part_layer} not found]')
    elif model_type == 'sdxl':
        if part_layer == 'unet':
            model = StableDiffusionXLModel.from_repo(
                ("../stable-diffusion-xl-base-1.0", "../SD3-Controlnet-Canny"), "cuda:0",
                unet=model_part
            )
        else:
            raise ValueError(f'[{part_layer} not found]')
    else:
        raise ValueError(f'[{model_type} not found]')
    
    return model

def run_real_module():
    with torch.no_grad():
        root_dir = "sdxl_test_linear"
        os.makedirs(root_dir, exist_ok=True)

        dataset = COCODataset(
            path="../dataset/controlnet_datasets/COCO-Caption2017-canny", cache_size=16
        )

        max_timestep = 50
        max_num = 1

        ### 1
        model_type = 'sdxl'

        latents = get_randn_latents(model_type)

        model = get_full_model(model_type)
        trace_pic(
            model,
            os.path.join(root_dir, "pics/quant"),
            dataset.get_dataloader(),
            latents,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        del model
        print("model_quant completed")

def run_any_module():
    with torch.no_grad():
        root_dir = "sdxl_test_linear"
        os.makedirs(root_dir, exist_ok=True)

        dataset = DCIDataset(
            path="../dataset/controlnet_datasets/COCO-Caption2017-canny", cache_size=16
        )

        max_timestep = 50
        max_num = 1

        def quant_or_load(model_type: str, quant_layer: str, model_target_path, quant_config, latents):
            if not os.path.exists(model_target_path):
                print(f"[INFO] {model_target_path} not found, start quantizing...")
                print('quant config:')
                print(quant_config)
                quantized_model = get_quantized_model(
                    model_type, quant_layer, quant_config, dataset, calib_args, latents
                )
                os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
                torch.save(quantized_model, model_target_path)
                print(f"[INFO] Model quantizing ok, saved to {model_target_path}")
            else:
                print(f"[INFO] {model_target_path} found, start loading...")
                quantized_model = torch.load(model_target_path, weights_only=False)
            return quantized_model

        model_type = 'sdxl'
        quant_layer = 'unet'

        latents = get_randn_latents(model_type)
        model_quant_path = os.path.join(root_dir, f"model/{model_type}/{quant_layer}/model_quant.pt")
        model_part = quant_or_load(model_type, quant_layer, model_quant_path, quant_config, latents)
        model = get_full_model_by_quantized_part(model_type, quant_layer, model_part)
        trace_pic(
            model,
            os.path.join(root_dir, "pics/quant"),
            dataset.get_dataloader(),
            latents,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        del model
        print("model_quant completed")

if __name__ == "__main__":
    run_any_module()
