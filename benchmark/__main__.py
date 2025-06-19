import os
import torch
from backend.torch.modules.controlnet_sd3 import SD3ControlNetModel
from backend.torch.modules.transformer_sd3 import SD3Transformer2DModel
from benchmark import trace_pic
from segquant.config import Calibrate, DType, Optimum, SegPattern
from segquant.torch.affiner import load_affiner
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
from huggingface_hub import snapshot_download, try_to_load_from_cache
import argparse

calib_args = {
    "max_timestep": 50,
    "sample_size": 1,
    "timestep_per_sample": 50,
    "controlnet_conditioning_scale": 0,
    "guidance_scale": 7,
    "shuffle": False,
}


def check_model_exists(repo_id, local_path=None, model_type=None):
    """
    Check if model exists locally or in HuggingFace cache and validate structure.
    
    Args:
        repo_id (str): HuggingFace repository ID
        local_path (str, optional): Local path to check
        model_type (str, optional): Type of model for validation
    
    Returns:
        tuple: (exists, path) where exists is bool and path is str or None
    """
    # Check local path first
    if local_path and os.path.exists(local_path):
        print(f"Found valid local model at: {local_path}")
        return True, local_path
    
    # Check HuggingFace cache
    try:
        cached_path = try_to_load_from_cache(repo_id, filename="config.json")
        if cached_path is not None:
            # Get the parent directory (model directory)
            model_cache_dir = os.path.dirname(cached_path)
            print(f"Found valid cached model at: {model_cache_dir}")
            return True, model_cache_dir
    except Exception as e:
        print(f"Cache check failed for {repo_id}: {e}")
    
    print(f"Model {repo_id} not found locally or in cache")
    return False, None

def auto_download_model(repo_id, local_path=None, model_type=None):
    """
    Automatically download model from HuggingFace Hub with existence check.
    
    Args:
        repo_id (str): HuggingFace repository ID
        local_path (str, optional): Local path to check first
        model_type (str, optional): Type of model for validation
    
    Returns:
        str: Path to the model (local or downloaded)
    """
    # Check if model exists and we're not forcing download
    exists, existing_path = check_model_exists(repo_id, local_path, model_type)
    if exists:
        return existing_path
    
    try:        
        print(f"Downloading model from: {repo_id}")
        # Download to default cache directory
        cache_dir = snapshot_download(repo_id=repo_id)
        
        print(f"Model downloaded and validated at: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")
        # Try to find existing model as fallback
        exists, existing_path = check_model_exists(repo_id, local_path, model_type)
        if exists:
            print(f"Using existing model at: {existing_path}")
            return existing_path
        else:
            raise RuntimeError(f"Cannot load model {repo_id} - download failed and no existing model found")

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
        model_path = auto_download_model(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            "../stable-diffusion-3-medium-diffusers",
            model_type='sd3'
        )
        controlnet_path = auto_download_model(
            "InstantX/SD3-Controlnet-Canny", 
            "../SD3-Controlnet-Canny",
            model_type='controlnet'
        )
        model_real = StableDiffusion3ControlNetModel.from_repo(
            (model_path, controlnet_path),
            "cuda:0",
        )
    elif model_type == 'flux':
        model_path = auto_download_model(
            "black-forest-labs/FLUX.1-dev", 
            "../FLUX.1-dev",
            model_type='flux'
        )
        controlnet_path = auto_download_model(
            "InstantX/FLUX.1-dev-Controlnet-Canny", 
            "../FLUX.1-dev-Controlnet-Canny",
            model_type='controlnet'
        )
        model_real = FluxControlNetModel.from_repo(
            (model_path, controlnet_path),
            "cuda:0",
            enable_control=False,
        )
    elif model_type == 'sdxl':
        model_path = auto_download_model(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "../stable-diffusion-xl-base-1.0",
            model_type='sdxl'
        )
        model_real = StableDiffusionXLModel.from_repo(
            model_path, "cuda:0"
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

def get_quantized_model(model_type: str, quant_layer: str, config, dataset, calibargs: dict, latents=None, per_layer_mode=False):
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
    return quantize(model, calib_loader, config, per_layer_mode=per_layer_mode, verbose=True)

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
                "../stable-diffusion-xl-base-1.0", "cuda:0",
                unet=model_part
            )
        else:
            raise ValueError(f'[{part_layer} not found]')
    else:
        raise ValueError(f'[{model_type} not found]')
    
    return model

def quant_or_load(dataset, model_type: str, quant_layer: str, model_target_path, quant_config, latents, per_layer_mode):
    if not os.path.exists(model_target_path):
        print(f"[INFO] {model_target_path} not found, start quantizing...")
        print('quant config:')
        print(quant_config)
        quantized_model = get_quantized_model(
            model_type, quant_layer, quant_config, dataset, calib_args, latents, per_layer_mode
        )
        os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
        torch.save(quantized_model, model_target_path)
        print(f"[INFO] Model quantizing ok, saved to {model_target_path}")
    else:
        print(f"[INFO] {model_target_path} found, start loading...")
        quantized_model = torch.load(model_target_path, weights_only=False)
    return quantized_model

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

def run_any_module(quant_config, model_type, quant_layer, per_layer_mode=False, root_dir='tmp_test_linear'):
    with torch.no_grad():
        os.makedirs(root_dir, exist_ok=True)

        dataset = COCODataset(
            path="../dataset/controlnet_datasets/COCO-Caption2017-canny", cache_size=16
        )

        max_timestep = 50
        max_num = 1

        latents = get_randn_latents(model_type)
        model_quant_path = os.path.join(root_dir, f"model/{model_type}/{quant_layer}/model_quant.pt")
        model_part = quant_or_load(dataset, model_type, quant_layer, model_quant_path, quant_config, latents, per_layer_mode)
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

def run_flux():
    flux_config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            "real_quant": True,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 64,
                "search_alpha_config": {
                    "enable": False,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                },
                "verbose": True,
            },
            "calib": {
                "type": Calibrate.GPTQ,
                "cpu_storage": False,
                "verbose": False,
            },
            "input_quant": {
                "type": DType.INT4,
                # "axis": None,
                "axis": -1, # per-token, input shape (..., in)
                "dynamic": True,
            },
            "weight_quant": {
                "type": DType.INT4,
                # "axis": None,
                "axis": 1, # per-channel, weight shape (out, in)
            },
        },
    }
    model_type = 'flux'
    quant_layer = 'dit'
    per_layer_mode = True
    root_dir = 'flux_tmp_linear'
    run_any_module(flux_config, model_type, quant_layer, per_layer_mode, root_dir)

def run_sd3():
    sd3_config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 64,
                "search_alpha_config": {
                    "enable": True,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                },
                "verbose": True,
            },
            "calib": {
                "type": Calibrate.GPTQ,
                "cpu_storage": False,
                "verbose": False,
            },
            "input_quant": {
                "type": DType.INT8,
                # "axis": None,
                "axis": -1, # per-token, input shape (..., in)
                "dynamic": True,
            },
            "weight_quant": {
                "type": DType.INT4,
                # "axis": None,
                "axis": 1, # per-channel, weight shape (out, in)
            },
        },
    }
    model_type = 'sd3'
    quant_layer = 'dit'
    per_layer_mode = False
    root_dir = 'sd3_tmp_linear'
    run_any_module(sd3_config, model_type, quant_layer, per_layer_mode, root_dir)

def run_sdxl():
    sd3xl_config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 64,
                "search_alpha_config": {
                    "enable": True,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                },
                "verbose": True,
            },
            "calib": {
                "type": Calibrate.GPTQ,
                "cpu_storage": False,
                "verbose": False,
            },
            "input_quant": {
                "type": DType.FP16,
                "axis": None,
                # "axis": -1, # per-token, input shape (..., in)
                # "dynamic": True,
            },
            "weight_quant": {
                "type": DType.INT4,
                # "axis": None,
                "axis": 1, # per-channel, weight shape (out, in)
            },
        },
    }
    model_type = 'sdxl'
    quant_layer = 'unet'
    per_layer_mode = False
    root_dir = 'sdxl_tmp_linear'
    run_any_module(sd3xl_config, model_type, quant_layer, per_layer_mode, root_dir)

def main():
    parser = argparse.ArgumentParser(description='Benchmark script with automatic model downloading')
    parser.add_argument('--model-type', choices=['sd3', 'flux', 'sdxl'], default='sdxl',
                       help='Type of model to use')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Custom cache directory for downloaded models')
    
    args = parser.parse_args()
    
    # Set cache directory if specified
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
    
    model_type = args.model_type
    
    model_real = get_full_model(model_type)
    

if __name__ == "__main__":
    main()
