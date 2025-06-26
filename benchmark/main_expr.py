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
from benchmark import trace_pic
from benchmark.yaml_parser import parse_yaml
import os

from segquant.sample.sampler import QDiffusionSampler
from segquant.torch.calibrate_set import generate_calibrate_set, load_calibrate_set
from segquant.torch.quantization import quantize

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

def run_real_baseline(dataset_type, model_type, calib_config, max_num=None, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")

    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        os.makedirs(latent_root_dir, exist_ok=True)
        print(f"Created latent root directory: {latent_root_dir}")
    
    latents_path = os.path.join(latent_root_dir, 'latents.pt')
    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"Loaded latents with shape: {latents.shape}")
    else:
        print(f"Latents file {latents_path} does not exist. Generating new latents.")
        latents = get_latents(model_type, device="cuda:0")
        torch.save(latents, latents_path)
    
    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, 'real')
    if os.path.exists(exp_root_dir):
        print(f"Experiment directory {exp_root_dir} already exists. Skipping creation.")
    else:
        os.makedirs(exp_root_dir, exist_ok=True)
        print(f"Created experiment directory: {exp_root_dir}")

    model = get_full_model(model_type, device="cuda:0")

    ##### run inference
    print("Running inference...")
    pic_path = os.path.join(exp_root_dir, 'pic')
    dataset = get_dataset(dataset_type, dataset_root_dir)

    trace_pic(
        model,
        pic_path,
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_config["controlnet_conditioning_scale"],
        guidance_scale=calib_config["guidance_scale"],
        num_inference_steps=calib_config['max_timestep'],
    )
    del model
    print("Real completed")

def run_experiment(dataset_type, model_type, layer_type, exp_all_name, config, calib_config, max_num=None, per_layer_mode=False, affiner=None, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets', calibrate_root_dir='calibset_record'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print(f"Layer Type: {layer_type}")
    print(f"Running experiment: {exp_all_name}")
    print('--------------------------')
    print("Configuration:")
    print('--------------------------')
    print(f"Root Directory: {root_dir}")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print(f"Calibrate Root Directory: {calibrate_root_dir}")
    print(config)
    print('--------------------------')

    ### exp root and latents
    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        os.makedirs(latent_root_dir, exist_ok=True)
        print(f"Created latent root directory: {latent_root_dir}")
    
    latents_path = os.path.join(latent_root_dir, 'latents.pt')
    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"Loaded latents with shape: {latents.shape}")
    else:
        print(f"Latents file {latents_path} does not exist. Generating new latents.")
        latents = get_latents(model_type, device="cuda:0")
        torch.save(latents, latents_path)

    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)
    if os.path.exists(exp_root_dir):
        print(f"Experiment directory {exp_root_dir} already exists. Skipping creation.")
    else:
        os.makedirs(exp_root_dir, exist_ok=True)
        print(f"Created experiment directory: {exp_root_dir}")

    ##### find partial model
    print("Checking for partial model...")
    part_model_path = os.path.join(exp_root_dir, 'part_model.pt')
    if os.path.exists(part_model_path):
        print(f"Partial model file {part_model_path} already exists. Skipping creation.")
        quantized_model = torch.load(part_model_path, weights_only=False)
    else:
        print(f"Creating partial model file: {part_model_path}, need to be quantized.")
        part_model = get_part_model(model_type, layer_type, device="cuda:0")

        ### find calibrate data
        calibset = get_calibrate_data(
            dataset_type, model_type, layer_type, dataset_root_dir, calibrate_root_dir, calib_config
        )
        print(f"Calibrate set loaded with {len(calibset)} samples.")
        calib_loader = calibset.get_dataloader(batch_size=1)
        quantized_model = quantize(
            part_model, calib_loader, config, 
            per_layer_mode=per_layer_mode, verbose=True
        )
        torch.save(quantized_model, part_model_path)
        print(f"Partial model saved to {part_model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_full_model_with_quantized_part(model_type, layer_type, quantized_model, device="cuda:0")

    ##### run inference
    print("Running inference...")
    pic_path = os.path.join(exp_root_dir, 'pic')
    dataset = get_dataset(dataset_type, dataset_root_dir)

    trace_pic(
        model,
        pic_path,
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        steper=affiner,
        controlnet_conditioning_scale=calib_config["controlnet_conditioning_scale"],
        guidance_scale=calib_config["guidance_scale"],
        num_inference_steps=calib_config['max_timestep'],
    )
    del model
    print("Benchmark completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Benchmark for Diffusion Models')
    parser.add_argument('-d', '--dataset-type', type=str, default='COCO', choices=['COCO', 'MJHQ', 'DCI'], help='Type of the dataset to use')
    parser.add_argument('-m', '--model-type', type=str, default='sd3', choices=['flux', 'sd3', 'sdxl'], help='Type of the model to benchmark')
    parser.add_argument('-l', '--layer-type', type=str, default='dit', choices=['dit', 'controlnet', 'unet'], help='Type of the layer to benchmark')
    parser.add_argument('-q', '--quant-type', type=str, default='int8w8a8', help='Type of the quant to benchmark')
    parser.add_argument('-e', '--exp-name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('-c', '--config-dir', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('-C', '--calibrate-config', type=str, default='config/calibrate_config.json', help='Path to the calibration configuration file')
    parser.add_argument('-n', '--max-num', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-p', '--per-layer-mode', action='store_true', help='Run in per-layer quantization mode')
    parser.add_argument('-r', '--root-dir', type=str, default='../benchmark_results', help='Root directory for benchmark results')
    parser.add_argument('--dataset-root', type=str, default='../dataset/controlnet_datasets', help='Root directory for datasets')
    parser.add_argument('--calibrate-root', type=str, default='../calibset_record', help='Root directory for calibration sets')
    parser.add_argument('-R', '--run-real-baseline', action='store_true', help='Run the real baseline experiment without quantization')
    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_type = args.model_type
    layer_type = args.layer_type
    quant_type = args.quant_type
    exp_name = args.exp_name
    root_dir = args.root_dir
    max_num = args.max_num
    per_layer_mode = args.per_layer_mode
    dataset_root_dir = args.dataset_root
    calibrate_root_dir = args.calibrate_root

    calib_config_path = args.calibrate_config
    if not os.path.exists(calib_config_path):
        raise FileNotFoundError(f"Calibration configuration file {calib_config_path} does not exist.")
    with open(calib_config_path, 'r') as f:
        calib_config = json.load(f)
    
    if args.run_real_baseline:
        # Run the real baseline experiment without quantization
        run_real_baseline(
            dataset_type, model_type, calib_config, 
            max_num=max_num, root_dir=root_dir, 
            dataset_root_dir=dataset_root_dir
        )
    else:
        # Load configuration
        config_dir = args.config_dir
        config_path = os.path.join(config_dir, model_type, layer_type, f'{quant_type}.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
        from benchmark.yaml_parser import parse_yaml
        configs = parse_yaml(model_type, layer_type, quant_type, config_path)
        exp_all_name = f'{model_type}-{layer_type}-{quant_type}-{exp_name}'
        if exp_all_name in configs:
            run_experiment(
                dataset_type, model_type, layer_type, exp_all_name, 
                configs[exp_all_name], calib_config, max_num=max_num, 
                per_layer_mode=per_layer_mode, root_dir=root_dir, 
                dataset_root_dir=dataset_root_dir, calibrate_root_dir=calibrate_root_dir
            )
        else:
            print(f"Experiment {exp_all_name} not found in the configuration file.")