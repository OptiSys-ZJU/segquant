import os
import torch
import argparse
import sys
sys.path.append('../')

from benchmark import trace_pic
from benchmark.config import (
    BenchmarkConfig, CalibrationConfig, ModelConfigs, AffineConfig,
    MODEL_TYPE_CHOICES, LAYER_TYPE_CHOICES, DATASET_TYPE_CHOICES, QUANT_CONFIG_CHOICES
)
from segquant.torch.affiner import load_affiner
from segquant.sample.sampler import QDiffusionSampler
from segquant.torch.calibrate_set import generate_calibrate_set, load_calibrate_set
from segquant.torch.quantization import quantize
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from backend.torch.models.flux_controlnet import FluxControlNetModel
from backend.torch.models.stable_diffusion_xl import StableDiffusionXLModel
from backend.torch.modules.controlnet_sd3 import SD3ControlNetModel
from backend.torch.modules.transformer_sd3 import SD3Transformer2DModel
from backend.torch.modules.controlnet_flux import FluxControlNetModel as ControlNetFlux
from backend.torch.modules.transformer_flux import FluxTransformer2DModel
from backend.torch.modules.unet_2d_condition import UNet2DConditionModel
from huggingface_hub import snapshot_download, try_to_load_from_cache


class ModelManager:
    """Handles model loading and management"""
    
    @staticmethod
    def check_model_exists(repo_id, local_path=None):
        """Check if model exists locally or in HuggingFace cache"""
        if local_path and os.path.exists(local_path):
            print(f"Found valid local model at: {local_path}")
            return True, local_path
        
        try:
            cached_path = try_to_load_from_cache(repo_id, filename="config.json")
            if cached_path is not None:
                model_cache_dir = os.path.dirname(cached_path)
                print(f"Found valid cached model at: {model_cache_dir}")
                return True, model_cache_dir
        except Exception as e:
            print(f"Cache check failed for {repo_id}: {e}")
        
        print(f"Model {repo_id} not found locally or in cache")
        return False, None

    @staticmethod
    def auto_download_model(repo_id, local_path=None):
        """Automatically download model with existence check"""
        exists, existing_path = ModelManager.check_model_exists(repo_id, local_path)
        if exists:
            return existing_path
        
        try:        
            print(f"Downloading model from: {repo_id}")
            cache_dir = snapshot_download(repo_id=repo_id)
            print(f"Model downloaded at: {cache_dir}")
            return cache_dir
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")
            exists, existing_path = ModelManager.check_model_exists(repo_id, local_path)
            if exists:
                print(f"Using existing model at: {existing_path}")
                return existing_path
            else:
                raise RuntimeError(f"Cannot load model {repo_id} - download failed and no existing model found")

    @staticmethod
    def get_full_model(model_type, device="cuda:0"):
        """Get full model for the specified type"""
        if model_type == 'sd3':
            model_path = ModelManager.auto_download_model(
                "stabilityai/stable-diffusion-3-medium-diffusers", 
                "../stable-diffusion-3-medium-diffusers"
            )
            controlnet_path = ModelManager.auto_download_model(
                "InstantX/SD3-Controlnet-Canny", 
                "../SD3-Controlnet-Canny"
            )
            return StableDiffusion3ControlNetModel.from_repo(
                (model_path, controlnet_path), device
            )
        elif model_type == 'flux':
            model_path = ModelManager.auto_download_model(
                "black-forest-labs/FLUX.1-dev", 
                "../FLUX.1-dev"
            )
            controlnet_path = ModelManager.auto_download_model(
                "InstantX/FLUX.1-dev-Controlnet-Canny", 
                "../FLUX.1-dev-Controlnet-Canny"
            )
            return FluxControlNetModel.from_repo(
                (model_path, controlnet_path), device, enable_control=False
            )
        elif model_type == 'sdxl':
            model_path = ModelManager.auto_download_model(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                "../stable-diffusion-xl-base-1.0"
            )
            return StableDiffusionXLModel.from_repo(model_path, device)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    @staticmethod
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

    @staticmethod
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


class QuantizationManager:
    """Handles model quantization and calibration"""
    
    @staticmethod
    def quant_or_load(benchmark_config, model_target_path, quant_config, calib_args):
        """Quantize or load quantized model"""
        if not os.path.exists(model_target_path):
            print(f"[INFO] {model_target_path} not found, start quantizing...")
            print('Quantization config:')
            print(quant_config)
            
            quantized_model = QuantizationManager.get_quantized_model(
                benchmark_config, quant_config, calib_args
            )
            os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
            torch.save(quantized_model, model_target_path)
            print(f"[INFO] Model quantizing ok, saved to {model_target_path}")
        else:
            print(f"[INFO] {model_target_path} found, start loading...")
            quantized_model = torch.load(model_target_path, weights_only=False)
        
        return quantized_model

    @staticmethod
    def get_quantized_model(benchmark_config, quant_config, calib_args):
        """Generate quantized model"""
        calib_key = (
            f"maxT{calib_args['max_timestep']}_"
            f"sz{calib_args['sample_size']}_"
            f"tps{calib_args['timestep_per_sample']}_"
            f"cond{calib_args['controlnet_conditioning_scale']}_"
            f"gs{calib_args['guidance_scale']}_"
            f"{'shuffle' if calib_args['shuffle'] else 'noshuffle'}"
        )

        calibset_path = os.path.join(
            "../segquant/calibset_record", 
            benchmark_config.model_type, 
            benchmark_config.layer_type, 
            calib_key
        )
        
        calibset = load_calibrate_set(calibset_path)
        if calibset is None:
            sampler = QDiffusionSampler()
            sample_dataloader = benchmark_config.dataset.get_dataloader(
                batch_size=1, shuffle=calib_args["shuffle"]
            )
            model_real = ModelManager.get_full_model(
                benchmark_config.model_type, f"cuda:{benchmark_config.gpu_id}"
            )
            calibset = generate_calibrate_set(
                model_real,
                sampler,
                sample_dataloader,
                benchmark_config.layer_type,
                calibset_path,
                max_timestep=calib_args["max_timestep"],
                sample_size=calib_args["sample_size"],
                timestep_per_sample=calib_args["timestep_per_sample"],
                controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                guidance_scale=calib_args["guidance_scale"],
                latents=benchmark_config.latents,
            )
            del model_real

        calib_loader = calibset.get_dataloader(batch_size=1)
        model = ModelManager.get_part_model(
            benchmark_config.model_type, 
            benchmark_config.layer_type, 
            f"cuda:{benchmark_config.gpu_id}"
        )
        return quantize(
            model, calib_loader, quant_config, 
            per_layer_mode=benchmark_config.per_layer_mode, verbose=True
        )


def run_benchmark(benchmark_config):
    """Main benchmark execution function"""
    with torch.no_grad():
        print(f"[INFO] Benchmark configuration: {benchmark_config}")
        
        # Create directories
        os.makedirs(benchmark_config.result_dir, exist_ok=True)
        os.makedirs(os.path.dirname(benchmark_config.get_model_quant_path()), exist_ok=True)
        os.makedirs(benchmark_config.get_pic_store_path(), exist_ok=True)
        
        # Prepare calibration arguments
        calib_args = CalibrationConfig.to_dict()
        
        # Generate real pics if requested
        if benchmark_config.generate_real_pics:
            model = ModelManager.get_full_model(
                benchmark_config.model_type, f"cuda:{benchmark_config.gpu_id}"
            )
            trace_pic(
                model,
                benchmark_config.get_pic_store_path(),
                benchmark_config.dataset.get_dataloader(),
                benchmark_config.latents,
                max_num=benchmark_config.benchmark_size,
                controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                guidance_scale=calib_args["guidance_scale"],
                num_inference_steps=benchmark_config.max_timestep,
            )
            del model
            print("Real model pics generated")
            return

        # Quantization workflow (if quant_config specified) or baseline
        if benchmark_config.quant_config:
            print(f"Running quantization with {benchmark_config.quant_config}")
            quant_config = benchmark_config.model_quant_config
        else:
            print("Running baseline test (using original model config)")
            quant_config = ModelConfigs.get_config(benchmark_config.model_type)
        
        model_quant_path = benchmark_config.get_model_quant_path()
        
        model_part = QuantizationManager.quant_or_load(
            benchmark_config, model_quant_path, quant_config, calib_args
        )
        
        model = ModelManager.get_full_model_with_quantized_part(
            benchmark_config.model_type, 
            benchmark_config.layer_type, 
            model_part, 
            f"cuda:{benchmark_config.gpu_id}"
        )
        
        # Handle affine if enabled
        if benchmark_config.enable_affine:
            model_real = ModelManager.get_full_model(benchmark_config.model_type, "cpu")
            print("[INFO] Learning affine transformation...")
            affiner = load_affiner(
                AffineConfig.to_dict(), 
                benchmark_config.dataset, 
                model_real, 
                model, 
                latents=benchmark_config.latents, 
                shuffle=True
            )
            print("[INFO] Affine learning completed")
            del model_real
        else:
            affiner = None
        
        # Generate quantized pics
        trace_pic(
            model,
            benchmark_config.get_pic_store_path(),
            benchmark_config.dataset.get_dataloader(),
            benchmark_config.latents,
            max_num=benchmark_config.benchmark_size,
            steper=affiner,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=benchmark_config.max_timestep,
        )
        del model
        print("Benchmark completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Benchmark for Diffusion Models')
    
    parser.add_argument(
        '-m', '--model-type', 
        type=str, 
        choices=MODEL_TYPE_CHOICES,
        default='sdxl',
        help='Model type to benchmark'
    )
    
    parser.add_argument(
        '-l', '--layer-type',
        type=str,
        help='Layer type to quantize (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '-d', '--dataset-type',
        type=str, 
        choices=DATASET_TYPE_CHOICES,
        default='COCO',
        help='Dataset type for calibration and evaluation'
    )
    
    parser.add_argument(
        '-r', '--real',
        action='store_true',
        help='Generate real pictures without quantization'
    )
    
    parser.add_argument(
        '-n', '--benchmark-size',
        type=int,
        default=1,
        help='Number of images to generate for benchmark'
    )
    
    parser.add_argument(
        '--per-layer-mode',
        action='store_true',
        help='Enable per-layer quantization mode'
    )
    
    parser.add_argument(
        '--enable-affine',
        action='store_true',
        help='Enable affine transformation learning'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Custom cache directory for models'
    )
    
    # Quantization configuration 
    parser.add_argument(
        '-q', '--quant-config', 
        type=str,
        choices=QUANT_CONFIG_CHOICES,
        help='Weight/Activation bit configuration (e.g., W4A4, W8A8). If not specified, runs baseline test'
    )
    
    return parser.parse_args()


def main_expr():
    """Main entry point"""
    args = parse_args()
    
    # Set cache directory if specified
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
    
    # Validate layer type
    if args.layer_type and args.layer_type not in LAYER_TYPE_CHOICES[args.model_type]:
        print(f"Error: layer_type '{args.layer_type}' not valid for model_type '{args.model_type}'")
        print(f"Valid choices: {LAYER_TYPE_CHOICES[args.model_type]}")
        return
    
    # Auto-detect layer type if not specified
    if not args.layer_type:
        defaults = {"sd3": "dit", "flux": "dit", "sdxl": "unet"}
        args.layer_type = defaults.get(args.model_type, "dit")
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        model_type=args.model_type,
        layer_type=args.layer_type,
        dataset_type=args.dataset_type,
        num_images=args.benchmark_size,
        generate_real_pics=args.real,
        enable_affine=args.enable_affine,
        per_layer_mode=args.per_layer_mode,
        quant_config=args.quant_config
    )
    
    
    # Run benchmark
    run_benchmark(benchmark_config)


if __name__ == "__main__":
    main_expr()
