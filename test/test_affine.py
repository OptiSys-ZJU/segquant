import os
import argparse
import torch
from torch import nn
from benchmark import trace_pic
from benchmark.config import BenchmarkConfig,CalibrationConfig,AffineConfig,QuantizationConfigs,QUANT_METHOD_CHOICES,DATASET_TYPE_CHOICES
from segquant.torch.affiner import process_affiner
from segquant.sample.sampler import QDiffusionSampler, model_map
from segquant.torch.calibrate_set import generate_calibrate_set
from segquant.torch.quantization import quantize

from backend.torch.models.stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetModel,
)


def quant_or_load(model_target_path, target_config, dataset, calib_args, gpu_id):
    # Determine quant_layer from config
    quant_layer = "dit" if target_config["default"]["dit"] else "controlnet"
    
    # if model_target_path not exist, quantize model
    if not os.path.exists(model_target_path):
        print(f"[INFO] {model_target_path} not found, start quantizing...")

        model = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
            f"cuda:{gpu_id}",
        )
        # decide to quant, transfer to cpu for saving
        target_model = quant_model(
            model, quant_layer, target_config, dataset, calib_args
        ).to("cpu")
        # save model - save the appropriate component based on quant_layer
        os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
        if quant_layer == "dit":
            torch.save(target_model.transformer, model_target_path)
        else:
            torch.save(target_model.controlnet, model_target_path)
        print(f"[INFO] Model quantizing ok, saved to {model_target_path}")
    # if model_target_path exist, load model
    else:
        print(f"[INFO] {model_target_path} found, start loading...")
        target_model = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
            f"cuda:{gpu_id}",  # Load base model to target GPU, not CPU
        )
        # Load the appropriate component based on quant_layer
        if quant_layer == "dit":
            target_model.transformer = torch.load(model_target_path, weights_only=False, map_location=f"cuda:{gpu_id}")
        else:
            target_model.controlnet = torch.load(model_target_path, weights_only=False, map_location=f"cuda:{gpu_id}")

    return target_model

def quant_model(
    model_real: nn.Module, quant_layer: str, config, dataset, calibargs: dict
) -> nn.Module:
    """
    quantize model
    Args:
        model_real: real model
        quant_layer: quant layer
        config: config
        dataset: dataset
        calibargs: calib args
    Returns:
        quantized model
    """
    calib_key = (
        f"maxT{calibargs['max_timestep']}_"
        f"sz{calibargs['sample_size']}_"
        f"tps{calibargs['timestep_per_sample']}_"
        f"cond{calibargs['controlnet_conditioning_scale']}_"
        f"gs{calibargs['guidance_scale']}_"
        f"{'shuffle' if calibargs['shuffle'] else 'noshuffle'}"
    )
    calibset_path = os.path.join("../segquant/calibset_record", quant_layer, calib_key)
    sampler = QDiffusionSampler()
    sample_dataloader = dataset.get_dataloader(
        batch_size=1, shuffle=calibargs["shuffle"]
    )
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
    )

    calib_loader = calibset.get_dataloader(batch_size=1)
    if quant_layer == "dit":
        model_real.transformer = quantize(
            model_map[quant_layer](model_real), calib_loader, config, True
        )
    else:
        model_real.controlnet = quantize(
            model_map[quant_layer](model_real), calib_loader, config, True
        )
    return model_real

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    # 量化方法参数
    parser.add_argument(
        "-q", "--quant_method", 
        type=str,
        choices=QUANT_METHOD_CHOICES,  # 根据你的QuantMethod调整
        default="int8smooth",
        help="Quantization method"
    )
    # 数据集类型参数
    parser.add_argument(
        "-d", "--dataset_type",
        type=str, 
        choices=DATASET_TYPE_CHOICES,  # 根据你的BenchmarkType调整
        default="COCO",
        help="Dataset type"
    )
    
    # 布尔值参数
    parser.add_argument(
        "-r", "--real",
        action="store_true",  # 如果指定则为True
        help="Generate real pictures without quantization"
    )

    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    benchmark = BenchmarkConfig(
        quant_method=args.quant_method,
        dataset_type=args.dataset_type,
        generate_real_pics=args.real, # No quant, generate real pics
        gpu_id=args.gpu_id,
    )
    print(f"[INFO] benchmark: {benchmark}")

    quant_config = QuantizationConfigs.get_config(benchmark.quant_method)
    affine = quant_config["default"]["affine"]

    quant_layer_type="dit" if quant_config["default"]["dit"] else "controlnet"

    if "affine" in benchmark.quant_method: # if have affine, delete affine from quant_method
        quant_method = benchmark.quant_method.replace("_affine", "")
    else:
        quant_method = benchmark.quant_method
    if "affine" in benchmark.res_dir:
        model_dir = benchmark.res_dir.replace("_affine", "")
    else:
        model_dir = benchmark.res_dir
    model_quant_path = f"model/{quant_layer_type}/model_quant_{quant_method}.pt"
    model_quant_path = os.path.join(model_dir, model_quant_path)
    
    # quantize or load model
    model_quant = quant_or_load(model_quant_path, quant_config, benchmark.dataset, CalibrationConfig.to_dict(), benchmark.gpu_id)
    
    # get model to cpu, ready to learn better affine
    model_real = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
    )
    # learning better affine
    print(f"[INFO] learning better affine...")
    affiner = process_affiner(
        AffineConfig.to_dict(), benchmark.dataset, model_real, model_quant, latents=benchmark.latents, shuffle=True
    )
    print(f"[INFO] affine learning completed")

    model_quant = model_quant.to(f"cuda:{benchmark.gpu_id}")

    print(f"[INFO] generating pics with affine base on model_quant_{benchmark.quant_method}")
    trace_pic(
        model_quant,
        os.path.join(benchmark.res_dir, f"pics/quant_{benchmark.quant_method}"),
        benchmark.dataset.get_dataloader(batch_size=16),
        benchmark.latents,
        steper=affiner,
        max_num=1,
        continue_process=True,
        controlnet_conditioning_scale=benchmark.controlnet_conditioning_scale,
        guidance_scale=benchmark.guidance_scale,
        num_inference_steps=benchmark.max_timestep,
    )
    print(f"[INFO] pics generating with affine based on model_quant_{benchmark.quant_method} is completed")

    del model_quant
