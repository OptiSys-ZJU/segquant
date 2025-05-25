import copy
import os

import torch
from tqdm import tqdm
from backend.torch.models.stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetModel,
)
from benchmark import trace_pic
from dataset.coco.coco_dataset import COCODataset
from segquant.sample.sampler import Q_DiffusionSampler, model_map
from segquant.config import DType, Optimum, SegPattern
from segquant.torch.calibrate_set import generate_calibrate_set
from segquant.torch.quantization import quantize
import modelopt.torch.quantization as mtq


calib_args = {
    "max_timestep": 50,
    "sample_size": 1,
    "timestep_per_sample": 50,
    "controlnet_conditioning_scale": 0,
    "guidance_scale": 7,
    "shuffle": False,
}

dataset = COCODataset(path="../dataset/controlnet_datasets/coco_canny", cache_size=16)
model_real = StableDiffusion3ControlNetModel.from_repo(
    ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cuda"
)
sampler = Q_DiffusionSampler()
sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
calibset = generate_calibrate_set(
    model_real,
    sampler,
    sample_dataloader,
    "dit",
    max_timestep=calib_args["max_timestep"],
    sample_size=calib_args["sample_size"],
    timestep_per_sample=calib_args["timestep_per_sample"],
    controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
    guidance_scale=calib_args["guidance_scale"],
)
calib_loader = calibset.get_dataloader(batch_size=1)


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return tuple(move_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def modelopt_loop(model):
    for batch in calib_loader:
        this_input_tuple = move_to_device(batch[0], "cuda")
        model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
            **this_input_tuple
        )


def modelopt_test():
    config = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": -1},
            "nn.BatchNorm1d": {"*": {"enable": False}},
            "nn.BatchNorm2d": {"*": {"enable": False}},
            "nn.BatchNorm3d": {"*": {"enable": False}},
            "nn.LeakyReLU": {"*": {"enable": False}},
            "*lm_head*": {"enable": False},
            "*proj_out.*": {
                "enable": False
            },  # In Whisper model, lm_head has key name proj_out
            "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
            "*router*": {"enable": False},  # Skip the MOE router
            "*mlp.gate.*": {"enable": False},  # Skip the MOE router
            "*mlp.shared_expert_gate.*": {"enable": False},  # Skip the MOE router
            "*output_layer*": {"enable": False},
            "output.*": {"enable": False},
            "default": {"enable": False},
            "*pos_embed.proj*": {"enable": False},
            "*transformer_blocks.*.norm1.linear.weight_quantizer": {
                "num_bits": 8,
                "block_sizes": {0: 1536},
                "enable": True,
            },
            "*transformer_blocks.*.norm1_context.linear.weight_quantizer": {
                "num_bits": 8,
                "block_sizes": {0: 1536},
                "enable": True,
            },
        },
        "algorithm": {"method": "smoothquant", "alpha": 0.5},
    }
    model_real.transformer = mtq.quantize(
        model_map["dit"](model_real), config, modelopt_loop
    )
    mtq.print_quant_summary(model_real.transformer)


def segquant_test():
    quant_config = {
        "default": {
            "enable": True,
            "input_dtype": DType.INT8,
            "weight_dtype": DType.INT8,
            "opt": Optimum.SMOOTH,
            "seglinear": True,
            "search_patterns": SegPattern.seg(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
        "*proj_out*": {"enable": False,},
    }
    model_real.transformer = quantize(
        model_map["dit"](model_real), calib_loader, quant_config, True
    )


if __name__ == "__main__":
    root_dir = "."
    max_timestep = 50
    max_num = 1

    modelopt_test()
    latents = torch.load("../latents.pt")
    trace_pic(
        model_real,
        os.path.join(root_dir, "pics/modelopt"),
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
        guidance_scale=calib_args["guidance_scale"],
        num_inference_steps=max_timestep,
    )

    del model_real
    model_real = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cuda"
    )
    segquant_test()
    latents = torch.load("../latents.pt")
    trace_pic(
        model_real,
        os.path.join(root_dir, "pics/segquant"),
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
        guidance_scale=calib_args["guidance_scale"],
        num_inference_steps=max_timestep,
    )
