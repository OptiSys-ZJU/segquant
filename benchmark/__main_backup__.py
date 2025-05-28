import os
import torch
from torch import nn
from benchmark import trace_pic
from segquant.config import DType, Optimum, SegPattern
from segquant.torch.affiner import process_affiner
from segquant.sample.sampler import QDiffusionSampler, model_map
from segquant.torch.calibrate_set import generate_calibrate_set
from segquant.torch.quantization import quantize
from dataset.coco.coco_dataset import COCODataset
from backend.torch.models.stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetModel,
)
from backend.torch.utils import randn_tensor

COCODatasetSize = 5000

calib_args = {
    "max_timestep": 50,
    "sample_size": 256,
    "timestep_per_sample": 50,
    "controlnet_conditioning_scale": 0,
    "guidance_scale": 7,
    "shuffle": True,
}

affine_config = {
    "solver": {
        "type": "mserel",
        "blocksize": 8,
        "alpha": 0.5,
        "lambda1": 0.1,
        "lambda2": 0.1,
        "sample_mode": "interpolate",
        "percentile": 100,
        "greater": True,
        "scale": 1,
        "verbose": True,
    },
    "stepper": {
        "type": "blockwise",
        "max_timestep": 50,
        "sample_size": 1,
        "recurrent": True,
        "noise_target": "uncond",
        "enable_latent_affine": False,
        "enable_timesteps": None,
    },
    "extra_args": {"controlnet_conditioning_scale": 0, "guidance_scale": 7,},
}

# quantize model with default config
quant_config_default = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": False,
        "search_patterns": [],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    },
}

# quantize model with seg
quant_config_seg = {
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
        "dit": True,
        "controlnet": False,
    },
}

# quantize model with dual scale
quant_config_dual_scale = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": True,
        "search_patterns": [SegPattern.ACTIVATION2LINEAR],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    },
}

quant_config_seg_dual = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": True,
        "search_patterns": [SegPattern.seg(), SegPattern.ACTIVATION2LINEAR],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    },
}

def quant_or_load(model_target_path, target_config, dataset):
    # if model_target_path not exist, quantize model
    if not os.path.exists(model_target_path):
        print(f"[INFO] {model_target_path} not found, start quantizing...")

        model = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
            "cuda:0",
        )
        # decide to quant, transfer to cpu for saving
        target_model = quant_model(
            model, "dit", target_config, dataset, calib_args
        ).to("cpu")
        # save model
        os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
        torch.save(target_model.transformer, model_target_path)
        print(f"[INFO] Model quantizing ok, saved to {model_target_path}")
    # if model_target_path exist, load model
    else:
        print(f"[INFO] {model_target_path} found, start loading...")
        target_model = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
            "cpu",
        )
        target_model.transformer = torch.load(model_target_path, weights_only=False)

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

def run_module(root_dir, dataset, max_timestep, max_num, latents, quant_method, quant_config, affine=False):
    # setting up model path
    quant_model_type="dit" if quant_config["default"]["dit"] else "controlnet"
    model_quant_path = f"model/{quant_model_type}/model_quant_{quant_method}.pt"
    model_quant_path = os.path.join(root_dir, model_quant_path)
    
    # quantize or load model
    model_quant = quant_or_load(model_quant_path, quant_config, dataset)
    

    if not affine:
        # get model ready to generate pics
        model_quant = model_quant.to("cuda")
        # generate quant pics
        print(f"[INFO] generating pics with model_quant_{quant_method}")
        trace_pic(
            model_quant,
            os.path.join(root_dir, f"pics/quant_{quant_method}"),
            dataset.get_dataloader(),
            latents,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        print(f"[INFO] pics generating with model_quant_{quant_method} is completed")
    else:
        # get model to cpu, ready to learn better affine
        model_real = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
        )
        # learning better affine
        print(f"[INFO] learning better affine...")
        affiner = process_affiner(
            affine_config, dataset, model_real, model_quant, latents=latents, shuffle=True
        )
        print(f"[INFO] affine learning completed")
        # get model to gpu, ready to generate pics
        model_quant = model_quant.to("cuda")
        # generate pics with affine
        print(f"[INFO] generating pics with affine base on model_quant_{quant_method}")
        trace_pic(
            model_quant,
            os.path.join(root_dir, f"pics/quant_{quant_method}"),
            dataset.get_dataloader(),
            latents,
            steper=affiner,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        print(f"[INFO] pics generating with affine based on model_quant_{quant_method} is completed")
    del model_quant

    print(f"[INFO] pics generating by model_quant_{quant_method} is completed")

def run_seg_module():
    root_dir = "../segquant/benchmark_record/run_seg_module"
    os.makedirs(root_dir, exist_ok=True)

    dataset = COCODataset(
        path="../dataset/controlnet_datasets/coco_canny", cache_size=16
    )

    max_timestep = 50
    # need to adjust according to the dataset size
    max_num = 5000

    # load model
    model_real = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
    )
    model_real = model_real.to("cuda")
    
    # load latent
    latents = torch.load("../latents.pt")

    # find if real pics exist, create if not
    pic_path = os.path.join(root_dir, "pics/real")
    if not os.path.exists(pic_path):
        print(f"[INFO] generating pics in [{pic_path}]...")
        trace_pic(
            model_real,
            pic_path,
            dataset.get_dataloader(),
            latents,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        print("model_real completed")
    else:
        print(f"[INFO] found pics in [{pic_path}], skip")

    # quantize or load model
    model_quant_path = os.path.join(root_dir, "model/dit/model_quant.pt")
    model_quant = quant_or_load(model_quant_path, quant_config_default, dataset)
    model_quant = model_quant.to("cuda")


    # generate quant pics
    trace_pic(
        model_quant,
        os.path.join(root_dir, "pics/quant"),
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
        guidance_scale=calib_args["guidance_scale"],
        num_inference_steps=max_timestep,
    )

    # delete model
    del model_quant

    print(f"[INFO] pics generating by model_quant is completed")

    # quantize model with seg
    quant_config_with_seg = {
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
    }
    model_quant_seg_path = os.path.join(root_dir, "model/dit/model_quant_seg.pt")
    model_quant_seg = quant_or_load(model_quant_seg_path, quant_config_with_seg, dataset)
    model_quant_seg = model_quant_seg.to("cuda")
    latents = torch.load("../latents.pt")
    trace_pic(
        model_quant_seg,
        os.path.join(root_dir, "pics/quant_seg"),
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
        guidance_scale=calib_args["guidance_scale"],
        num_inference_steps=max_timestep,
    )
    del model_quant_seg
    print(f"[INFO] pics generating by model_quant_seg is completed")

def run_dual_scale_module():
    root_dir = "../segquant/benchmark_record/run_dual_scale_module"
    os.makedirs(root_dir, exist_ok=True)

    dataset = COCODataset(
        path="../dataset/controlnet_datasets/coco_canny", cache_size=16
    )

    max_timestep = 50
    max_num = 5000

    # quantize model with dual scale
    quant_config_with_dual_scale = {
        "default": {
            "enable": True,
            "input_dtype": DType.INT8,
            "weight_dtype": DType.INT8,
            "opt": Optimum.SMOOTH,
            "seglinear": True,
            "search_patterns": [SegPattern.ACTIVATION2LINEAR],
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }
    model_quant_dual_scale_path = os.path.join(
        root_dir, "model/dit/model_quant_dual_scale.pt"
    )
    model_quant_dual_scale = quant_or_load(
        model_quant_dual_scale_path, quant_config_with_dual_scale, dataset
    )
    model_quant_dual_scale = model_quant_dual_scale.to("cuda")
    latents = torch.load("../latents.pt")
    trace_pic(
        model_quant_dual_scale,
        os.path.join(root_dir, "pics/quant_dual_scale"),
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
        guidance_scale=calib_args["guidance_scale"],
        num_inference_steps=max_timestep,
    )
    del model_quant_dual_scale
    print(f"[INFO] pics generating by model_quant_dual_scale is completed")

def run_affiner_module():
    # randomly generate latents to learn better affine
    latents = randn_tensor(
        (1, 16, 128, 128,), device=torch.device("cuda:0"), dtype=torch.float16
    )

    max_num = 5000

    os.makedirs(root_dir, exist_ok=True)
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/coco_canny", cache_size=16
    )

    model_quant = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
    )
    model_quant.transformer = torch.load(
        "../segquant/benchmark_record/run_seg_module/model/dit/model_quant_seg.pt",
        weights_only=False,
    )
    model_real = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
    )
    # first use random latents to learn better affine
    affiner = process_affiner(
        affine_config, dataset, model_real, model_quant, latents=latents, shuffle=True
    )

    # then use real latent and affined models to generate pics
    max_num = 1
    model_quant = model_quant.to("cuda")
    trace_pic(
        model_quant,
        "affine_pics/blockaffine",
        dataset.get_dataloader(),
        latents,
        steper=affiner,
        max_num=max_num,
        num_inference_steps=affine_config["stepper"]["max_timestep"],
        **affine_config["extra_args"],
    )
    trace_pic(
        model_quant,
        "affine_pics/quant",
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        num_inference_steps=affine_config["stepper"]["max_timestep"],
        **affine_config["extra_args"],
    )
    del model_quant
    model_real = model_real.to("cuda")
    trace_pic(
        model_real,
        "affine_pics/real",
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        num_inference_steps=affine_config["stepper"]["max_timestep"],
        **affine_config["extra_args"],
    )


if __name__ == "__main__":
    # setting up args
    quant_method = "default"
    # quant_method = "seg"
    # quant_method = "dual_scale"
    # quant_method = "seg_dual"
    root_dir = f"../segquant/benchmark_record/run_{quant_method}_module"
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/coco_canny", cache_size=16
    )

    max_timestep = 50
    max_num = COCODatasetSize
    latents = torch.load("../latents.pt")

    # if needed to generate real pics under fp16model
    generate_real_pics = False
    if generate_real_pics:
        # load real model
        model_real = StableDiffusion3ControlNetModel.from_repo(
            ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), "cpu"
        )
        model_real = model_real.to("cuda")

        # generate real pics
        real_dir = f"../segquant/benchmark_record/run_real_module" 
        pic_path = os.path.join(real_dir, "pics/real")
        print(f"[INFO] generating real pics in [{pic_path}]...")
        trace_pic(
            model_real,
            pic_path,
            dataset.get_dataloader(),
            latents,
            max_num=max_num,
            controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
            guidance_scale=calib_args["guidance_scale"],
            num_inference_steps=max_timestep,
        )
        print(f"[INFO] real pics generated in [{pic_path}]")
    
    run_module(root_dir, dataset, max_timestep, max_num, latents, quant_method, quant_config_default, affine=True)
    
