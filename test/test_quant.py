import torch
from segquant.config import DType, Optimum, SegPattern
from segquant.torch.quantization import quantize


def cali(quant_layer):
    from dataset.coco.coco_dataset import COCODataset
    from segquant.sample.sampler import Q_DiffusionSampler
    from segquant.torch.calibrate_set import generate_calibrate_set
    from backend.torch.models.stable_diffusion_3_controlnet import (
        StableDiffusion3ControlNetModel,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), device
    )
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/controlnet_canny_dataset", cache_size=16
    )
    calibset = generate_calibrate_set(
        model,
        Q_DiffusionSampler(),
        dataset.get_dataloader(batch_size=1),
        quant_layer,
        max_timestep=30,
        sample_size=32,
        timestep_per_sample=30,
        controlnet_conditioning_scale=0.7,
        guidance_scale=3.5,
    )
    return calibset


def sample_noise_output(config, calibset, latents, quant_layer):
    from dataset.coco.coco_dataset import COCODataset
    from segquant.sample.sampler import Q_DiffusionSampler
    from backend.torch.models.stable_diffusion_3_controlnet import (
        StableDiffusion3ControlNetModel,
    )
    from segquant.sample.sampler import model_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), device
    )
    if quant_layer == "controlnet":
        model.controlnet = quantize(
            model_map[quant_layer](model),
            calibset.get_dataloader(batch_size=1),
            config,
            verbose=True,
        )
    elif quant_layer == "dit":
        model.transformer = quantize(
            model_map[quant_layer](model),
            calibset.get_dataloader(batch_size=1),
            config,
            verbose=True,
        )

    all_outputs = []
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/controlnet_canny_dataset", cache_size=16
    )
    for sample_data in Q_DiffusionSampler().sample(
        model,
        dataset.get_dataloader(),
        target_layer=model.transformer,
        sample_layer="dit",
        max_timestep=30,
        sample_size=1,
        timestep_per_sample=30,
        sample_mode="output",
        controlnet_conditioning_scale=0.7,
        guidance_scale=3.5,
        latents=latents,
    ):

        all_outputs.append([])
        for d in sample_data:
            output = d["output"]
            all_outputs[-1].append(output.clone().detach().cpu())

    return all_outputs


def sample_noise_dis(config, calibset, latents, quant_layer):
    from dataset.coco.coco_dataset import COCODataset
    from segquant.sample.sampler import Q_DiffusionSampler
    from backend.torch.models.stable_diffusion_3_controlnet import (
        StableDiffusion3ControlNetModel,
    )
    from segquant.sample.sampler import model_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), device
    )
    if quant_layer == "controlnet":
        model.controlnet = quantize(
            model_map[quant_layer](model),
            calibset.get_dataloader(batch_size=1),
            config,
            verbose=True,
        )
    elif quant_layer == "dit":
        model.transformer = quantize(
            model_map[quant_layer](model),
            calibset.get_dataloader(batch_size=1),
            config,
            verbose=True,
        )

    all_outputs = []
    dataset = COCODataset(
        path="../dataset/controlnet_datasets/controlnet_canny_dataset", cache_size=16
    )
    for sample_data in Q_DiffusionSampler().sample(
        model,
        dataset.get_dataloader(),
        target_layer=model.transformer,
        sample_layer="dit",
        max_timestep=30,
        sample_size=1,
        timestep_per_sample=30,
        sample_mode="output",
        controlnet_conditioning_scale=0.7,
        guidance_scale=3.5,
        latents=latents,
    ):
        all_outputs.append([])
        for d in sample_data:
            output = d["output"]  # [batch, channel, h, w]
            mean = output.mean()
            std = output.std()
            all_outputs[-1].append((mean.item(), std.item()))

    return all_outputs


def compute_frobenius_norms(outputs_a, outputs_b):
    outputs_a_T = list(zip(*outputs_a))
    outputs_b_T = list(zip(*outputs_b))

    norms = []
    for timestep_outputs_a, timestep_outputs_b in zip(outputs_a_T, outputs_b_T):
        fro_norms = [
            torch.norm(a - b, p="fro")
            for a, b in zip(timestep_outputs_a, timestep_outputs_b)
        ]
        avg_norm = torch.stack(fro_norms).mean().item()
        norms.append(avg_norm)
    return norms


base_config = {
    "default": {"enable": False,},
}
default_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": False,
        "search_patterns": SegPattern.all(),
        "input_axis": None,
        "weight_axis": None,
        "alpha": 1.0,
    },
}
seg_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": True,
        "search_patterns": [
            SegPattern.Linear2Chunk,
            SegPattern.Linear2Split,
            SegPattern.Concat2Linear,
            SegPattern.Stack2Linear,
        ],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 1.0,
    },
}
seg_dual_config = {
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
enable_latent_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": False,
        "search_patterns": SegPattern.all(),
        "input_axis": None,
        "weight_axis": None,
        "alpha": 1.0,
    },
    "*time_text_embed*": {"enable": False,},
    "*transformer_blocks.*.norm1.linear*": {"enable": False,},
    "*transformer_blocks.*.norm1_context.linear*": {"enable": False,},
}
enable_time_config = {
    "default": {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": False,
        "search_patterns": SegPattern.all(),
        "input_axis": None,
        "weight_axis": None,
        "alpha": 1.0,
    },
    "*pos_embed.proj*": {"enable": False,},
    "*pos_embed_input.proj*": {"enable": False,},
    "*context_embedder.proj*": {"enable": False,},
    "*transformer_blocks.*.attn*": {"enable": False,},
    "*transformer_blocks.*.ff.net*": {"enable": False,},
    "*transformer_blocks.*.ff_context.net*": {"enable": False,},
    "*controlnet_blocks*": {"enable": False,},
}


def cal_multistep_for():
    results = {}
    configs = {
        "base": base_config,
        "default": default_config,
        "latent": enable_latent_config,
        "time": enable_time_config,
    }

    latents = torch.load("../latents.pt")

    quant_layer = "dit"

    calibset = cali(quant_layer)

    base_outputs = sample_noise_output(base_config, calibset, latents, quant_layer)

    for name, cfg in configs.items():
        print(f"Running config: {name}")
        outputs = sample_noise_output(cfg, calibset, latents, quant_layer)
        norms = compute_frobenius_norms(outputs, base_outputs)
        results[name] = norms

    print("\nFrobenius Norms per Timestep:\n")
    max_len = max(len(v) for v in results.values())

    for i in range(max_len):
        line = f"Timestep {i:03d}: "
        for name in configs:
            val = results[name][i] if i < len(results[name]) else None
            line += f"{name} = {val:.4f} " if val is not None else f"{name} = N/A "
        print(line)


def cal_multistep_dis():
    results = {}

    latents = torch.load("../latents.pt")

    quant_layer = "dit"

    calibset = cali(quant_layer)

    base_outputs = sample_noise_dis(base_config, calibset, latents, quant_layer)[0]
    outputs = sample_noise_dis(default_config, calibset, latents, quant_layer)[0]

    for b, i in zip(base_outputs, outputs):
        print("mean", b[0], i[0], "std", b[1], i[1])


if __name__ == "__main__":
    cal_multistep_dis()
