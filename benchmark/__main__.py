
import torch
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from benchmark import trace_pic
from dataset.coco.coco_dataset import COCODataset
from sample.sampler import Q_DiffusionSampler, DNTCSampler
from segquant.torch.calibrate_set import generate_calibrate_set

def my_pre_hook(module, input):
    x = input[0]
    x_modified = x.clone()
    x_absmax = x_modified.abs().max()
    scale = x_absmax / 127.0 if x_absmax > 0 else 1.0
    negative_mask = x_modified < 0
    negative_values = x_modified[negative_mask]

    if negative_values.numel() > 0:
        q = torch.clamp((negative_values / scale).round(), -127, 127)
        dequant = q * scale
        x_modified[negative_mask] = dequant

    return (x_modified,)

def bench_sd3():
    latents = torch.load('../latents.pt')
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)

    hooks = []
    for i, transformer_block in enumerate(model.transformer.transformer_blocks):
        handle1 = transformer_block.norm1.linear.register_forward_pre_hook(my_pre_hook)
        handle2 = transformer_block.ff.net[2].register_forward_pre_hook(my_pre_hook)
        hooks.append((handle1, handle2))

    trace_pic(model, 'neg/avg', dataset.get_dataloader(), max_num=50, latents=latents, controlnet_conditioning_scale=0, num_inference_steps=60)
    for handle1, handle2 in hooks:
        handle1.remove()
        handle2.remove()

    trace_pic(model, 'neg/normal', dataset.get_dataloader(), max_num=50, latents=latents, controlnet_conditioning_scale=0, num_inference_steps=60)

if __name__ == '__main__':
    bench_sd3()