
import os
import torch
from tqdm import tqdm
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from dataset.coco.coco_dataset import COCODataset
from sample.sampler import Q_DiffusionSampler, DNTCSampler
from segquant.torch.calibrate_set import generate_calibrate_set

def trace_pic(model, path, data_loader, latents=None, max_num=None, **kwargs):
    os.makedirs(path, exist_ok=True)
    count = 0
    pbar = tqdm(total=max_num if max_num is not None else len(data_loader.dataset), desc="Tracing images")
    for batch in data_loader:
        for b in batch:
            if max_num is not None and count >= max_num:
                pbar.close()
                return
            prompt, _, control = b
            image = model.forward(
                prompt=prompt,
                control_image=control,
                latents=latents,
                **kwargs
            )[0]
            save_path = os.path.join(path, f"{count:04d}.jpg")
            image[0].save(save_path)
            count += 1
            pbar.update(1)
    pbar.close()

def bench_sd3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)

if __name__ == '__main__':
    bench_sd3()