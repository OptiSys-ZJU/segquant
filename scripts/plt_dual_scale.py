import torch
from sample.sampler import Q_DiffusionSampler
from dataset.coco.coco_dataset import COCODataset
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)

    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    sampler = Q_DiffusionSampler()
    for sample_data in sampler.sample(model, 
                                      dataset.get_dataloader(), 
                                    #   target_layer=model.transformer.transformer_blocks[0].norm1.linear,
                                      target_layer=model.transformer.transformer_blocks[0].ff.net[2],
                                      sample_layer='dit', 
                                      max_timestep=30, 
                                      sample_size=1, 
                                      timestep_per_sample=10, 
                                      sample_mode='input',
                                      controlnet_conditioning_scale=0.7,
                                      guidance_scale=3.5):
        
        for d in sample_data:
            t = d['timestep']
            input = d['input']['args'][0][0]
            silu_output = input.cpu().numpy()
            negative_ratio_sampled = (silu_output < -1e-6).sum() / silu_output.flatten().size
            print(t, input.shape, (silu_output < 0).sum(), silu_output.flatten().size, f"Negative value ratio: {negative_ratio_sampled:.4f}")

            # plt.figure(figsize=(12, 6))
            # plt.hist(silu_output.flatten(), bins=100, alpha=0.7, density=True, color='blue')
            # plt.xlabel("Activation Value")
            # plt.ylabel("Density")
            # plt.title("Activation Distribution Across All Channels (SiLU Output)")
            # plt.tight_layout()

            # plt.figure(figsize=(10, 3))
            # plt.plot(range(len(silu_output)), silu_output, color='purple', linewidth=1)
            # plt.title("SiLU Activation across Channels")
            # plt.xlabel("Channel Index")
            # plt.ylabel("Activation Value")
            # plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            # plt.grid(True, linestyle='--', alpha=0.4)
            
            # plt.savefig("silu.png", dpi=300, bbox_inches='tight')

            
            
