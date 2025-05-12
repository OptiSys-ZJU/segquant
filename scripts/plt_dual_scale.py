import torch
from sample.sampler import Q_DiffusionSampler
from dataset.coco.coco_dataset import COCODataset
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json

def stat_channels(layers, model, dataset, output_file='channel_stats.json'):
    sampler = Q_DiffusionSampler()
    stats = []

    layer_map = {id(m): name for name, m in model.named_modules()}

    for layer in layers:
        total_negative_channels = 0
        total_positive_channels = 0
        total_channels = 0

        for sample_data in sampler.sample(model, 
                                          dataset.get_dataloader(shuffle=True), 
                                          target_layer=layer,
                                          sample_layer='dit', 
                                          max_timestep=30, 
                                          sample_size=2, 
                                          timestep_per_sample=30, 
                                          sample_mode='input',
                                          controlnet_conditioning_scale=0.7,
                                          guidance_scale=3.5):
            
            for d in sample_data:
                input = d['input']['args'][0][0]
                silu_output = input.cpu().numpy()

                if silu_output.ndim == 2:
                    for i in range(silu_output.shape[0]):
                        total_channels += silu_output.shape[1]
                        total_negative_channels += (silu_output[i] < -1e-7).sum()
                        total_positive_channels += (silu_output[i] > 1e-7).sum()
                else:
                    total_channels += silu_output.shape[0]
                    total_negative_channels += (silu_output < -1e-7).sum()
                    total_positive_channels += (silu_output > 1e-7).sum()

        negative_channel_ratio = total_negative_channels / total_channels if total_channels > 0 else 0
        positive_channel_ratio = total_positive_channels / total_channels if total_channels > 0 else 0

        layer_name = layer_map.get(id(layer), str(layer))
        layer_stats = {
            "Layer": layer_name,
            "Total Channels": int(total_channels),
            "Negative Channels": int(total_negative_channels),
            "Positive Channels": int(total_positive_channels),
            "Negative Channel Ratio": round(float(negative_channel_ratio), 6),
            "Positive Channel Ratio": round(float(positive_channel_ratio), 6)
        }
        stats.append(layer_stats)

        with open(output_file, 'w') as json_file:
            json.dump(stats, json_file, indent=4)

        print(f'{layer} ok -> saved to {output_file}')

    print(f"All statistics saved to {output_file}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    layers_to_stat = []
    for i in range(len(model.transformer.transformer_blocks)):
        layers_to_stat.append(model.transformer.transformer_blocks[i].norm1.linear)
        layers_to_stat.append(model.transformer.transformer_blocks[i].ff.net[2])
    for i in range(len(model.controlnet.transformer_blocks)):
        layers_to_stat.append(model.controlnet.transformer_blocks[i].norm1.linear)
        layers_to_stat.append(model.controlnet.transformer_blocks[i].ff.net[2])
    
    print(layers_to_stat)
    stat_channels(layers_to_stat, model, dataset)
