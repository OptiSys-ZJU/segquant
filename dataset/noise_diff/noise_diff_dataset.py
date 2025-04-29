import os
import numpy as np
import torch
from typing import List, Tuple, Union, Dict
from sample.noise_sampler import NoiseSampler
from torch.utils.data import Dataset, SubsetRandomSampler
import torch.nn as nn


class NoiseDiffDataset(Dataset):
    @staticmethod
    def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List]]]) -> Dict:
        first_elem = samples[0]["history_noise_pred"][0]
        is_tuple = isinstance(first_elem, tuple)

        def stack_history_field(key: str):
            return torch.stack([
                torch.stack(sample[key], dim=0)  # shape: [seq_len, ...]
                for sample in samples
            ], dim=0)  # shape: [batch_size, seq_len, ...]

        if is_tuple:
            a_stack = torch.stack([
                torch.stack([item[0] for item in sample["history_noise_pred"]], dim=0)
                for sample in samples
            ], dim=0)  # shape: [batch_size, seq_len, ...]
            b_stack = torch.stack([
                torch.stack([item[1] for item in sample["history_noise_pred"]], dim=0)
                for sample in samples
            ], dim=0)
            history_noise_pred = (a_stack.to(torch.float32), b_stack.to(torch.float32))

            last_a = a_stack[:, -1] 
            last_b = b_stack[:, -1]

            history_guidance_scale = stack_history_field("history_guidance_scale")
            quant_noise_pred = last_a + history_guidance_scale[:, -1].view(-1, 1, 1, 1) * (last_b - last_a)
        else:
            history_noise_pred = torch.stack([
                torch.stack(sample["history_noise_pred"], dim=0)
                for sample in samples
            ], dim=0).to(torch.float32)

            quant_noise_pred = history_noise_pred[:, -1]

        history_timestep = stack_history_field("history_timestep")
        if history_timestep.shape[-1] != 1:
            history_timestep = history_timestep.select(-1, 0).unsqueeze(-1)

        batched = {
            "history_noise_pred": history_noise_pred,
            "history_timestep": history_timestep.to(torch.float32),
            "history_controlnet_scale": stack_history_field("history_controlnet_scale").unsqueeze(-1).to(torch.float32),
            "history_guidance_scale": stack_history_field("history_guidance_scale").unsqueeze(-1).to(torch.float32),
            "real_noise_pred": torch.stack([sample["real_noise_pred"] for sample in samples], dim=0).to(torch.float32),
            "quant_noise_pred": quant_noise_pred.to(torch.float32),
        }

        return batched

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])
        self.index_map = []

        for file_idx, file_path in enumerate(self.file_list):
            data = torch.load(file_path, map_location='cpu')
            for item_idx in range(len(data)):
                self.index_map.append((file_idx, item_idx))

        self._cache = None
        self._cache_idx = -1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_idx, item_idx = self.index_map[index]

        if self._cache_idx != file_idx:
            self._cache = torch.load(self.file_list[file_idx], map_location='cpu')
            self._cache_idx = file_idx

        return self._cache[item_idx]
    
    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )


def generate_noise(model_real: nn.Module, model_quant: nn.Module, dataset:Dataset, target_device='cuda:0', max_timestep=100, sample_size=64, timestep_per_sample=100, window_size=3, controlnet_scale=0.7, guidance_scale=7, shuffle=True):
    '''
        all model must be load to cpu first
    '''
    
    assert next(model_real.parameters()).device == torch.device("cpu")
    assert next(model_quant.parameters()).device == torch.device("cpu")
    
    noise_sampler = NoiseSampler(controlnet_scale=controlnet_scale, guidance_scale=guidance_scale)

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    def sample_noise(model: nn.Module):
        outputs = []
        model.to(torch.device(target_device))
        data_loader = dataset.get_dataloader(sampler=SubsetRandomSampler(indices.copy()))
        for sample_data in noise_sampler.sample(model, 
                                          data_loader, 
                                          max_timestep=max_timestep, 
                                          sample_size=sample_size, 
                                          timestep_per_sample=timestep_per_sample):
            outputs.append(sample_data)
        model.to(torch.device('cpu'))
        return outputs

    real_outputs = sample_noise(model_real)
    quant_outputs = sample_noise(model_quant)

    def process_noise_pred(noise_pred: torch.Tensor, mix=False, guidance=None):
        if noise_pred.dim() == 0:
            raise ValueError("Unexpected scalar tensor")
        
        if noise_pred.size(0) == 2:
            # uncond, text
            if mix:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                return noise_pred
            else:
                return (noise_pred[0], noise_pred[1])
        elif noise_pred.size(0) == 1:
            return noise_pred.squeeze(0)
        else:
            raise ValueError(f"Unsupported first dimension size: {noise_pred.size(0)}")

    for real_data, quant_data in zip(real_outputs, quant_outputs):
        for i in range(window_size, len(quant_data)):
            history = quant_data[i - window_size + 1 : i + 1]
            target = real_data[i]

            this_data = {
                "history_noise_pred": [process_noise_pred(h['noise_pred']) for h in history],
                "history_timestep": [h['extra_features']['timestep'] for h in history],
                "history_controlnet_scale": [torch.tensor(h['extra_features']['controlnet_scale']) for h in history],
                "history_guidance_scale": [torch.tensor(h['extra_features']['guidance_scale']) for h in history],
                "real_noise_pred": process_noise_pred(target['noise_pred'], mix=True, guidance=target['extra_features']['guidance_scale']),
            }

            yield this_data

def generate_n_dump_noise(dump_dir: str, model_real: nn.Module, model_quant: nn.Module, dataset:Dataset, target_device='cuda:0', max_timestep=100, sample_size=64, timestep_per_sample=100, window_size=3, controlnet_scales=[0.7], guidance_scales=[7], shuffle=True):
    os.makedirs(dump_dir, exist_ok=True)

    buffer = []
    buffer_size = 64
    file_count = 0

    def dump_buffer():
        file_path = os.path.join(dump_dir, f"noise_{file_count:05d}.pt")
        torch.save(buffer, file_path)
        print(f"Saved {file_path} with {len(buffer)} samples")

    for c in controlnet_scales:
        for g in guidance_scales:
            for this_data in generate_noise(model_real, model_quant, dataset, target_device, max_timestep, sample_size, timestep_per_sample, window_size, c, g, shuffle):
                buffer.append(this_data)

                if len(buffer) >= buffer_size:
                    dump_buffer()
                    buffer = []
                    file_count += 1
    
    if buffer:
        dump_buffer()

if __name__ == '__main__':
    from dataset.coco.coco_dataset import COCODataset
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)


    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')

    def quant_model(model_real: nn.Module, quant_layer: str, config) -> nn.Module:
        from sample.sampler import Q_DiffusionSampler, model_map
        from segquant.torch.calibrate_set import generate_calibrate_set
        from segquant.torch.quantization import quantize

        sampler = Q_DiffusionSampler()
        sample_dataloader = dataset.get_dataloader(batch_size=1)
        calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, 
                                          max_timestep=30,
                                          sample_size=16,
                                          timestep_per_sample=10,
                                          controlnet_conditioning_scale=0.7,
                                          guidance_scale=7)

        calib_loader = calibset.get_dataloader(batch_size=1)
        model_real.controlnet = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
        return model_real
    
    from segquant.config import DType, SegPattern
    config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
        },
    }
    
    ##### quant model
    model_quant = quant_model(model, 'controlnet', config).to('cpu')

    ##### load real model
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')

    ##### dump data
    #generate_n_dump_noise('../noise_dataset/train', model_real, model_quant, dataset, 'cuda:0', max_timestep=30, sample_size=64, timestep_per_sample=30, window_size=3, controlnet_scales=[0.7], guidance_scales=[7], shuffle=True)
    generate_n_dump_noise('../noise_dataset/val', model_real, model_quant, dataset, 'cuda:0', max_timestep=30, sample_size=8, timestep_per_sample=30, window_size=3, controlnet_scales=[0.7], guidance_scales=[7], shuffle=True)

