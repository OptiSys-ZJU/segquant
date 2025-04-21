from typing import List, Literal
import numpy as np
import torch
import torch.nn as nn

from dataset.coco.coco_dataset import COCODataset

model_map = {
    "dit": lambda x: x.transformer,
    "controlnet": lambda x:x.controlnet,
}

class BaseSampler:
    hook_map = {
        'input': '_hook_input',
        'output': '_hook_output',
        'inoutput': '_hook_inoutput',
    }

    def __init__(self):
        self.cur_t = 0
        self.sample_line = []
        self.sample_timesteps = []
    
    def _trigger_timestep(self):
        self.cur_t -= 1
        return self.cur_t in self.sample_timesteps

    def _hook_input(self, model, input, output):
        if self._trigger_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "input": input,
            })
    
    def _hook_output(self, model, input, output):
        if self._trigger_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "output": output,
            })
    
    def _hook_inoutput(self, model, input, output):
        if self._trigger_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "input": input,
                "output": output,
            })

    def _sample_timesteps(self, max_timestep, nums) -> List[int]:
        pass

    def _step_sample(self, model: nn.Module, target_layer: nn.Module, input, num_inference_steps: int, timestep_per_sample: int, sample_mode: str):
        self.cur_t = num_inference_steps
        self.sample_timesteps = self._sample_timesteps(num_inference_steps, timestep_per_sample)

        hook_function = getattr(self, self.hook_map[sample_mode])
        handle = target_layer.register_forward_hook(hook_function)
        with torch.no_grad():
            prompt, _, control = input
            model(prompt=prompt, control_image=control, num_inference_steps=num_inference_steps)
        handle.remove()
        self.cur_t = 0

        yield self.sample_line
        self.sample_line = []

    def sample(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, sample_layer: Literal['dit', 'controlnet'] = 'dit', max_timestep=30, sample_size=1, timestep_per_sample=5, sample_mode: Literal['input', 'output', 'inoutput'] = 'input'):
        target_layer = model_map[sample_layer](model)
        for i, batch in enumerate(data_loader):
            if i > sample_size:
                break
            yield from self._step_sample(model, target_layer, batch[0], max_timestep, timestep_per_sample, sample_mode)


class UniformSampler(BaseSampler):
    def __init__(self):
        super().__init__()
    
    def _sample_timesteps(self, max_timestep, nums) -> List[int]:
        return list(np.linspace(max_timestep, 0, nums, dtype=int))

class NormalSampler(BaseSampler):
    def __init__(self, mean):
        super().__init__()
        self.mean = mean
    
    def _sample_timesteps(self, max_timestep: int, nums: int) -> List[int]:
        std = max_timestep / 2
        samples = torch.normal(mean=self.mean, std=std, size=(nums,))
        clamped = samples.clamp(min=0, max=max_timestep)
        rounded = [int(round(v.item())) for v in clamped]
        unique_sorted = sorted(set(rounded))
        return unique_sorted

DNTCSampler = NormalSampler

Q_DiffusionSampler = UniformSampler

if __name__ == '__main__':
    from backend.torch.models.flux_controlnet import FluxControlNetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluxControlNetModel.from_repo(('../FLUX.1-dev', '../FLUX.1-dev-Controlnet-Canny'), device)

    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    sampler = Q_DiffusionSampler()
    for sample_data in sampler.sample(model, dataset.get_dataloader(), sample_layer='dit', max_timestep=30, sample_size=5, timestep_per_sample=5, sample_mode='input'):
        print(sample_data)