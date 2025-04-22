from typing import List, Literal
import numpy as np
import torch
import torch.nn as nn

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
        self.device = 'cpu'
        self.sample_line = []
        self.sample_timesteps = []
    
    def _process_tensor_tree(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone().to(self.device)
        elif isinstance(obj, (list, tuple)):
            return [self._process_tensor_tree(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._process_tensor_tree(v) for k, v in obj.items()}
        else:
            return obj
    
    def _valid_timestep(self):
        return self.cur_t in self.sample_timesteps

    def _trigger_timestep(self):
        self.cur_t -= 1

    def _hook_input(self, model, args, kwargs, output):
        if self._valid_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "input": {
                    'args': self._process_tensor_tree(args),
                    'kwargs': self._process_tensor_tree(kwargs),
                },
            })
        
        self._trigger_timestep()
    
    def _hook_output(self, model, args, kwargs, output):
        if self._valid_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "output": self._process_tensor_tree(output),
            })
        
        self._trigger_timestep()
    
    def _hook_inoutput(self, model, args, kwargs, output):
        if self._valid_timestep():
            self.sample_line.append({
                "timestep": self.cur_t,
                "input": {
                    'args': self._process_tensor_tree(args),
                    'kwargs': self._process_tensor_tree(kwargs),
                },
                "output": self._process_tensor_tree(output),
            })
        
        self._trigger_timestep()

    def _sample_timesteps(self, max_timestep, nums) -> List[int]:
        pass

    def _step_sample(self, model: nn.Module, target_layer: nn.Module, input, num_inference_steps: int, timestep_per_sample: int, sample_mode: str, **kwargs):
        self.cur_t = num_inference_steps - 1
        self.sample_timesteps = self._sample_timesteps(num_inference_steps - 1, timestep_per_sample)

        hook_function = getattr(self, self.hook_map[sample_mode])
        handle = target_layer.register_forward_hook(hook_function, with_kwargs=True)
        with torch.no_grad():
            prompt, _, control = input
            model(prompt=prompt, control_image=control, num_inference_steps=num_inference_steps, **kwargs)
        handle.remove()
        self.cur_t = 0

        yield self.sample_line
        self.sample_line = []

    def sample(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, sample_layer: Literal['dit', 'controlnet'] = 'dit', max_timestep=30, sample_size=1, timestep_per_sample=5, sample_mode: Literal['input', 'output', 'inoutput'] = 'input', device='cpu', **kwargs):
        self.device = device
        
        target_layer = model_map[sample_layer](model)
        sample_count = 0
        for batch in data_loader:
            batch_size = len(batch)
            for i in range(batch_size):
                if sample_count >= sample_size:
                    return
                yield from self._step_sample(model, target_layer, batch[i], max_timestep, timestep_per_sample, sample_mode, **kwargs)
                sample_count += 1


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
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from backend.torch.models.flux_controlnet import FluxControlNetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FluxControlNetModel.from_repo(('../FLUX.1-dev', '../FLUX.1-dev-Controlnet-Canny'), device)
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)

    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    sampler = Q_DiffusionSampler()
    for sample_data in sampler.sample(model, 
                                      dataset.get_dataloader(), 
                                      sample_layer='dit', 
                                      max_timestep=30, 
                                      sample_size=1, 
                                      timestep_per_sample=5, 
                                      sample_mode='input',
                                      controlnet_conditioning_scale=0.7,
                                      guidance_scale=3.5):
        print(sample_data)