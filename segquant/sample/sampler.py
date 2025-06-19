"""
This module provides various samplers for extracting data from models at specified timesteps.
It includes a base sampler and specialized samplers such as UniformSampler and NormalSampler.
These samplers are designed to work with PyTorch models and datasets.
"""

from typing import List, Literal
import numpy as np
import torch
from torch import nn

model_map = {
    "dit": lambda x: x.transformer,
    "controlnet": lambda x: x.controlnet,
    "unet": lambda x: x.unet,
}


class BaseSampler:
    """Base class for samplers that sample data from a model at specified timesteps."""
    hook_map = {
        "input": "_hook_input",
        "output": "_hook_output",
        "inoutput": "_hook_inoutput",
    }

    def __init__(self):
        self.cur_t = 0
        self.device = "cpu"
        self.sample_line = []
        self.sample_timesteps = []

    def _process_tensor_tree(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone().to(self.device)
        if isinstance(obj, (list, tuple)):
            return [self._process_tensor_tree(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._process_tensor_tree(v) for k, v in obj.items()}
        return obj

    def _valid_timestep(self):
        return self.cur_t in self.sample_timesteps

    def _trigger_timestep(self):
        self.cur_t -= 1

    def _hook_input(self, _model, args, kwargs, _output):
        if self._valid_timestep():
            self.sample_line.append(
                {
                    "timestep": self.cur_t,
                    "input": {
                        "args": self._process_tensor_tree(args),
                        "kwargs": self._process_tensor_tree(kwargs),
                    },
                }
            )

        self._trigger_timestep()

    def _hook_output(self, _model, _args, _kwargs, output):
        if self._valid_timestep():
            self.sample_line.append(
                {
                    "timestep": self.cur_t,
                    "output": self._process_tensor_tree(output[0]),
                }
            )

        self._trigger_timestep()

    def _hook_inoutput(self, _model, args, kwargs, output):
        if self._valid_timestep():
            self.sample_line.append(
                {
                    "timestep": self.cur_t,
                    "input": {
                        "args": self._process_tensor_tree(args),
                        "kwargs": self._process_tensor_tree(kwargs),
                    },
                    "output": self._process_tensor_tree(output[0]),
                }
            )

        self._trigger_timestep()

    def _sample_timesteps(self, max_timestep, nums) -> List[int]:
        # Default implementation for BaseSampler
        return list(range(max_timestep, max_timestep - nums, -1))

    def _step_sample(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        input_data: tuple,
        num_inference_steps: int,
        timestep_per_sample: int,
        sample_mode: str,
        **kwargs
    ):
        self.cur_t = num_inference_steps - 1
        self.sample_timesteps = self._sample_timesteps(
            num_inference_steps - 1, timestep_per_sample
        )

        hook_function = getattr(self, self.hook_map[sample_mode])
        handle = target_layer.register_forward_hook(hook_function, with_kwargs=True)
        with torch.no_grad():
            prompt, _, control = input_data
            model(
                prompt=prompt,
                control_image=control,
                num_inference_steps=num_inference_steps,
                **kwargs
            )
        handle.remove()
        self.cur_t = 0

        yield self.sample_line
        self.sample_line = []

    def sample(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        target_layer=None,
        sample_layer: Literal["dit", "controlnet"] = "dit",
        max_timestep=30,
        sample_size=1,
        timestep_per_sample=5,
        sample_mode: Literal["input", "output", "inoutput"] = "input",
        device="cpu",
        **kwargs
    ):
        """
        Sample data from the model at specified timesteps.
        Args:
            model (nn.Module): The model to sample from.
            data_loader (DataLoader): DataLoader providing the input data.
            target_layer (nn.Module, optional): The layer to hook for sampling.
            sample_layer (str, optional): The type of layer to sample from ("dit" or "controlnet").
            max_timestep (int, optional): Maximum timestep for sampling.
            sample_size (int, optional): Number of samples to collect.
            timestep_per_sample (int, optional): Number of timesteps per sample.
            sample_mode (str, optional): Mode of sampling ("input", "output", "inoutput").
            device (str, optional): Device to run the model on.
            **kwargs: Additional arguments for the model's forward method.
        Yields:
            List[dict]: Sampled data containing timesteps, inputs, and outputs.
        """
        self.device = device

        if target_layer is None:
            target_layer = model_map[sample_layer](model)
        sample_count = 0
        for batch in data_loader:
            batch_size = len(batch)
            for i in range(batch_size):
                if sample_count >= sample_size:
                    return
                yield from self._step_sample(
                    model,
                    target_layer,
                    batch[i],
                    max_timestep,
                    timestep_per_sample,
                    sample_mode,
                    **kwargs
                )
                sample_count += 1


class UniformSampler(BaseSampler):
    """Sampler that samples uniformly across timesteps."""
    def _sample_timesteps(self, max_timestep, nums) -> List[int]:
        return list(np.linspace(max_timestep, 0, nums, dtype=int))


class NormalSampler(BaseSampler):
    """Sampler that samples timesteps based on a normal distribution."""
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

QDiffusionSampler = UniformSampler

if __name__ == "__main__":
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import (
        StableDiffusion3ControlNetModel,
    )

    sd3 = StableDiffusion3ControlNetModel.from_repo(
        ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = COCODataset(
        path="../dataset/controlnet_datasets/controlnet_canny_dataset", cache_size=16
    )

    sampler = QDiffusionSampler()
    for sample_data in sampler.sample(
        sd3,
        dataset.get_dataloader(),
        target_layer=sd3.controlnet.transformer_blocks[0].norm1.linear,
        sample_layer="dit",
        max_timestep=30,
        sample_size=1,
        timestep_per_sample=10,
        sample_mode="inoutput",
        controlnet_conditioning_scale=0.7,
        guidance_scale=3.5,
    ):

        for d in sample_data:
            t = d["timestep"]

            input_ = d["input"]["args"][0][0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = d[
                "output"
            ].chunk(6, dim=1)
