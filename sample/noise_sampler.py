import torch
from sample.sampler import UniformSampler


class NoiseSampler(UniformSampler):
    def __init__(self, controlnet_scale=0.7, guidance_scale=7):
        super().__init__()
        self.controlnet_scale = controlnet_scale
        self.guidance_scale = guidance_scale

    def _hook_input(self, model, args, kwargs, output):
        raise NotImplementedError('NoiseSampler: _hook_input not implemented')
    
    def _hook_output(self, model, args, kwargs, output):
        raise NotImplementedError('NoiseSampler: _hook_output not implemented')
    
    def _hook_inoutput(self, model, args, kwargs, output):
        if self._valid_timestep():
            if 'timestep' in kwargs:
                self.sample_line.append({
                    "timestep": self.cur_t,
                    "extra_features": {
                        'timestep': self._process_tensor_tree(kwargs['timestep']),
                        'controlnet_scale': self.controlnet_scale,
                        'guidance_scale': self.guidance_scale,
                    },
                    "noise_pred": self._process_tensor_tree(output[0]),
                })
            else:
                raise ValueError('timestep not in kwargs')
        
        self._trigger_timestep()
    
    def sample(self, model, data_loader, max_timestep=30, sample_size=1, timestep_per_sample=5, device='cpu'):
        return super().sample(model, 
                              data_loader, 
                              None, 
                              'dit', 
                              max_timestep, 
                              sample_size, 
                              timestep_per_sample, 
                              'inoutput', 
                              device, 
                              controlnet_conditioning_scale=self.controlnet_scale, 
                              guidance_scale=self.guidance_scale,)


if __name__ == '__main__':
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FluxControlNetModel.from_repo(('../FLUX.1-dev', '../FLUX.1-dev-Controlnet-Canny'), device)
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)

    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    sampler = NoiseSampler(controlnet_scale=0.7, guidance_scale=7)
    data_loader = dataset.get_dataloader()
    for sample_data in sampler.sample(model, 
                                      dataset.get_dataloader(), 
                                      max_timestep=30, 
                                      sample_size=1, 
                                      timestep_per_sample=1):
        
        for single_data in sample_data:
            print(single_data)
