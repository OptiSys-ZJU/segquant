import numpy as np
import torch
from dataset.processor import SubsetWrapper
from segquant.solver.blockwise_affiner import BlockwiseAffiner

def process_affiner(config, dataset, model_real, model_quant, latents=None, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
        dataset = SubsetWrapper(dataset, indices)
    
    affiner = BlockwiseAffiner(max_timestep=config['stepper']['max_timestep'],
                               blocksize=config['solver']['blocksize'],
                               sample_size=config['stepper']['sample_size'],
                               solver_type=config['solver']['type'],
                               solver_config=config['solver'],
                               latents=latents,
                               recurrent=config['stepper']['recurrent'],
                               noise_target=config['stepper']['noise_target'],
                               enable_latent=config['stepper']['enable_latent'],
                               enable_timesteps=config['stepper']['enable_timesteps'])

    if config['stepper']['recurrent']:
        for _ in range(config['max_timestep']):
            affiner.learning_real(model_real=model_real, data_loader=dataset.get_dataloader(), **config['extra_args'])
            affiner.learning_quant(model_quant=model_quant, data_loader=dataset.get_dataloader(), **config['extra_args'])
            affiner.replay_real(model_real=model_real, data_loader=dataset.get_dataloader(), **config['extra_args'])
            affiner.replay_quant(model_quant=model_quant, data_loader=dataset.get_dataloader(), **config['extra_args'])
    else:
        affiner.learning_real(model_real=model_real, data_loader=dataset.get_dataloader(), **config['extra_args'])
        affiner.learning_quant(model_quant=model_quant, data_loader=dataset.get_dataloader(), **config['extra_args'])
    
    affiner.finish_learning()

    return affiner

if __name__ == '__main__':
    from backend.torch.utils import randn_tensor
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel

    latents = randn_tensor((1, 16, 128, 128,), device=torch.device('cuda:0'), dtype=torch.float16)
    dataset = COCODataset(path='../dataset/controlnet_datasets/coco_canny', cache_size=16)

    model_quant = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')
    model_quant.transformer = torch.load('benchmark_record/run_seg_module/model/dit/model_quant_seg.pt', weights_only=False)
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')

    config = {
        "solver": {
            "type": 'mserel',
            "blocksize": 1,
            "alpha": 0.5,
            "lambda1": 0.1,
            "lambda2": 0.1,
            "sample_mode": 'block',
            "percentile": 100,
            "greater": True,
            "scale": 1,
            'verbose': True,
        },

        "stepper": {
            'max_timestep': 30,
            'sample_size': 1,
            'recurrent': False,
            'noise_target': 'all',
            'enable_latent': False,
            'enable_timesteps': None,
        },

        'extra_args': {
            "controlnet_conditioning_scale": 0,
            "guidance_scale": 7,
        }
    }

    affiner = process_affiner(config, dataset, model_real, model_quant, latents=latents, shuffle=False)

    #############################################
    from benchmark import trace_pic
    max_num = 1
    model_quant = model_quant.to('cuda')
    trace_pic(model_quant, f'affine_pics/blockaffine', dataset.get_dataloader(), latents, steper=affiner, max_num=max_num, num_inference_steps=config['stepper']['max_timestep'], **config['extra_args'])
    trace_pic(model_quant, f'affine_pics/quant', dataset.get_dataloader(), latents, max_num=max_num, num_inference_steps=config['stepper']['max_timestep'], **config['extra_args'])
    del model_quant
    model_real = model_real.to('cuda')
    trace_pic(model_real, f'affine_pics/real', dataset.get_dataloader(), latents, max_num=max_num, num_inference_steps=config['stepper']['max_timestep'], **config['extra_args'])



