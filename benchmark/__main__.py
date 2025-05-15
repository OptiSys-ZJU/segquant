
import torch
from benchmark import trace_pic
import torch.nn as nn
from segquant.config import DType

calib_args = {
    "max_timestep": 50,
    "sample_size": 256,
    "timestep_per_sample": 50,
    "controlnet_conditioning_scale": 0,
    "guidance_scale": 7,
    "shuffle": True,
}

quant_config = {
    "default": {
        "enable": True,
        "dtype": DType.INT8SMOOTH,
        "seglinear": True,
        'search_patterns': [],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
    },
}

def run_dual_scale_module():
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from segquant.config import SegPattern

    quant_config_with_dual_scale = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': [SegPattern.Activation2Linear],
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }

    def quant_model(model_real: nn.Module, quant_layer: str, config, dataset, calib_args: dict) -> nn.Module:
        from sample.sampler import Q_DiffusionSampler, model_map
        from segquant.torch.calibrate_set import generate_calibrate_set
        from segquant.torch.quantization import quantize

        sampler = Q_DiffusionSampler()
        sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
        calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, 
                                        max_timestep=calib_args["max_timestep"],
                                        sample_size=calib_args["sample_size"],
                                        timestep_per_sample=calib_args["timestep_per_sample"],
                                        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                                        guidance_scale=calib_args["guidance_scale"])

        calib_loader = calibset.get_dataloader(batch_size=1)
        model_real.transformer = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
        return model_real
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    max_timestep = 50
    max_num = 64

    ### 0
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')    
    model_real = model_real.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_real, 'pics/run_dual_scale_module/real', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    print('model_real completed')

    ### 1
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant = quant_model(model, 'dit', quant_config, dataset, calib_args).to('cpu')
    model_quant = model_quant.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant, 'pics/run_dual_scale_module/quant', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant
    print('model_quant completed')

    ### 2
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant_dual_scale = quant_model(model, 'dit', quant_config_with_dual_scale, dataset, calib_args).to('cpu')
    model_quant_dual_scale = model_quant_dual_scale.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant_dual_scale, 'pics/run_dual_scale_module/quant_dual_scale', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant_dual_scale
    print('model_quant_dual_scale completed')

def run_seg_module():
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from segquant.config import SegPattern

    quant_config_with_seg = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.seg(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }

    def quant_model(model_real: nn.Module, quant_layer: str, config, dataset, calib_args: dict) -> nn.Module:
        from sample.sampler import Q_DiffusionSampler, model_map
        from segquant.torch.calibrate_set import generate_calibrate_set
        from segquant.torch.quantization import quantize

        sampler = Q_DiffusionSampler()
        sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
        calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, 
                                        max_timestep=calib_args["max_timestep"],
                                        sample_size=calib_args["sample_size"],
                                        timestep_per_sample=calib_args["timestep_per_sample"],
                                        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                                        guidance_scale=calib_args["guidance_scale"])

        calib_loader = calibset.get_dataloader(batch_size=1)
        model_real.transformer = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
        return model_real
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    max_timestep = 50
    max_num = 64

    ### 0
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')    
    model_real = model_real.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_real, 'pics/run_seg_module/real', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    print('model_real completed')

    ### 1
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant = quant_model(model, 'dit', quant_config, dataset, calib_args).to('cpu')
    model_quant = model_quant.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant, 'pics/run_seg_module/quant', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant
    print('model_quant completed')

    ### 2
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant_seg = quant_model(model, 'dit', quant_config_with_seg, dataset, calib_args).to('cpu')
    model_quant_seg = model_quant_seg.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant_seg, 'pics/run_seg_module/quant_seg', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant_seg
    print('model_quant_seg completed')

def run_seg_dual_module():
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from segquant.config import SegPattern

    quant_config_with_seg_dual = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }

    def quant_model(model_real: nn.Module, quant_layer: str, config, dataset, calib_args: dict) -> nn.Module:
        from sample.sampler import Q_DiffusionSampler, model_map
        from segquant.torch.calibrate_set import generate_calibrate_set
        from segquant.torch.quantization import quantize

        sampler = Q_DiffusionSampler()
        sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
        calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, 
                                        max_timestep=calib_args["max_timestep"],
                                        sample_size=calib_args["sample_size"],
                                        timestep_per_sample=calib_args["timestep_per_sample"],
                                        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                                        guidance_scale=calib_args["guidance_scale"])

        calib_loader = calibset.get_dataloader(batch_size=1)
        model_real.transformer = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
        return model_real
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    max_timestep = 50
    max_num = 64

    ### 0
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')    
    model_real = model_real.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_real, 'pics/run_seg_dual_module/real', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    print('model_real completed')

    ### 1
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant = quant_model(model, 'dit', quant_config, dataset, calib_args).to('cpu')
    model_quant = model_quant.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant, 'pics/run_seg_dual_module/quant', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant
    print('model_quant completed')

    ### 2
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
    model_quant_seg_dual = quant_model(model, 'dit', quant_config_with_seg_dual, dataset, calib_args).to('cpu')
    model_quant_seg_dual = model_quant_seg_dual.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant_seg_dual, 'pics/run_seg_dual_module/quant_dual_scale', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant_seg_dual
    print('model_quant_seg_dual completed')

if __name__ == '__main__':
    run_seg_module()