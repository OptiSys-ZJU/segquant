
import torch
from benchmark import trace_pic
import torch.nn as nn
from segquant.config import DType, Optimum
import os

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
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "seglinear": False,
        'search_patterns': [],
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
    },
}

def quant_model(model_real: nn.Module, quant_layer: str, config, dataset, calib_args: dict) -> nn.Module:
    from segquant.sample.sampler import Q_DiffusionSampler, model_map
    from segquant.torch.calibrate_set import generate_calibrate_set
    from segquant.torch.quantization import quantize

    calib_key = (
        f"maxT{calib_args['max_timestep']}_"
        f"sz{calib_args['sample_size']}_"
        f"tps{calib_args['timestep_per_sample']}_"
        f"cond{calib_args['controlnet_conditioning_scale']}_"
        f"gs{calib_args['guidance_scale']}_"
        f"{'shuffle' if calib_args['shuffle'] else 'noshuffle'}"
    )
    calibset_path = os.path.join('calibset_record', quant_layer, calib_key)
    sampler = Q_DiffusionSampler()
    sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
    calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, calibset_path, 
                                    max_timestep=calib_args["max_timestep"],
                                    sample_size=calib_args["sample_size"],
                                    timestep_per_sample=calib_args["timestep_per_sample"],
                                    controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                                    guidance_scale=calib_args["guidance_scale"])

    calib_loader = calibset.get_dataloader(batch_size=1)
    if quant_layer == 'dit':
        model_real.transformer = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
    else:
        model_real.controlnet = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
    return model_real

def run_seg_module():
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from segquant.config import SegPattern

    root_dir = 'benchmark_record/run_seg_module'
    os.makedirs(root_dir, exist_ok=True)
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/coco_canny', cache_size=16)

    max_timestep = 50
    max_num = 1024

    ### 0
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')    
    model_real = model_real.to('cuda')
    latents = torch.load('../latents.pt')
    pic_path = os.path.join(root_dir, 'pics/real')
    if not os.path.exists(pic_path):
        print(f'[INFO] generating pics in [{pic_path}]...')
        trace_pic(model_real, pic_path, dataset.get_dataloader(), latents, max_num=max_num, 
                controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
        print('model_real completed')
    else:
        print(f'[INFO] found pics in [{pic_path}], skip')

    def quant_or_load(model_target_path, target_config):
        if not os.path.exists(model_target_path):
            print(f'[INFO] {model_target_path} not found, start quantizing...')

            model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
            target_model = quant_model(model, 'dit', target_config, dataset, calib_args).to('cpu')

            os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
            torch.save(target_model.transformer, model_target_path)
            print(f'[INFO] Model quantizing ok, saved to {model_target_path}')
        else:
            print(f'[INFO] {model_target_path} found, start loading...')
            target_model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')
            target_model.transformer = torch.load(model_target_path, weights_only=False)
        
        return target_model

    ### 1
    model_quant_path = os.path.join(root_dir, 'model/dit/model_quant.pt')
    model_quant = quant_or_load(model_quant_path, quant_config)
    model_quant = model_quant.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant, os.path.join(root_dir, 'pics/quant'), dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant
    print('model_quant completed')


    ### 2
    quant_config_with_seg = {
        "default": {
            "enable": True,
            "input_dtype": DType.INT8,
            "weight_dtype": DType.INT8,
            "opt": Optimum.SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.seg(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }
    model_quant_seg_path = os.path.join(root_dir, 'model/dit/model_quant_seg.pt')
    model_quant_seg = quant_or_load(model_quant_seg_path, quant_config_with_seg)
    model_quant_seg = model_quant_seg.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant_seg, os.path.join(root_dir, 'pics/quant_seg'), dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant_seg
    print('model_quant_seg completed')

def run_dual_scale_module():
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    from segquant.config import SegPattern

    root_dir = 'benchmark_record/run_dual_scale_module'
    os.makedirs(root_dir, exist_ok=True)
    
    dataset = COCODataset(path='../dataset/controlnet_datasets/coco_canny', cache_size=16)

    max_timestep = 50
    max_num = 1024

    # ### 0
    # model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')    
    # model_real = model_real.to('cuda')
    # latents = torch.load('../latents.pt')
    # pic_path = os.path.join(root_dir, 'pics/real')
    # if not os.path.exists(pic_path):
    #     print(f'[INFO] generating pics in [{pic_path}]...')
    #     trace_pic(model_real, pic_path, dataset.get_dataloader(), latents, max_num=max_num, 
    #             controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    #     print('model_real completed')
    # else:
    #     print(f'[INFO] found pics in [{pic_path}], skip')

    def quant_or_load(model_target_path, target_config):
        if not os.path.exists(model_target_path):
            print(f'[INFO] {model_target_path} not found, start quantizing...')

            model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')
            target_model = quant_model(model, 'dit', target_config, dataset, calib_args).to('cpu')

            os.makedirs(os.path.dirname(model_target_path), exist_ok=True)
            torch.save(target_model.transformer, model_target_path)
            print(f'[INFO] Model quantizing ok, saved to {model_target_path}')
        else:
            print(f'[INFO] {model_target_path} found, start loading...')
            target_model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')
            target_model.transformer = torch.load(model_target_path, weights_only=False)
        
        return target_model

    # ### 1
    # model_quant_path = os.path.join(root_dir, 'model/dit/model_quant.pt')
    # model_quant = quant_or_load(model_quant_path, quant_config)
    # model_quant = model_quant.to('cuda')
    # latents = torch.load('../latents.pt')
    # trace_pic(model_quant, os.path.join(root_dir, 'pics/quant'), dataset.get_dataloader(), latents, max_num=max_num, 
    #           controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    # del model_quant
    # print('model_quant completed')


    ### 2
    quant_config_with_dual_scale = {
        "default": {
            "enable": True,
            "input_dtype": DType.INT8,
            "weight_dtype": DType.INT8,
            "opt": Optimum.SMOOTH,
            "seglinear": True,
            'search_patterns': [SegPattern.Activation2Linear],
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }
    model_quant_dual_scale_path = os.path.join(root_dir, 'model/dit/model_quant_dual_scale.pt')
    model_quant_dual_scale = quant_or_load(model_quant_dual_scale_path, quant_config_with_dual_scale)
    model_quant_dual_scale = model_quant_dual_scale.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant_dual_scale, os.path.join(root_dir, 'pics/quant_dual_scale'), dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"], guidance_scale=calib_args["guidance_scale"], num_inference_steps=max_timestep)
    del model_quant_dual_scale
    print('model_quant_dual_scale completed')

if __name__ == '__main__':
    run_dual_scale_module()