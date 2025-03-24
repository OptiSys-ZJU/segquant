from stable_diff.model.onnx.controlnet_sd3 import SD3ControlNetONNXModel
from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
import torch
import glob
import os
import modelopt.torch.quantization as mtq
from pathlib import Path
import copy

def load_controlnet_inputs(folder_path, regex):
    file_paths = sorted(glob.glob(os.path.join(folder_path, regex)))
    # file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_input_fp16_canny_0.9_28_0.pt")))
    tensors = [torch.load(path) for path in file_paths]
    file_names = [os.path.basename(path) for path in file_paths]

    res = []
    for data in tensors:
        res.append((data['hidden_states'], data['timestep'], data['encoder_hidden_states'], data['pooled_projections'], data['controlnet_cond'], data['conditioning_scale']))
    
    return res, file_names


#################################################
data_loader, _ = load_controlnet_inputs('calibration', "controlnet_input*.pt")

def forward_loop(model: SD3ControlNetModel):
    for hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale in data_loader:
        model.forward(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)

def quant(model):
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)    
    model = mtq.quantize(control_net, config, forward_loop)
    mtq.print_quant_summary(model)
    return model

if __name__ == '__main__':
    config_path = 'stable_diff/configs/controlnet.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # control_net = SD3ControlNetONNXModel.from_config(config_path, 'controlnet_int8_default.onnx')
    # control_net = SD3ControlNetONNXModel.from_config(config_path, 'controlnet_int8_smoothquant_new.onnx')
    control_net = SD3ControlNetModel.from_config(config_path, 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)
    control_net = quant(control_net)


    ###############################################
    data_list, filenames = load_controlnet_inputs('latent_dump_input/controlnet', 'controlnet_input_fp16_*_*_28_*.pt')

    for data, filename in zip(data_list, filenames):
        hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale = data
        res = control_net(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)
        torch.save(res, os.path.join('latent_dump_output', 'controlnet', filename.replace('input', 'output').replace('fp16', 'fp8')))