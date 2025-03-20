from stable_diff.model.onnx.controlnet_sd3 import SD3ControlNetONNXModel
from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
import torch
import glob
import os

def load_controlnet_inputs(folder_path):
    file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_input_*_*_*_28_*.pt")))
    # file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_input_fp16_canny_0.8_10_0.pt")))
    tensors = [torch.load(path) for path in file_paths]
    file_names = [os.path.basename(path) for path in file_paths]

    res = []
    for data in tensors:
        res.append((data['hidden_states'], data['timestep'], data['encoder_hidden_states'], data['pooled_projections'], data['controlnet_cond'], data['conditioning_scale']))
    
    return res, file_names

if __name__ == '__main__':
    config_path = 'stable_diff/configs/controlnet.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # control_net = SD3ControlNetONNXModel.from_config(config_path, 'controlnet_int8_default.onnx')
    control_net = SD3ControlNetONNXModel.from_config(config_path, 'controlnet_int8_smoothquant.onnx')
    # control_net = SD3ControlNetModel.from_config(config_path, 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)
    
    ###############################################
    data_list, filenames = load_controlnet_inputs('latent_dump_input/controlnet')

    for data, filename in zip(data_list, filenames):
        hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale = data
        res = control_net(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)
        torch.save(res, filename.replace('input', 'output').replace('fp16', 'int8smooth'))