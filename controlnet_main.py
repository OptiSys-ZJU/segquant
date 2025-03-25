from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
from stable_diff.model.onnx.controlnet_sd3 import SD3ControlNetONNXModel
import torch
import glob
import os

def load_controlnet_inputs(folder_path):
    file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_input_fp16_canny_0.9_28_0.pt")))
    tensors = [torch.load(path) for path in file_paths]

    res = []
    for data in tensors:
        res.append((data['hidden_states'], data['timestep'], data['encoder_hidden_states'], data['pooled_projections'], data['controlnet_cond'], data['conditioning_scale']))
    return res

if __name__ == '__main__':
    config_path = 'stable_diff/configs/controlnet.json'
    safetensor_path = 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # control_net = SD3ControlNetModel.from_config(config_path, 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)
    control_net = SD3ControlNetONNXModel.from_config(config_path, 'controlnet_int8_default.onnx')
    
    ###############################################
    data_loader = load_controlnet_inputs('calibration')

    for hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale in data_loader:
        res = control_net(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)

        torch.save(res[0], 'controlnet_output_canny_1_0.8.pt')