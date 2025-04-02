import torch
import os
import glob
import copy

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import QuantModuleRegistry

from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
from stable_diff.model.embeddings import TimestepEmbedding

def load_controlnet_inputs(folder_path):
    file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_input*.pt")))
    tensors = [torch.load(path) for path in file_paths]

    res = []
    for data in tensors:
        res.append((data['hidden_states'], data['timestep'], data['encoder_hidden_states'], data['pooled_projections'], data['controlnet_cond'], data['conditioning_scale']))
    return res


data_loader = load_controlnet_inputs('calibration')

def forward_loop(model: SD3ControlNetModel):
    for hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale in data_loader:
        model.forward(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control_net = SD3ControlNetModel.from_config('stable_diff/configs/controlnet.json', 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)

    # config = mtq.INT8_DEFAULT_CFG
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    config["quant_cfg"]["*transformer_blocks.*.norm1.linear.weight_quantizer"] = {"num_bits": 8, "block_sizes": {0: 1536}, "enable": True, 'fake_quant': True}
    model = mtq.quantize(control_net, config, forward_loop)

    mtq.print_quant_summary(model)

    hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale = data_loader[0]
    torch.onnx.export(model, (
        hidden_states,
        controlnet_cond,
        conditioning_scale,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        None,
    ), 'int8smooth-block.onnx')