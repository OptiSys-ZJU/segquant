from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
from stable_diff.model.transformer_sd3 import SD3Transformer2DModel
from stable_diff.model.scheduler import FlowMatchEulerDiscreteScheduler
from stable_diff.model.autoencoder_kl import AutoencoderKL
from stable_diff.model.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from stable_diff.utils import load_image

from stable_diff.model.onnx.controlnet_sd3 import SD3ControlNetONNXModel
import torch
import modelopt.torch.quantization as mtq
from pathlib import Path
import os
import glob
import copy

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


type_dict = {
    "canny": ('SD3-Controlnet-Canny', 
              "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg",
              'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image',
              'NSFW, nude, naked, porn, ugly'
              ),
    "tile": ('SD3-Controlnet-Tile', 
             'https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg',
             'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image',
             'NSFW, nude, naked, porn, ugly'
             ),
    "pose": ('SD3-Controlnet-Pose', 
             "https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg",
             'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image',
             'NSFW, nude, naked, porn, ugly'
             ),
    "depth": ('SD3-Controlnet-Depth', 
              "https://huggingface.co/InstantX/SD3-Controlnet-Depth/resolve/main/images/depth.jpeg",
              "a panda cub, captured in a close-up, in forest, is perched on a tree trunk. good composition, Photography, the cub's ears, a fluffy black, are tucked behind its head, adding a touch of whimsy to its appearance. a lush tapestry of green leaves in the background. depth of field, National Geographic",
             "bad hands, blurry, NSFW, nude, naked, porn, ugly, bad quality, worst quality"
             ),
}


def infer(prefix, model, type, num_inference_steps=1, scale=0.5, latents=None):
    # load pipeline
    _, file, prompt, n_prompt = type_dict[type]
    control_image = load_image(file)
    image = model.forward(
        fake_controlnet = False,
        fake_controlnet_pt = f'new_output_int8smooth_{scale}_28',
        dump_tensor = True,
        dump_prefix = f'{prefix}_{type}_{num_inference_steps}_{scale}',
        prompt = prompt, 
        negative_prompt=n_prompt, 
        control_image=control_image, 
        controlnet_conditioning_scale=scale,
        num_inference_steps=num_inference_steps,
        latents=latents,
    )[0]

    dir_path = Path(f"pic/{prefix}/{scale}")
    dir_path.mkdir(parents=True, exist_ok=True)
    image[0].save(f'pic/{prefix}/{scale}/{prefix}-{type}-{num_inference_steps}-{scale}.jpg')
    print("=============================")
    return image[0]

###########################################################################
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

def quant(model):
    # config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)    
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)  

    # config["quant_cfg"]["*timestep_embedder*"] = {"enable": False}
    # config["quant_cfg"]["*transformer_blocks.0.norm1.linear*"] = {"enable": False}
    config["quant_cfg"]["*transformer_blocks.*.norm1.linear*"] = {"enable": False}
    # config["quant_cfg"]["*controlnet_blocks*"] = {"enable": False}

    model = mtq.quantize(control_net, config, forward_loop)
    mtq.print_quant_summary(model)
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = CLIPTextModelWithProjection.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer')
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder_2')
    tokenizer_2 = CLIPTokenizer.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer_2')
    text_encoder_3 = T5EncoderModel.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder_3')
    tokenizer_3 = T5TokenizerFast.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer_3')
    scheduler = FlowMatchEulerDiscreteScheduler.from_config('stable_diff/configs/scheduler.json')

    # control_net = SD3ControlNetONNXModel.from_config('stable_diff/configs/controlnet.json', 'controlnet_int8_default.onnx')
    # control_net = SD3ControlNetONNXModel.from_config('stable_diff/configs/controlnet.json', 'controlnet_int8_smoothquant.onnx')
    # control_net = SD3ControlNetONNXModel.from_config('stable_diff/configs/controlnet.json', 'controlnet_int8_smoothquant_new.onnx')
    # control_net = SD3ControlNetONNXModel.from_config('stable_diff/configs/controlnet.json', 'controlnet_int8_smoothquant_ver3.onnx')
    control_net = SD3ControlNetModel.from_config('stable_diff/configs/controlnet.json', 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)
    control_net = quant(control_net)

    vae = AutoencoderKL.from_config('stable_diff/configs/vae.json', 'stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors').half().to(device)
    dit = SD3Transformer2DModel.from_config('stable_diff/configs/transformer.json', 'stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors').half().to(device)
    sd3 = StableDiffusion3ControlNetModel(dit, scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3, control_net).half().to(device)

    #######################
    type = 'int8smooth_ver3'
    latents = torch.load('latents.pt')
    if os.path.exists(f"pic/{type}"):
        raise FileExistsError(f"Error: Directory 'pic/{type}' already exists!")
    for scale in [0.2, 0.5, 0.8]:
        for i in range(1, 29, 3):
            infer(type, sd3, 'canny', i, scale, latents)
