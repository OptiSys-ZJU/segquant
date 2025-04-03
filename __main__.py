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

from stable_diff.utils.hook_dump import DebugContext


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


def infer(prefix, model, type, num_inference_steps=1, scale=0.5, latents=None, enable_save=True, enable_res=False, single_step_sim=False):
    # load pipeline
    DebugContext.set_prefix(prefix, type, str(scale))

    _, file, prompt, n_prompt = type_dict[type]
    control_image = load_image(file)
    image = model.forward(
        enable_res = enable_res,
        single_step_sim = single_step_sim,
        dump_tensor = False,
        prompt = prompt, 
        negative_prompt=n_prompt, 
        control_image=control_image, 
        controlnet_conditioning_scale=scale,
        num_inference_steps=num_inference_steps,
        latents=latents,
    )[0]
    if enable_save:
        dir_path = Path(f"pic/{prefix}/{scale}")
        dir_path.mkdir(parents=True, exist_ok=True)
        image[0].save(f'pic/{prefix}/{scale}/{prefix}-{type}-{num_inference_steps}-{scale}.jpg')
    print("=============================")
    return image[0]

###########################################################################
def load_controlnet_inputs(folder_path, controlnet_type):
    file_paths = sorted(glob.glob(os.path.join(folder_path, f"controlnet_input_fp16_{controlnet_type}_*.pt")))
    tensors = [torch.load(path) for path in file_paths]

    res = []
    for data in tensors:
        res.append((data['hidden_states'], data['timestep'], data['encoder_hidden_states'], data['pooled_projections'], data['controlnet_cond'], data['conditioning_scale']))
    return res

# controlnet_type = 'depth'
controlnet_type = 'canny'
# controlnet_type = 'tile'
# controlnet_type = 'pose'

data_loader = load_controlnet_inputs('calibration', controlnet_type)

def forward_loop(model: SD3ControlNetModel):
    for hidden_states, timestep, encoder_hidden_states, pooled_projections, controlnet_cond, conditioning_scale in data_loader:
        model.forward(hidden_states=hidden_states, 
              timestep=timestep, 
              encoder_hidden_states=encoder_hidden_states, 
              pooled_projections=pooled_projections, 
              controlnet_cond=controlnet_cond, 
              conditioning_scale=conditioning_scale)

def quant(model, config, fake_quant):
    config["quant_cfg"]["*input_quantizer"]["fake_quant"] = fake_quant
    config["quant_cfg"]["*weight_quantizer"]["fake_quant"] = fake_quant
    print(config)
    model = mtq.quantize(model, config, forward_loop)
    mtq.print_quant_summary(model)
    # exit(0)
    return model

def get_controlnet(type, fake_quant):
    repo = type_dict[controlnet_type][0]
    
    control_net = SD3ControlNetModel.from_config('stable_diff/configs/controlnet.json', f'{repo}/diffusion_pytorch_model.safetensors').half().to(device)

    if type == 'fp16':
        return control_net
    elif type == 'fp8':
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)  
        return quant(control_net, config, fake_quant)
    elif type == 'int8_default':
        config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)  
        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)  
        config["quant_cfg"]["*pos_embed.proj*"] = {"enable": False}
        config["quant_cfg"]["*pos_embed_input.proj*"] = {"enable": False}
        config["quant_cfg"]["*context_embedder.proj*"] = {"enable": False}

        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth_enablelatent':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        config["quant_cfg"]["*pos_embed.proj*"] = {"enable": False}
        config["quant_cfg"]["*pos_embed_input.proj*"] = {"enable": False}
        config["quant_cfg"]["*context_embedder.proj*"] = {"enable": False}

        config["quant_cfg"]["*time_text_embed*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.norm1.linear*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.norm1_context.linear*"] = {"enable": False}
        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth_enabletime':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        config["quant_cfg"]["*pos_embed.proj*"] = {"enable": False}
        config["quant_cfg"]["*pos_embed_input.proj*"] = {"enable": False}
        config["quant_cfg"]["*context_embedder.proj*"] = {"enable": False}

        config["quant_cfg"]["*transformer_blocks.*.attn*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.ff.net*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.ff_context.net*"] = {"enable": False}
        config["quant_cfg"]["*controlnet_blocks*"] = {"enable": False}
        
        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth_enabletime_block':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        config["quant_cfg"]["*pos_embed.proj*"] = {"enable": False}
        config["quant_cfg"]["*pos_embed_input.proj*"] = {"enable": False}
        config["quant_cfg"]["*context_embedder.proj*"] = {"enable": False}

        config["quant_cfg"]["*transformer_blocks.*.attn*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.ff.net*"] = {"enable": False}
        config["quant_cfg"]["*transformer_blocks.*.ff_context.net*"] = {"enable": False}
        config["quant_cfg"]["*controlnet_blocks*"] = {"enable": False}

        config["quant_cfg"]["*transformer_blocks.*.norm1.linear.weight_quantizer"] = {"num_bits": 8, "block_sizes": {0: 1536}, "enable": True, 'fake_quant': fake_quant}
        
        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth_block' or type == 'int8_smooth_blockres':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)  
        config["quant_cfg"]["*transformer_blocks.*.norm1.linear.weight_quantizer"] = {"num_bits": 8, "block_sizes": {0: 1536}, "enable": True, 'fake_quant': fake_quant}
        return quant(control_net, config, fake_quant)
    elif type == 'int8_smooth_disnorm1':
        config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)  
        config["quant_cfg"]["*transformer_blocks.0.norm1.linear*"] = {"enable": False}
        return quant(control_net, config, fake_quant)
    elif type == 'fp8_disnorm1':
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)  
        config["quant_cfg"]["*transformer_blocks.0.norm1.linear*"] = {"enable": False}
        return quant(control_net, config, fake_quant)
    elif type == 'fp8_block':
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)  
        config["quant_cfg"]["*transformer_blocks.*.norm1.linear.weight_quantizer"] = {"num_bits": (4, 3), "block_sizes": {0: 1536}, "enable": True, 'fake_quant': fake_quant}
        return quant(control_net, config, fake_quant)
    else:
        raise ValueError(f'Unsupported Type {type}')

if __name__ == '__main__':
    enable_res = False
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
    
    type = 'fp16'
    # type = 'int8_smooth'
    # type = 'int8_smooth_enablelatent'
    # type = 'int8_smooth_enabletime'
    # type = 'int8_smooth_enabletime_block'
    # type = 'int8_smooth_block'
    # type = 'int8_smooth_blockres'
    # type = 'int8_default'
    # type = 'int8_smooth_disnorm1'
    # type = 'fp8'
    # type = 'fp8_disnorm1'
    # type = 'fp8_block'
    # enable_save = True
    enable_save = True
    # fake_quant = False
    fake_quant = True

    if type == 'int8_smooth_blockres':
        enable_res = True

    print('controlnet_type', controlnet_type, 'Type', type, 'enable_save', enable_save, 'fake_quant', fake_quant, 'enable_res', enable_res)

    control_net = get_controlnet(type, fake_quant)

    vae = AutoencoderKL.from_config('stable_diff/configs/vae.json', 'stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors').half().to(device)
    dit = SD3Transformer2DModel.from_config('stable_diff/configs/transformer.json', 'stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors').half().to(device)
    sd3 = StableDiffusion3ControlNetModel(dit, scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3, control_net).half().to(device)

    #######################
    latents = torch.load('latents.pt')
    if enable_save and os.path.exists(f"pic/{type}"):
        raise FileExistsError(f"Error: Directory 'pic/{type}' already exists!")
    # for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for scale in [0.8]:
        for i in [1]:
        # for i in range(1, 29, 1):
            infer(type, sd3, controlnet_type, i, scale, latents, enable_save, enable_res, single_step_sim=False)
    
    if type != 'fp16':
        for scale in [0.8]:
            for i in [200]:
            # for i in range(1, 29, 1):
                infer(type, sd3, controlnet_type, i, scale, latents, enable_save, enable_res, single_step_sim=True)
