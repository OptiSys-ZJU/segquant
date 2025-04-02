from stable_diff.model.controlnet_sd3 import SD3ControlNetModel
from stable_diff.model.transformer_sd3 import SD3Transformer2DModel
from stable_diff.model.scheduler import FlowMatchEulerDiscreteScheduler
from stable_diff.model.autoencoder_kl import AutoencoderKL
from stable_diff.model.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from stable_diff.utils import load_image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torch

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


def record_calibration(prefix, model, type, num_inference_steps=1, scale=0.5):
    # load pipeline
    _, file, prompt, n_prompt = type_dict[type]
    control_image = load_image(file)
    model.forward(
        dump_input_tensor = True,
        dump_prefix = f'{prefix}_{type}_{scale}_{num_inference_steps}',
        prompt = prompt, 
        negative_prompt=n_prompt, 
        control_image=control_image, 
        controlnet_conditioning_scale=scale,
        num_inference_steps=num_inference_steps,
    )
    print("=============================")

def record_latent_input(prefix, model, type, num_inference_steps=1, scale=0.5, latents=None):
    _, file, prompt, n_prompt = type_dict[type]
    control_image = load_image(file)
    model.forward(
        fake_controlnet = False,
        fake_controlnet_pt = f'controlnet_output_{type}_{num_inference_steps}_{scale}.pt',
        dump_tensor = True,
        dump_prefix = f'{prefix}_{type}_{scale}_{num_inference_steps}',
        prompt = prompt, 
        negative_prompt=n_prompt, 
        control_image=control_image, 
        controlnet_conditioning_scale=scale,
        num_inference_steps=num_inference_steps,
        latents=latents,
    )
    print("=============================")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = CLIPTextModelWithProjection.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer')
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder_2')
    tokenizer_2 = CLIPTokenizer.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer_2')
    text_encoder_3 = T5EncoderModel.from_pretrained('stable-diffusion-3-medium-diffusers/text_encoder_3')
    tokenizer_3 = T5TokenizerFast.from_pretrained('stable-diffusion-3-medium-diffusers/tokenizer_3')
    scheduler = FlowMatchEulerDiscreteScheduler.from_config('stable_diff/configs/scheduler.json')
    control_net = SD3ControlNetModel.from_config('stable_diff/configs/controlnet.json', 'SD3-Controlnet-Canny/diffusion_pytorch_model.safetensors').half().to(device)

    vae = AutoencoderKL.from_config('stable_diff/configs/vae.json', 'stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors').half().to(device)
    dit = SD3Transformer2DModel.from_config('stable_diff/configs/transformer.json', 'stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors').half().to(device)
    sd3 = StableDiffusion3ControlNetModel(dit, scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3, control_net).half().to(device)

    #######################
    latents = torch.load('latents.pt')
    for scale in [0.2, 0.5, 0.8]:
        for i in [28]:
            record_calibration('fp16', sd3, 'tile', i, scale)
        # record_latent_input('fp16-new', sd3, 'canny', 28, scale, latents)
