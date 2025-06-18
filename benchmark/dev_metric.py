import os
import json
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim   
from benchmark.config import QUANT_METHOD_CHOICES,DATASET_TYPE_CHOICES
from benchmark import ImageReward as RM
from torchvision import transforms



def generate_ir_metric(path, prompt_path):
    names = os.listdir(path)
    ir_scores = []
    metainfo = json.load(open(prompt_path))
    model = RM.load("ImageReward-v1.0")
    for name in tqdm(names, desc="Calculating IR metrics"):
        id = int(name.split(".")[0])
        img_path = os.path.join(path, name)
        prompt = metainfo[id]["prompt"]
        with torch.inference_mode():
            score = model.score(prompt, img_path)
        ir_scores.append(score)
    print(np.mean(ir_scores))

def generate_clip_metric(path, prompt_path, metric_name):
    names = os.listdir(path)
    metainfo = json.load(open(prompt_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if metric_name == "clip_iqa":
        metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    elif metric_name == "clip_score":
        metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    else:
        raise NotImplementedError(f"Metric {metric_name} not implemented")
    
    for name in tqdm(names, desc=f"Calculating {metric_name} metrics"):
        id = int(name.split(".")[0])
        img_path = os.path.join(path, name)
        pil_img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(pil_img).to(device)
        prompt = metainfo[id]["prompt"]
        if metric_name == "clip_iqa":
            metric.update(img.to(torch.float32).unsqueeze(0))
        else:
            metric.update(img.to(torch.float32).unsqueeze(0), prompt)

    print(metric.compute().mean().item())


if __name__ == "__main__":

    quant_pic_dir = f"../segquant/benchmark_record/COCO/run_int8smooth_module/pics/quant_int8smooth"
    quant_prompt_dir = f"../dataset/controlnet_datasets/COCO-Caption2017-canny/metadata.json"
    generate_ir_metric(quant_pic_dir, quant_prompt_dir)
    generate_clip_metric(quant_pic_dir, quant_prompt_dir, "clip_iqa")
    generate_clip_metric(quant_pic_dir, quant_prompt_dir, "clip_score")

