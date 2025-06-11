import os
import json
import torch
from tqdm import tqdm
import lpips
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from benchmark.config import QUANT_METHOD_CHOICES,DATASET_TYPE_CHOICES

lpips_model = lpips.LPIPS(net="alex")


def calculate_fid(real_dir, fake_dir, batch_size=50, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir], batch_size=batch_size, device=device, dims=2048
    )
    return fid_value


def calculate_lpips(img1, img2):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    img1 = transform(img1).unsqueeze(0) * 2 - 1
    img2 = transform(img2).unsqueeze(0) * 2 - 1

    lpips_score = lpips_model(img1, img2)
    return lpips_score.item()


def calculate_psnr(img1, img2):
    transform = transforms.ToTensor()
    img1 = transform(img1).numpy()
    img2 = transform(img2).numpy()

    return psnr(img1, img2, data_range=img1.max() - img1.min())


def calculate_ssim(img1, img2):
    img1 = np.array(img1.convert("L"))
    img2 = np.array(img2.convert("L"))

    return ssim(img1, img2, data_range=img1.max() - img1.min())


def generate_metric(path1, path2):
    print(path1, path2)
    fid = calculate_fid(path1, path2)
    names1 = set(os.listdir(path1))
    names2 = set(os.listdir(path2))
    common_names = sorted(names1 & names2)
    lpips_scores, psnr_scores, ssim_scores = [], [], []
    for name in tqdm(common_names, desc="Calculating metrics"):
        img1_path = os.path.join(path1, name)
        img2_path = os.path.join(path2, name)

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

        lpips_scores.append(calculate_lpips(img1, img2))
        psnr_scores.append(calculate_psnr(img1, img2))
        ssim_scores.append(calculate_ssim(img1, img2))
    result = {
        "FID": fid,
        "LPIPS": np.mean(lpips_scores) if lpips_scores else None,
        "PSNR": np.mean(psnr_scores) if psnr_scores else None,
        "SSIM": np.mean(ssim_scores) if ssim_scores else None,
    }

    return result


if __name__ == "__main__":
    result = json.load(open("../segquant/benchmark_record/metric.json"))
    for dataset in DATASET_TYPE_CHOICES:
        for method in QUANT_METHOD_CHOICES:
            real_dir = f"../segquant/benchmark_record/{dataset}/run_real_module/pics/real"
            quant_dir = f"../segquant/benchmark_record/{dataset}/run_{method}_module/pics/quant_{method}"
            if os.path.exists(real_dir) and os.path.exists(quant_dir):
                if f"{dataset}_{method}" not in result:
                    print(f"Updating {dataset}_{method} to metric.json")
                    result[f"{dataset}_{method}"] = generate_metric(real_dir, quant_dir)
                else:
                    print(f"{dataset}_{method} already in metric.json")
    print(result)
    with open("../segquant/benchmark_record/metric.json", "w") as f:
        json.dump(result, f)
        print("Successfully saved json file at ../segquant/benchmark_record/metric.json")
