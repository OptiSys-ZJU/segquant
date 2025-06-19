import os
import json
from tqdm import tqdm
import lpips
import numpy as np
import torch
import torchmetrics
import torchvision
from torchvision import transforms
from torch.utils import data
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from benchmark.config import QUANT_METHOD_CHOICES,DATASET_TYPE_CHOICES,BenchmarkConfig
from benchmark import ImageReward as RM
import datasets


class PromptImageDataset(data.Dataset):
    def __init__(self, ref_dataset: datasets.Dataset, gen_dirpath: str):
        super(data.Dataset, self).__init__()
        self.ref_dataset, self.gen_dirpath = ref_dataset, gen_dirpath
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_dataset)

    def __getitem__(self, idx: int):
        row = self.ref_dataset[idx]
        gen_image = Image.open(os.path.join(self.gen_dirpath, row["filename"] + ".png")).convert("RGB")
        gen_tensor = torch.from_numpy(np.array(gen_image)).permute(2, 0, 1)
        prompt = row["prompt"]
        return [gen_tensor, prompt]


def compute_image_multimodal_metrics(
    ref_dataset: datasets.Dataset,
    gen_dirpath: str,
    metrics: tuple[str, ...] = ("clip_iqa", "clip_score"),
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    metric_names = metrics
    metrics: dict[str, torchmetrics.Metric] = {}
    for metric_name in metric_names:
        if metric_name == "clip_iqa":
            metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        elif metric_name == "clip_score":
            metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not implemented")
        metrics[metric_name] = metric
    dataset = PromptImageDataset(ref_dataset, gen_dirpath)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"{ref_dataset.config_name} multimodal metrics")):
            batch[0] = batch[0].to(device)
            for metric_name, metric in metrics.items():
                if metric_name == "clip_iqa":
                    metric.update(batch[0].to(torch.float32))
                else:
                    prompts = list(batch[1])
                    metric.update(batch[0], prompts)
    result = {metric_name: metric.compute().mean().item() for metric_name, metric in metrics.items()}
    return result


def compute_image_reward(
    ref_dataset: datasets.Dataset,
    gen_dirpath: str,
) -> dict[str, float]:
    scores = []
    model = RM.load("ImageReward-v1.0")
    for batch in tqdm(
        ref_dataset.iter(batch_size=1, drop_last_batch=False),
        desc=f"{ref_dataset.config_name} image reward",
        total=len(ref_dataset),
        dynamic_ncols=True,
    ):
        filename = batch["filename"][0]
        path = os.path.join(gen_dirpath, f"{filename}.png")
        prompt = batch["prompt"][0]
        with torch.inference_mode():
            score = model.score(prompt, path)
        scores.append(score)
    result = {"image_reward": sum(scores) / len(scores)}
    return result




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

def calculate_ir(img_path, prompt):
    model = RM.load("ImageReward-v1.0")
    with torch.inference_mode():
        score = model.score(prompt, img_path)
    return score

def generate_metric(path1, path2, prompt_dir):
    print(path1, path2)
    fid = calculate_fid(path1, path2)
    names1 = os.listdir(path1)
    names2 = os.listdir(path2)
    common_names = sorted(set(names1) & set(names2))
    metainfo = json.load(open(prompt_dir))
    # setting up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # quality metrics
    ir_scores = []
    metric_iqa = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    metric_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    # similarity metrics
    lpips_scores, psnr_scores, ssim_scores = [], [], []

    debug_count = 5
    for name in tqdm(common_names, desc="Calculating metrics"):
        img1_path = os.path.join(path1, name)
        img2_path = os.path.join(path2, name)
        id = int(name.split(".")[0])
        prompt = metainfo[id]["prompt"]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img2).to(device)

        ir_scores.append(calculate_ir(img2_path, prompt))
        lpips_scores.append(calculate_lpips(img1, img2))
        psnr_scores.append(calculate_psnr(img1, img2))
        ssim_scores.append(calculate_ssim(img1, img2))
        metric_iqa.update(img_tensor.to(torch.float32).unsqueeze(0))
        metric_score.update(img_tensor.to(torch.float32).unsqueeze(0), prompt)
        # TODO: only for debug, remove this later
        debug_count -= 1
        if debug_count == 0:
            break


    result = {
        "FID": fid,
        "LPIPS": np.mean(lpips_scores) if lpips_scores else None,
        "PSNR": np.mean(psnr_scores) if psnr_scores else None,
        "SSIM": np.mean(ssim_scores) if ssim_scores else None,
        "IR": np.mean(ir_scores) if ir_scores else None,
        "CLIP_IQ": metric_iqa.compute().mean().item(),
        "CLIP_SCORE": metric_score.compute().mean().item(),
    }

    return result


if __name__ == "__main__":
    # create a new metric.json file if not exists
    if not os.path.exists("../segquant/benchmark_record/metric.json"):
        with open("../segquant/benchmark_record/metric.json", "w") as f:
            json.dump({}, f)

    result = json.load(open("../segquant/benchmark_record/metric.json"))
    for dataset in DATASET_TYPE_CHOICES:
        for method in QUANT_METHOD_CHOICES:
            real_dir = f"../segquant/benchmark_record/{dataset}/run_real_module/pics/real"
            quant_dir = f"../segquant/benchmark_record/{dataset}/run_{method}_module/pics/quant_{method}"
            prompt_dir = os.path.join(BenchmarkConfig.DATASET_PATH[dataset], "metadata.json")
            if os.path.exists(real_dir) and os.path.exists(quant_dir):
                if f"{dataset}_{method}" not in result:
                    print(f"Updating {dataset}_{method} to metric.json")
                    result[f"{dataset}_{method}"] = generate_metric(real_dir, quant_dir, prompt_dir)
                    # save immediately
                    with open("../segquant/benchmark_record/metric.json", "w") as f:
                        json.dump(result, f)
                else:
                    print(f"{dataset}_{method} already in metric.json")
    print(result)
    # with open("../segquant/benchmark_record/metric.json", "w") as f:
    #     json.dump(result, f)
    #     print("Successfully saved json file at ../segquant/benchmark_record/metric.json")
