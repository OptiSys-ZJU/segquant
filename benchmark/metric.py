import argparse
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
from benchmark import ImageReward as RM
import datasets

from benchmark.utils import get_dataset_prompt_metadata_file

lpips_model = lpips.LPIPS(net="alex")
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

def calculate_fid(real_dir, fake_dir, batch_size=50, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir], batch_size=batch_size, device=device, dims=2048, num_workers=64,
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

def generate_metric(path1, path2, prompt_meta):
    print('real:', path1, 'quant:', path2)
    print('prompt_meta:', prompt_meta)
    fid = calculate_fid(path1, path2)
    names1 = os.listdir(path1)
    names2 = os.listdir(path2)
    common_names = sorted(set(names1) & set(names2))
    print('total images:', len(common_names))
    with open(prompt_meta, 'r') as f:
        metainfo = json.load(f)
    # setting up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # quality metrics
    ir_scores = []
    metric_iqa = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    metric_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    # similarity metrics
    lpips_scores, psnr_scores, ssim_scores = [], [], []

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
    parser = argparse.ArgumentParser(description='Metrics for Diffusion Models')
    parser.add_argument('-d', '--dataset-type', type=str, default='COCO', choices=['COCO', 'MJHQ', 'DCI'], help='Type of the dataset to use')
    parser.add_argument('-m', '--model-type', type=str, default='sd3', choices=['flux', 'sd3', 'sdxl'], help='Type of the model to benchmark')
    
    parser.add_argument('-l', '--layer-type', type=str, default='dit', choices=['dit', 'controlnet', 'unet'], help='Type of the layer to benchmark')
    parser.add_argument('-q', '--quant-type', type=str, default='int8w8a8', help='Type of the quant to benchmark')
    parser.add_argument('-e', '--exp-name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('-a', '--affine-dir', type=str, default=None, help='Path to the affine directory')

    parser.add_argument('-r', '--root-dir', type=str, default='../benchmark_results', help='Root directory for benchmark results')
    parser.add_argument('--dataset-root', type=str, default='../dataset/controlnet_datasets', help='Root directory for datasets')

    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_type = args.model_type

    layer_type = args.layer_type
    quant_type = args.quant_type
    exp_name = args.exp_name
    affine_dir = args.affine_dir

    root_dir = args.root_dir
    dataset_root_dir = args.dataset_root

    # check real dir
    real_dir = os.path.join(root_dir, dataset_type, model_type, "real")
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Real directory {real_dir} does not exist.")
    # check quant dir
    exp_all_name = f'{model_type}-{layer_type}-{quant_type}-{exp_name}'
    if affine_dir is not None:
        quant_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name, affine_dir)
    else:
        quant_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)
    if not os.path.exists(quant_dir):
        raise FileNotFoundError(f"Quant directory {quant_dir} does not exist.")
    
    # check prompt metadata file
    metadata_file = get_dataset_prompt_metadata_file(dataset_type, dataset_root_dir)
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    
    # generate metrics
    res = generate_metric(os.path.join(real_dir, 'pic'), os.path.join(quant_dir, 'pic'), metadata_file)
    
    dump_file = os.path.join(quant_dir, 'metric.json')
    with open(dump_file, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Metrics saved to {dump_file}")
    print("Metrics:", res)
