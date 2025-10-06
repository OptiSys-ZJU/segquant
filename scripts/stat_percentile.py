import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from benchmark import trace_pic
from benchmark.utils import get_calibrate_data, get_dataset, get_full_model_with_quantized_part, get_latents, get_part_model
from segquant.torch.quantization import _move_to_device

from segquant.layers.givens_orthogonal import GivensOrthogonal


def buildQ(pairs, vecs, k, dtype=torch.float32, device=None):
    cs = torch.empty((len(vecs), 2), device=device, dtype=dtype)
    GivensOrthogonal.cal_cs_inplace(cs, vecs)
    Q = GivensOrthogonal.build_Q(k, pairs, cs, dtype=dtype, device=device)
    return Q


def get_all_linears(model):
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module
    return linears

def attach_input_hooks(linears):
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            inp = input[0].detach()
            if name not in activations:
                activations[name] = []
            activations[name].append(inp)

        return hook

    hooks = []
    for name, module in linears.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    return activations, hooks

def compute_per_channel_stats_tensor(tensor_list, device="cpu"):
    """
    tensor_list: list of tensors with shape (batch, in_features)
    Returns dict with statistics along channel (feature) dimension
    """
    all_data = torch.cat(tensor_list, dim=0).to(
        dtype=torch.float32, device=device
    )  # concat batch dimension
    stats = {}
    stats["min"] = torch.min(all_data, dim=0)[0].to(device="cpu")
    stats["max"] = torch.max(all_data, dim=0)[0].to(device="cpu")
    stats["mean"] = torch.mean(all_data, dim=0).to(device="cpu")
    stats["std"] = torch.std(all_data, dim=0).to(device="cpu")
    stats["1%"] = torch.quantile(all_data, 0.01, dim=0).to(device="cpu")
    stats["99%"] = torch.quantile(all_data, 0.99, dim=0).to(device="cpu")
    stats["25%"] = torch.quantile(all_data, 0.25, dim=0).to(device="cpu")
    stats["75%"] = torch.quantile(all_data, 0.75, dim=0).to(device="cpu")
    return stats


def collect_linear_input_stats(model, dataloader, givens_dict, device="cuda"):
    linears = get_all_linears(model)
    activations = {}

    hooks = []
    for name, module in model.named_modules():
        if name in linears:

            def get_hook(n):
                def hook_fn(_mod, inp, _out, n=n):
                    x = inp[0].detach().cpu()
                    # smooth
                    x = x * givens_dict[n]["smooth"].to(device=x.device)

                    if x.dim() > 2:
                        x = x.view(-1, x.shape[-1])
                    elif x.dim() == 1:
                        x = x.unsqueeze(0)

                    if n not in activations:
                        activations[n] = []
                    activations[n].append(x)
                return hook_fn

            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="[Stat Layers] Running model on calibration data"
        ):
            this_input_tuple = _move_to_device(batch[0], device=device)
            _ = model(*this_input_tuple) if isinstance(this_input_tuple, tuple) else model(
                **this_input_tuple
            )
    for h in hooks:
        h.remove()

    linear_stats = {}
    for name, inputs in tqdm(activations.items(), desc="[Stat Layers] Computing statistics"):
        linear_stats[name] = compute_per_channel_stats_tensor(inputs, device="cpu")

    ks = [inp[0].shape[-1] for inp in activations.values()]

    rotation_stats = {}
    for i, (name, inputs) in tqdm(enumerate(activations.items()), desc="[Stat Layers] Rotation", total=len(activations)):
        if name not in givens_dict:
            print(f"[Skip] Missing {name}")
            continue

        # build Q
        vecs = givens_dict[name]["vecs"]
        pairs = givens_dict[name]["selected_pairs"]
        k = ks[i]
        Q = buildQ(pairs, vecs, k, dtype=torch.float32, device="cpu")

        for j in range(len(inputs)):
            inp = inputs[j]
            inputs[j] = (inp.to("cuda") @ Q.to(dtype=inp.dtype, device='cuda')).to("cpu")

        rotation_stats[name] = compute_per_channel_stats_tensor(
            inputs, device="cpu"
        )

    for h in hooks:
        h.remove()

    return linear_stats, rotation_stats


def run_percentile(
    dataset_type,
    model_type,
    layer_type,
    calib_config,
    calib_max_cache_size=1,
    calib_max_len=None,
    max_num=None,
    root_dir="benchmark_results",
    dataset_root_dir="../dataset/controlnet_datasets",
    calibrate_root_dir="calibset_record",
):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print(f"Layer Type: {layer_type}")
    print("--------------------------")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print(f"Calibrate Root Directory: {calibrate_root_dir}")
    print("--------------------------")

    ### exp root and latents
    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        raise ValueError(f"Latent root dir not found: {latent_root_dir}")

    latents_path = os.path.join(latent_root_dir, "latents.pt")
    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"Loaded latents with shape: {latents.shape}, path: {latents_path}")
    else:
        raise ValueError(f"Latents file not found at {latents_path}. Please generate latents first.")

    ### find calibrate data
    calibset = get_calibrate_data(
        dataset_type,
        model_type,
        layer_type,
        dataset_root_dir,
        calibrate_root_dir,
        calib_config,
        max_cache_size=calib_max_cache_size,
        max_len=calib_max_len,
    )
    part_model = get_part_model(model_type, layer_type, device="cuda:0")
    print(f"Calibrate set loaded with {len(calibset)} samples.")
    calib_loader = calibset.get_dataloader(batch_size=1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # create givens dict
    linear_name_lst = torch.load("anomaly/linear_names.pt")
    givens_dict = {}
    for i, name in enumerate(linear_name_lst):
        if i == 27:
            print(f"[{i}] {name}")
            givens = torch.load(f"anomaly/givens_{i}_chunk0.pt")
            pairs = torch.load(f"anomaly/pairs_{i}_chunk_0.pt")
            givens_dict[name] = {
                "vecs": givens["vecs"],
                "selected_pairs": pairs["selected_pairs"],
                "smooth": givens["smooth"],
            }
    torch.save(givens_dict, "anomaly/givens_dict.pt")
    stats = collect_linear_input_stats(
        part_model, calib_loader, givens_dict, device="cuda:0"
    )
    torch.save(stats, "anomaly/linear_input_stats.pt")


if __name__ == "__main__":
    print("Environment Variables:")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    parser = argparse.ArgumentParser(
        description="Unified Benchmark for Diffusion Models"
    )
    parser.add_argument(
        "-d",
        "--dataset-type",
        type=str,
        default="COCO",
        choices=["COCO", "MJHQ", "DCI"],
        help="Type of the dataset to use",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="sd3",
        choices=["flux", "sd3", "sdxl"],
        help="Type of the model to benchmark",
    )
    parser.add_argument(
        "-l",
        "--layer-type",
        type=str,
        default="dit",
        choices=["dit", "controlnet", "unet"],
        help="Type of the layer to benchmark",
    )
    parser.add_argument(
        "-C",
        "--calibrate-config",
        type=str,
        default="config/calibrate_config.json",
        help="Path to the calibration configuration file",
    )
    parser.add_argument(
        "-Cc",
        "--calibrate-cache-size",
        type=int,
        default=1,
        help="Maximum cache size for calibration data",
    )
    parser.add_argument(
        "-Cl",
        "--calibrate-max-len",
        type=int,
        default=None,
        help="Maximum length of calibration data",
    )
    parser.add_argument(
        "-n",
        "--max-num",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        type=str,
        default="../benchmark_results",
        help="Root directory for benchmark results",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../dataset/controlnet_datasets",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--calibrate-root",
        type=str,
        default="../calibset_record",
        help="Root directory for calibration sets",
    )
    args = parser.parse_args()

    calib_config_path = args.calibrate_config
    if not os.path.exists(calib_config_path):
        raise FileNotFoundError(
            f"Calibration configuration file {calib_config_path} does not exist."
        )
    with open(calib_config_path, "r") as f:
        calib_config = json.load(f)

    run_percentile(
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        layer_type=args.layer_type,
        calib_config=calib_config,
        calib_max_cache_size=args.calibrate_cache_size,
        calib_max_len=args.calibrate_max_len,
        max_num=args.max_num,
        root_dir=args.root_dir,
        dataset_root_dir=args.dataset_root,
        calibrate_root_dir=args.calibrate_root,
    )
