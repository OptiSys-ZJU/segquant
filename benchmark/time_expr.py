import argparse
import os
import time

import torch

from benchmark.utils import get_full_model_with_quantized_part, get_latents, get_dataset, get_part_model
from segquant.config import Calibrate, DType, Optimum
from segquant.torch.quantization import quantize
from segquant.torch.calibrate_set import BaseCalibSet

def run_real_baseline(dataset_type, model_type, max_num=None, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print('--------------------------')
    print(f"Root Directory: {root_dir}")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print('--------------------------')
    
    print(f"Generating new latents for speeding.")
    latents = get_latents(model_type, device="cuda:0")

    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, 'real')
    if not os.path.exists(exp_root_dir):
        raise FileNotFoundError(f"Experiment root directory {exp_root_dir} does not exist. Please create it first.")
    
    part_model = get_part_model(model_type, layer_type, device="cuda:0")
    config = {
        "default": {
            "enable": True,
            "seglinear": False,
            "search_patterns": [],
            "real_quant": True,
            "opt": {
                "type": Optimum.DEFAULT,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.FP16,
            },
            "weight_quant": {
                "type": DType.FP16,
            },
        }
    }

    calibset = BaseCalibSet(data=[])
    calib_loader = calibset.get_dataloader(batch_size=1)
    quantized_model = quantize(
        part_model, calib_loader, config, verbose=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_full_model_with_quantized_part(model_type, layer_type, quantized_model, device="cuda:0")

    ##### run inference
    print("Running inference...")
    dataset = get_dataset(dataset_type, dataset_root_dir)

    # Get model device to ensure inputs are on the same device
    model_device = next(model.parameters()).device

    inference_times = []
    torch.cuda.reset_peak_memory_stats(device=model_device)
    count = 0

    for batch in dataset.get_dataloader():
        for b in batch:
            if max_num is not None and count >= max_num:
                print("Max_num reached, exiting...")
                avg_time = sum(inference_times) / len(inference_times)
                peak_mem = torch.cuda.max_memory_allocated(device=model_device) / 1024 / 1024
                print(f"\nAverage inference time: {avg_time:.4f} s")
                print(f"Peak memory usage: {peak_mem:.2f} MB")
                return

            prompt, _, control = b

            if hasattr(control, 'to'):
                control = control.to(model_device)
            elif hasattr(control, 'device'):
                if control.device != model_device:
                    control = control.to(model_device)

            with torch.cuda.device(model_device):
                torch.cuda.synchronize()
                start_time = time.time()

                _ = model.forward(
                    prompt=prompt, control_image=control, latents=latents,
                    controlnet_conditioning_scale=0,
                    guidance_scale=7,
                    num_inference_steps=50,
                )[0]

                torch.cuda.synchronize()
                end_time = time.time()

            elapsed = end_time - start_time
            inference_times.append(elapsed)
            print(f"[{count}] Inference time: {elapsed:.4f} s")

            count += 1

    # Final report
    avg_time = sum(inference_times) / len(inference_times)
    peak_mem = torch.cuda.max_memory_allocated(device=model_device) / 1024 / 1024
    print(f"\nAverage inference time: {avg_time:.4f} s")
    print(f"Peak memory usage: {peak_mem:.2f} MB")

    del model
    print("Benchmark completed")

def run_experiment(dataset_type, model_type, layer_type, exp_all_name, max_num=None, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print(f"Layer Type: {layer_type}")
    print(f"Running experiment: {exp_all_name}")
    print('--------------------------')
    print(f"Root Directory: {root_dir}")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print('--------------------------')
    
    print(f"Generating new latents for speeding.")
    latents = get_latents(model_type, device="cuda:0")

    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)
    if not os.path.exists(exp_root_dir):
        raise FileNotFoundError(f"Experiment root directory {exp_root_dir} does not exist. Please create it first.")

    ##### find partial model
    print("Checking for partial model...")
    part_model_path = os.path.join(exp_root_dir, 'part_model.pt')
    if os.path.exists(part_model_path):
        print(f"Partial model file {part_model_path} already exists.")
        quantized_model = torch.load(part_model_path, weights_only=False)
    else:
        raise FileNotFoundError(f"Partial model file {part_model_path} does not exist. Please create it first.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_full_model_with_quantized_part(model_type, layer_type, quantized_model, device="cuda:0")

    ##### run inference
    print("Running inference...")
    dataset = get_dataset(dataset_type, dataset_root_dir)

    # Get model device to ensure inputs are on the same device
    model_device = next(model.parameters()).device

    inference_times = []
    torch.cuda.reset_peak_memory_stats(device=model_device)
    count = 0

    for batch in dataset.get_dataloader():
        for b in batch:
            if max_num is not None and count >= max_num:
                print("Max_num reached, exiting...")
                avg_time = sum(inference_times) / len(inference_times)
                peak_mem = torch.cuda.max_memory_allocated(device=model_device) / 1024 / 1024
                print(f"\nAverage inference time: {avg_time:.4f} s")
                print(f"Peak memory usage: {peak_mem:.2f} MB")
                return

            prompt, _, control = b

            if hasattr(control, 'to'):
                control = control.to(model_device)
            elif hasattr(control, 'device'):
                if control.device != model_device:
                    control = control.to(model_device)

            with torch.cuda.device(model_device):
                torch.cuda.synchronize()
                start_time = time.time()

                _ = model.forward(
                    prompt=prompt, control_image=control, latents=latents,
                    controlnet_conditioning_scale=0,
                    guidance_scale=7,
                    num_inference_steps=50,
                )[0]

                torch.cuda.synchronize()
                end_time = time.time()

            elapsed = end_time - start_time
            inference_times.append(elapsed)
            print(f"[{count}] Inference time: {elapsed:.4f} s")

            count += 1

    # Final report
    avg_time = sum(inference_times) / len(inference_times)
    peak_mem = torch.cuda.max_memory_allocated(device=model_device) / 1024 / 1024
    print(f"\nAverage inference time: {avg_time:.4f} s")
    print(f"Peak memory usage: {peak_mem:.2f} MB")

    del model
    print("Benchmark completed")

if __name__ == "__main__":
    print("Environment Variables:")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    parser = argparse.ArgumentParser(description='Time Benchmark for Diffusion Models')
    parser.add_argument('-d', '--dataset-type', type=str, default='COCO', choices=['COCO', 'MJHQ', 'DCI'], help='Type of the dataset to use')
    parser.add_argument('-m', '--model-type', type=str, default='sd3', choices=['flux', 'sd3', 'sdxl'], help='Type of the model to benchmark')
    parser.add_argument('-l', '--layer-type', type=str, default='dit', choices=['dit', 'controlnet', 'unet'], help='Type of the layer to benchmark')
    parser.add_argument('-q', '--quant-type', type=str, default='int8w8a8', help='Type of the quant to benchmark')
    parser.add_argument('-e', '--exp-name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('-n', '--max-num', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-r', '--root-dir', type=str, default='../benchmark_results', help='Root directory for benchmark results')
    parser.add_argument('--dataset-root', type=str, default='../dataset/controlnet_datasets', help='Root directory for datasets')
    parser.add_argument('-R', '--run-real-baseline', action='store_true', help='Run the real baseline experiment without quantization')

    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_type = args.model_type
    layer_type = args.layer_type
    quant_type = args.quant_type
    exp_name = args.exp_name
    max_num = args.max_num
    root_dir = args.root_dir
    dataset_root_dir = args.dataset_root

    if args.run_real_baseline:
        run_real_baseline(dataset_type, model_type, max_num, root_dir, dataset_root_dir)
    else:
        exp_all_name = f'{model_type}-{layer_type}-{quant_type}-{exp_name}'
        run_experiment(dataset_type, model_type, layer_type, exp_all_name, max_num, root_dir, dataset_root_dir)



