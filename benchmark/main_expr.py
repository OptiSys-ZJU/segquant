import argparse
import json
import torch
import os
from benchmark import trace_pic
from benchmark.yaml_parser import parse_yaml
from benchmark.utils import get_dataset, get_latents, get_calibrate_data, get_full_model, get_part_model, get_full_model_with_quantized_part
from segquant.torch.quantization import quantize

def run_real_baseline(dataset_type, model_type, calib_config, max_num=None, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")

    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        os.makedirs(latent_root_dir, exist_ok=True)
        print(f"Created latent root directory: {latent_root_dir}")
    
    latents_path = os.path.join(latent_root_dir, 'latents.pt')
    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"Loaded latents with shape: {latents.shape}")
    else:
        print(f"Latents file {latents_path} does not exist. Generating new latents.")
        latents = get_latents(model_type, device="cuda:0")
        torch.save(latents, latents_path)
    
    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, 'real')
    if os.path.exists(exp_root_dir):
        print(f"Experiment directory {exp_root_dir} already exists. Skipping creation.")
    else:
        os.makedirs(exp_root_dir, exist_ok=True)
        print(f"Created experiment directory: {exp_root_dir}")

    model = get_full_model(model_type, device="cuda:0")

    ##### run inference
    print("Running inference...")
    pic_path = os.path.join(exp_root_dir, 'pic')
    dataset = get_dataset(dataset_type, dataset_root_dir)

    trace_pic(
        model,
        pic_path,
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_config["controlnet_conditioning_scale"],
        guidance_scale=calib_config["guidance_scale"],
        num_inference_steps=calib_config['max_timestep'],
    )
    del model
    print("Real completed")

def run_experiment(dataset_type, model_type, layer_type, exp_all_name, config, calib_config, max_num=None, per_layer_mode=False, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets', calibrate_root_dir='calibset_record'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print(f"Layer Type: {layer_type}")
    print(f"Running experiment: {exp_all_name}")
    print('--------------------------')
    print("Configuration:")
    print(config)
    print('--------------------------')
    print(f"Root Directory: {root_dir}")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print(f"Calibrate Root Directory: {calibrate_root_dir}")
    print('--------------------------')

    ### exp root and latents
    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        os.makedirs(latent_root_dir, exist_ok=True)
        print(f"Created latent root directory: {latent_root_dir}")
    
    latents_path = os.path.join(latent_root_dir, 'latents.pt')
    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"Loaded latents with shape: {latents.shape}")
    else:
        print(f"Latents file {latents_path} does not exist. Generating new latents.")
        latents = get_latents(model_type, device="cuda:0")
        torch.save(latents, latents_path)

    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)
    if os.path.exists(exp_root_dir):
        print(f"Experiment directory {exp_root_dir} already exists. Skipping creation.")
    else:
        os.makedirs(exp_root_dir, exist_ok=True)
        print(f"Created experiment directory: {exp_root_dir}")

    ##### find partial model
    print("Checking for partial model...")
    part_model_path = os.path.join(exp_root_dir, 'part_model.pt')
    if os.path.exists(part_model_path):
        print(f"Partial model file {part_model_path} already exists. Skipping creation.")
        quantized_model = torch.load(part_model_path, weights_only=False)
    else:
        ### find calibrate data
        calibset = get_calibrate_data(
            dataset_type, model_type, layer_type, dataset_root_dir, calibrate_root_dir, calib_config
        )
        print(f"Creating partial model file: {part_model_path}, need to be quantized.")
        part_model = get_part_model(model_type, layer_type, device="cuda:0")
        print(f"Calibrate set loaded with {len(calibset)} samples.")
        calib_loader = calibset.get_dataloader(batch_size=1)
        quantized_model = quantize(
            part_model, calib_loader, config, 
            per_layer_mode=per_layer_mode, verbose=True
        )
        torch.save(quantized_model, part_model_path)
        print(f"Partial model saved to {part_model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = get_full_model_with_quantized_part(model_type, layer_type, quantized_model, device="cuda:0")

    ##### run inference
    print("Running inference...")
    pic_path = os.path.join(exp_root_dir, 'pic')
    dataset = get_dataset(dataset_type, dataset_root_dir)

    trace_pic(
        model,
        pic_path,
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        controlnet_conditioning_scale=calib_config["controlnet_conditioning_scale"],
        guidance_scale=calib_config["guidance_scale"],
        num_inference_steps=calib_config['max_timestep'],
    )
    del model
    print("Benchmark completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Benchmark for Diffusion Models')
    parser.add_argument('-d', '--dataset-type', type=str, default='COCO', choices=['COCO', 'MJHQ', 'DCI'], help='Type of the dataset to use')
    parser.add_argument('-m', '--model-type', type=str, default='sd3', choices=['flux', 'sd3', 'sdxl'], help='Type of the model to benchmark')
    parser.add_argument('-l', '--layer-type', type=str, default='dit', choices=['dit', 'controlnet', 'unet'], help='Type of the layer to benchmark')
    parser.add_argument('-q', '--quant-type', type=str, default='int8w8a8', help='Type of the quant to benchmark')
    parser.add_argument('-e', '--exp-name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('-c', '--config-dir', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('-C', '--calibrate-config', type=str, default='config/calibrate_config.json', help='Path to the calibration configuration file')
    parser.add_argument('-n', '--max-num', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-p', '--per-layer-mode', action='store_true', help='Run in per-layer quantization mode')
    parser.add_argument('-r', '--root-dir', type=str, default='../benchmark_results', help='Root directory for benchmark results')
    parser.add_argument('--dataset-root', type=str, default='../dataset/controlnet_datasets', help='Root directory for datasets')
    parser.add_argument('--calibrate-root', type=str, default='../calibset_record', help='Root directory for calibration sets')
    parser.add_argument('-R', '--run-real-baseline', action='store_true', help='Run the real baseline experiment without quantization')
    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_type = args.model_type
    layer_type = args.layer_type
    quant_type = args.quant_type
    exp_name = args.exp_name
    root_dir = args.root_dir
    max_num = args.max_num
    per_layer_mode = args.per_layer_mode
    dataset_root_dir = args.dataset_root
    calibrate_root_dir = args.calibrate_root

    calib_config_path = args.calibrate_config
    if not os.path.exists(calib_config_path):
        raise FileNotFoundError(f"Calibration configuration file {calib_config_path} does not exist.")
    with open(calib_config_path, 'r') as f:
        calib_config = json.load(f)
    
    if args.run_real_baseline:
        # Run the real baseline experiment without quantization
        run_real_baseline(
            dataset_type, model_type, calib_config, 
            max_num=max_num, root_dir=root_dir, 
            dataset_root_dir=dataset_root_dir
        )
    else:
        # Load configuration
        config_dir = args.config_dir
        config_path = os.path.join(config_dir, model_type, layer_type, f'{quant_type}.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
        from benchmark.yaml_parser import parse_yaml
        configs = parse_yaml(model_type, layer_type, quant_type, config_path)
        exp_all_name = f'{model_type}-{layer_type}-{quant_type}-{exp_name}'
        if exp_all_name in configs:
            run_experiment(
                dataset_type, model_type, layer_type, exp_all_name, 
                configs[exp_all_name], calib_config, max_num=max_num, 
                per_layer_mode=per_layer_mode, root_dir=root_dir, 
                dataset_root_dir=dataset_root_dir, calibrate_root_dir=calibrate_root_dir
            )
        else:
            print(f"Experiment {exp_all_name} not found in the configuration file.")