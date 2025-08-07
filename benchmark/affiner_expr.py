import argparse
import json
import torch
from segquant.torch.affiner import load_affiner, create_affiner
from benchmark import trace_pic
from benchmark.utils import get_dataset, get_full_model, get_full_model_with_quantized_part
import os

def run_experiment(dataset_type, model_type, layer_type, exp_all_name, affiner_type, affiner_config, affiner_name='', max_num=None, shuffle=True, force_process_pics=False, root_dir='benchmark_results', dataset_root_dir='../dataset/controlnet_datasets'):
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {model_type}")
    print(f"Layer Type: {layer_type}")
    print(f"Running base experiment: {exp_all_name}")
    print(f"Affiner Type: {affiner_type}")
    print(f"Affiner Config: {affiner_config}")
    print("shuffle: ", shuffle)
    print('--------------------------')
    print(f"Root Directory: {root_dir}")
    print(f"Dataset Root Directory: {dataset_root_dir}")
    print('--------------------------')

    ### exp root and latents
    latent_root_dir = os.path.join(root_dir, dataset_type, model_type)
    if not os.path.exists(latent_root_dir):
        raise FileNotFoundError(f"Latent root directory {latent_root_dir} does not exist.")
    
    latents_path = os.path.join(latent_root_dir, 'latents.pt')
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"Latent file {latents_path} does not exist. Please generate latents first.")
    
    latents = torch.load(latents_path)
    print(f"Loaded latents with shape: {latents.shape}")

    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)
    if not os.path.exists(exp_root_dir):
        raise FileNotFoundError(f"Experiment root directory {exp_root_dir} does not exist. Please create it first.")
    print(f"Find Experiment directory {exp_root_dir}")

    ##### find partial model
    print("Checking for partial model...")
    part_model_path = os.path.join(exp_root_dir, 'part_model.pt')
    if os.path.exists(part_model_path):
        print(f"Find Partial model file {part_model_path}.")
        quantized_model = torch.load(part_model_path, weights_only=False)
    else:
        raise FileNotFoundError(f"Partial model file {part_model_path} does not exist. Please create it first.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_quant = get_full_model_with_quantized_part(model_type, layer_type, quantized_model, device="cpu")

    ### learning
    affiner_path = os.path.join(exp_root_dir, affiner_type)
    if not os.path.exists(affiner_path):
        os.makedirs(affiner_path)
    affiner_dump_path = os.path.join(affiner_path, f'affiner{affiner_name}.pt')
    if os.path.exists(affiner_dump_path):
        print(f"Affiner found at {affiner_dump_path}, loading...")
        affiner = load_affiner(affiner_dump_path)
    else:
        print("Creating new affiner...")
        model_real = get_full_model(model_type, device="cpu")
        affiner = create_affiner(
            affiner_config, 
            dataset=get_dataset(dataset_type, dataset_root_dir),
            model_real=model_real,
            model_quant=model_quant,
            latents=latents,
            dump_path=affiner_dump_path,
            shuffle=shuffle,
        )
        del model_real

    ##### run inference
    print("Running inference...")
    model_quant = model_quant.to("cuda:0")
    pic_path = os.path.join(affiner_path, 'pic')
    dataset = get_dataset(dataset_type, dataset_root_dir)

    trace_pic(
        model_quant,
        pic_path,
        dataset.get_dataloader(),
        latents,
        max_num=max_num,
        force_process_pics=force_process_pics,
        steper=affiner,
        controlnet_conditioning_scale=affiner_config['extra_args']["controlnet_conditioning_scale"],
        guidance_scale=affiner_config['extra_args']["guidance_scale"],
        num_inference_steps=affiner_config['config']['max_timestep'],
    )
    del model_quant
    print("Benchmark completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Affine Benchmark for Diffusion Models')
    parser.add_argument('-d', '--dataset-type', type=str, default='COCO', choices=['COCO', 'MJHQ', 'DCI'], help='Type of the dataset to use')
    parser.add_argument('-m', '--model-type', type=str, default='sd3', choices=['flux', 'sd3', 'sdxl'], help='Type of the model to benchmark')
    parser.add_argument('-l', '--layer-type', type=str, default='dit', choices=['dit', 'controlnet', 'unet'], help='Type of the layer to benchmark')
    parser.add_argument('-q', '--quant-type', type=str, default='int8w8a8', help='Type of the quant to benchmark')
    parser.add_argument('-e', '--exp-name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('-a', '--affiner-type', type=str, default='blockwise', choices=['blockwise', 'ptqd', 'tac'], help='Type of the affiner to benchmark')
    parser.add_argument('-ac', '--affiner-config', type=str, default='config/affine/blockwise.json', help='Configuration file for the affiner')
    parser.add_argument('-an', '--affiner-name', type=str, default='', help='Name of the affiner, used for saving the affiner')
    parser.add_argument('-n', '--max-num', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-r', '--root-dir', type=str, default='../benchmark_results', help='Root directory for benchmark results')
    parser.add_argument('--dataset-root', type=str, default='../dataset/controlnet_datasets', help='Root directory for datasets')
    parser.add_argument('--dis-shuffle', action='store_false', dest='shuffle', help='Disable shuffling of the dataset')
    parser.add_argument('--force-process-pics',action='store_true', help='Force processing pictures even if they already exist')

    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_type = args.model_type
    layer_type = args.layer_type
    quant_type = args.quant_type
    exp_name = args.exp_name
    affiner_type = args.affiner_type
    affiner_config = args.affiner_config
    affiner_name = args.affiner_name
    
    root_dir = args.root_dir
    max_num = args.max_num
    force_process_pics = args.force_process_pics
    dataset_root_dir = args.dataset_root

    # load real and quant
    exp_all_name = f'{model_type}-{layer_type}-{quant_type}-{exp_name}'
    exp_root_dir = os.path.join(root_dir, dataset_type, model_type, layer_type, exp_all_name)

    # Load configuration
    with open(affiner_config, 'r') as f:
        affiner_config = json.load(f)
    
    run_experiment(
        dataset_type=dataset_type,
        model_type=model_type,
        layer_type=layer_type,
        exp_all_name=exp_all_name,
        affiner_type=affiner_type,
        affiner_config=affiner_config,
        affiner_name=affiner_name,
        max_num=max_num,
        force_process_pics=force_process_pics,
        root_dir=root_dir,
        dataset_root_dir=dataset_root_dir,
        shuffle=args.shuffle,
    )