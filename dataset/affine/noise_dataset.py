from enum import Enum
import os
import numpy as np
import torch
from sample.noise_sampler import NoiseSampler
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch.nn as nn
import json

class NoiseDataset(Dataset):
    def __init__(self, data_dir, max_timestep=None):
        self.data_dir = data_dir

        if max_timestep is None:
            metadata_path = os.path.join(data_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                max_timestep = metadata.get("sampling", {}).get("max_timestep", 30)
            else:
                max_timestep = 30
        self.max_timestep = max_timestep

        self.data_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.pt')
        ])
        self.data = self.load_data()

    def load_data(self):
        return [torch.load(os.path.join(self.data_dir, f)) for f in self.data_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_batch = self.data[idx]
        timesteps = []
        timestep_values = []
        real_noises = []
        quant_noises = []

        for sample in data_batch:
            ts_list = sample["timesteps"]
            tv_list = sample["timestep_values"]
            rs_list = sample["real_noises"]
            qs_list = sample["quant_noises"]

            for ts, tv, real, quant in zip(ts_list, tv_list, rs_list, qs_list):
                if isinstance(ts, torch.Tensor) and ts.numel() > 1:
                    ts = ts.view(-1)[0]
                if isinstance(tv, torch.Tensor) and tv.numel() > 1:
                    tv = tv.view(-1)[0]

                timesteps.append(ts)
                timestep_values.append(tv)
                real_noises.append(real)
                quant_noises.append(quant)

        return {
            "timesteps": torch.tensor(timesteps),              # [T]
            "timestep_values": torch.tensor(timestep_values),  # [T]
            "real_noises": torch.stack(real_noises),          # [T, ...]
            "quant_noises": torch.stack(quant_noises),        # [T, ...]
        }

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        def collate_timesteps_fn(batch):
            T = self.max_timestep

            real_stack = [[] for _ in range(T)]
            quant_stack = [[] for _ in range(T)]
            timestep_stack = [[] for _ in range(T)]
            timestep_value_stack = [[] for _ in range(T)]

            for sample in batch:
                t_len = sample['timesteps'].shape[0]
                pad_len = T - t_len

                if pad_len > 0:
                    raise ValueError('timesteps invalid')

                for t in range(T):
                    real_stack[t].append(sample['real_noises'][t])
                    quant_stack[t].append(sample['quant_noises'][t])
                    timestep_stack[t].append(sample['timesteps'][t])
                    timestep_value_stack[t].append(sample['timestep_values'][t])

            batch_list = []
            for t in range(T):
                batch_list.append({
                    "real": torch.cat(real_stack[t], dim=0),
                    "quant": torch.cat(quant_stack[t], dim=0),
                    "timestep": torch.stack(timestep_stack[t], dim=0),
                    "timestep_value": torch.stack(timestep_value_stack[t], dim=0)
                })

            return batch_list

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_timesteps_fn,
            **kwargs
        )



def generate_noise(
    model_real: nn.Module,
    model_quant: nn.Module,
    dataset: Dataset,
    save_dir: str,
    target_device='cuda:0',
    max_timestep=30,
    sample_size=32,
    timestep_per_sample=30,
    controlnet_scale=0.7,
    guidance_scale=7,
    shuffle=True,
    pack_per_file=2,
):
    assert next(model_real.parameters()).device == torch.device("cpu")
    assert next(model_quant.parameters()).device == torch.device("cpu")
    os.makedirs(save_dir, exist_ok=True)

    noise_sampler = NoiseSampler(controlnet_scale=controlnet_scale, guidance_scale=guidance_scale)

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    def sample_noise(model: nn.Module, t: str):
        file_idx = 0
        outputs = []
        model.to(torch.device(target_device))
        data_loader = dataset.get_dataloader(sampler=SubsetRandomSampler(indices.copy()))
        print(f"[INFO] Sampling noise for {t} model...")  # Log start of sampling
        for sample_data in noise_sampler.sample(
            model,
            data_loader,
            max_timestep=max_timestep,
            sample_size=sample_size,
            timestep_per_sample=timestep_per_sample
        ):
            outputs.append(sample_data)
            file_idx += 1
            if file_idx % pack_per_file == 0:
                save_path = os.path.join(save_dir, f'tmp_{t}{file_idx}.pt')
                torch.save(outputs, save_path)
                print(f"[INFO] Saved {file_idx} samples for {t} model to {save_path}")  # Log every save
                outputs = []  # Clear the outputs after saving

        # Ensure any remaining samples are saved
        if len(outputs) > 0:
            save_path = os.path.join(save_dir, f'tmp_{t}{file_idx}.pt')
            torch.save(outputs, save_path)
            print(f"[INFO] Saved remaining {len(outputs)} samples for {t} model to {save_path}")  # Log final save

        print(f"[INFO] Finished sampling noise for {t} model.")  # Log end of sampling
        model.to(torch.device('cpu'))
    
    def process_noise_pred(noise_pred: torch.Tensor, mix=False, guidance=None):
        if noise_pred.dim() == 0:
            raise ValueError("Unexpected scalar tensor")
        if noise_pred.size(0) == 2:
            if mix:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                return noise_pred
            else:
                return (noise_pred[0], noise_pred[1])
        elif noise_pred.size(0) == 1:
            return noise_pred.squeeze(0)
        else:
            raise ValueError(f"Unsupported first dimension size: {noise_pred.size(0)}")

    # Sample noise for both models
    print("[INFO] Starting noise sampling for real model.")
    sample_noise(model_real, 'real')
    print("[INFO] Starting noise sampling for quant model.")
    sample_noise(model_quant, 'quant')

    # Now we read the temporary files, process them, and write new files
    buffer = []
    file_idx = 0
    
    # List temporary files created during sampling
    real_files = sorted([f for f in os.listdir(save_dir) if f.startswith('tmp_real')])
    quant_files = sorted([f for f in os.listdir(save_dir) if f.startswith('tmp_quant')])

    print(f"[INFO] Found {len(real_files)} temporary files for real model.")
    print(f"[INFO] Found {len(quant_files)} temporary files for quant model.")

    for real_file, quant_file in zip(real_files, quant_files):
        print(f"[INFO] Processing files: {real_file} and {quant_file}")
        
        # Load the temp files
        real_datas = torch.load(os.path.join(save_dir, real_file))
        quant_datas = torch.load(os.path.join(save_dir, quant_file))

        # Process noise predictions
        for real_data, quant_data in zip(real_datas, quant_datas):
            timesteps = [r['timestep'] for r in real_data]
            timestep_values = [r['extra_features']['timestep'] for r in real_data]
            reals = [process_noise_pred(r['noise_pred'], mix=True, guidance=r['extra_features']['guidance_scale']) for r in real_data]
            quants = [process_noise_pred(q['noise_pred'], mix=True, guidance=q['extra_features']['guidance_scale']) for q in quant_data]

            # Prepare data for the final file
            this_sample = {
                "timesteps": timesteps,
                "timestep_values": timestep_values,
                "real_noises": reals,
                "quant_noises": quants,
            }

            buffer.append(this_sample)
        
        # Save every pack_per_file number of samples
        if len(buffer) >= pack_per_file:
            save_path = os.path.join(save_dir, f"sample_{file_idx:05d}.pt")
            torch.save(buffer, save_path)
            print(f"[INFO] Saved {len(buffer)} batches to {save_path}")  # Log each save
            buffer.clear()
            file_idx += 1

        # Delete temporary files after processing
        os.remove(os.path.join(save_dir, real_file))
        os.remove(os.path.join(save_dir, quant_file))
        print(f"[INFO] Deleted temporary files: {real_file}, {quant_file}")

    # Save any remaining data in buffer
    if buffer:
        save_path = os.path.join(save_dir, f"sample_{file_idx:05d}.pt")
        torch.save(buffer, save_path)
        print(f"[INFO] Saved remaining {len(buffer)} batches to {save_path}")

if __name__ == '__main__':
    from dataset.coco.coco_dataset import COCODataset
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)

    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cuda:0')

    from segquant.config import DType, SegPattern
    quant_config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 1.0,
        },
    }
    calib_args = {
        "max_timestep": 60,
        "sample_size": 16,
        "timestep_per_sample": 30,
        "controlnet_conditioning_scale": 0.7,
        "guidance_scale": 7,
        "shuffle": True,
    }
    sample_args = {
        "max_timestep": 60,
        "sample_size": 32,
        "timestep_per_sample": 60,
        "controlnet_scale": 0.7,
        "guidance_scale": 7,
        "shuffle": True,
        "pack_per_file": 2
    }

    def quant_model(model_real: nn.Module, quant_layer: str, config, dataset, calib_args: dict) -> nn.Module:
        from sample.sampler import Q_DiffusionSampler, model_map
        from segquant.torch.calibrate_set import generate_calibrate_set
        from segquant.torch.quantization import quantize

        sampler = Q_DiffusionSampler()
        sample_dataloader = dataset.get_dataloader(batch_size=1, shuffle=calib_args["shuffle"])
        calibset = generate_calibrate_set(model_real, sampler, sample_dataloader, quant_layer, 
                                        max_timestep=calib_args["max_timestep"],
                                        sample_size=calib_args["sample_size"],
                                        timestep_per_sample=calib_args["timestep_per_sample"],
                                        controlnet_conditioning_scale=calib_args["controlnet_conditioning_scale"],
                                        guidance_scale=calib_args["guidance_scale"])

        calib_loader = calibset.get_dataloader(batch_size=1)
        model_real.transformer = quantize(model_map[quant_layer](model_real), calib_loader, config, True)
        return model_real
    model_quant = quant_model(model, 'dit', quant_config, dataset, calib_args).to('cpu')
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')


    save_dir = '../dataset/affine_noise'
    generate_noise(model_real, model_quant, dataset, save_dir, 'cuda:0', **sample_args)

    import json, os
    metadata = {
        "calibration": calib_args,
        "quant_config": quant_config,
        "sampling": sample_args,
    }
    os.makedirs(save_dir, exist_ok=True)

    def custom_serializer(obj):
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializable")
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=custom_serializer)
    print(f"[INFO] Metadata saved to {os.path.join(save_dir, 'metadata.json')}")

