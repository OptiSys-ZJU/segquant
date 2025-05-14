import os
import numpy as np
import torch
import torch.nn as nn
from benchmark import trace_pic
from dataset.affine.noise_dataset import NoiseDataset
from sample.noise_sampler import NoiseSampler
from segquant.config import default_affine_config
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

class BlockwiseAffiner:
    def __init__(self, blocksize=128, alpha=0.5, lambda1=0.1, lambda2=0.1, max_timestep=30):
        self.solutions = {}  # timestep -> (mean_K, mean_b)
        self.cumulative = {}  # timestep -> {'sum_K': ..., 'sum_b': ..., 'count': ...}
        self.blocksize = blocksize
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_timestep = max_timestep
        self.learning = True

    def loss(self, K, b, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)
        epsilon_tilde = K * quantized + b
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - real) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        rel_loss = numerator / denominator
        k_penalty = torch.mean((K - 1.0) ** 2)
        b_penalty = torch.mean(b ** 2)
        loss = self.alpha * mse_loss + (1 - self.alpha) * rel_loss + self.lambda1 * k_penalty + self.lambda2 * b_penalty
        return loss
    
    def error(self, K, b, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        epsilon_tilde = K * quantized + b
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - real) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        rel_loss = numerator / denominator
        loss = self.alpha * mse_loss + (1 - self.alpha) * rel_loss
        return loss

    def get_solution(self, timestep):
        return self.solutions[timestep]
    
    def finish_learning(self):
        self.learning = False

    def step_learning(self, timestep, quantized, real):
        B, C, H, W = quantized.shape
        assert H % self.blocksize == 0 and W % self.blocksize == 0, "H and W must be divisible by block size"

        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        h_blocks = H // self.blocksize
        w_blocks = W // self.blocksize
        K_out = torch.zeros((B, C, H, W), device=quantized.device, dtype=torch.float32)
        b_out = torch.zeros((B, C, H, W), device=quantized.device, dtype=torch.float32)

        for i in range(h_blocks):
            for j in range(w_blocks):
                h_start = i * self.blocksize
                h_end = (i + 1) * self.blocksize
                w_start = j * self.blocksize
                w_end = (j + 1) * self.blocksize

                e_hat_block = quantized[:, :, h_start:h_end, w_start:w_end]  # shape: [B, C, Hb, Wb]
                e_block = real[:, :, h_start:h_end, w_start:w_end]

                if self.blocksize == 1:
                    e_hat_mean = e_hat_block.mean(dim=(0, 2, 3))  # shape: [C]
                    e_mean = e_block.mean(dim=(0, 2, 3))           # shape: [C]
                    A = self.alpha + (1 - self.alpha) / (e_mean ** 2 + 1e-8)  # shape: [C]
                    delta = e_mean - e_hat_mean                               # shape: [C]
                    denominator = 1 + (self.lambda2 * e_hat_mean ** 2) / self.lambda1 + self.lambda2 / A
                    b_block = delta / denominator                             # shape: [C]
                    K_block = 1 + (self.lambda2 * e_hat_mean / self.lambda1) * b_block  # shape: [C]
                    b_block_full = b_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)
                    K_block_full = K_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)
                    K_out[:, :, h_start:h_end, w_start:w_end] = K_block_full
                    b_out[:, :, h_start:h_end, w_start:w_end] = b_block_full
                else:
                    sum_e_hat_block = e_hat_block.sum(dim=(2, 3))
                    sum_sq_e_hat_block = (e_hat_block ** 2).sum(dim=(2, 3))

                    sum_e_block = e_block.sum(dim=(2, 3))
                    sum_sq_e_block = (e_block ** 2).sum(dim=(2, 3))

                    sum_two_block = (e_hat_block * e_block).sum(dim=(2, 3))

                    Hb = h_end - h_start
                    Wb = w_end - w_start
                    N = Hb * Wb
                    eps = 1e-8  # Prevent division by zero
                    A = self.alpha / N + (1 - self.alpha) / (sum_sq_e_block + eps)
                    lambda_1 = self.lambda1
                    lambda_2 = self.lambda2

                    # Ensure that denominator_b does not get too close to zero
                    denominator_b = (A * sum_e_hat_block) ** 2 - (A * sum_sq_e_hat_block + 2 * lambda_1) * (A * N + 2 * lambda_2)
                    denominator_b = torch.where(denominator_b.abs() < eps, torch.tensor(eps, dtype=torch.float32, device=quantized.device), denominator_b)

                    # Calculate b_block for this block
                    numerator_b = (A * sum_e_hat_block) * (A * sum_two_block + 2 * lambda_1) - (A * sum_sq_e_hat_block + 2 * lambda_1) * (A * sum_e_block)
                    b_block = numerator_b / denominator_b  # Shape: [B, C]

                    # Ensure that denominator_s does not get too close to zero
                    denominator_s = A * sum_sq_e_hat_block + 2 * lambda_1
                    denominator_s = torch.where(denominator_s.abs() < eps, torch.tensor(eps, dtype=torch.float32, device=quantized.device), denominator_s)

                    # Calculate s_block for this block
                    numerator_s = (A * sum_two_block + 2 * lambda_1) - (A * sum_e_hat_block) * b_block
                    s_block = numerator_s / denominator_s  # Shape: [B, C]

                    # Expand to match the block size for s_block and b_block
                    b_block_full = b_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)
                    s_block_full = s_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)

                    # Assign the computed s_block_full and b_block_full back to the output tensors
                    K_out[:, :, h_start:h_end, w_start:w_end] = s_block_full  # Assuming s_block is stored in K_out
                    b_out[:, :, h_start:h_end, w_start:w_end] = b_block_full

        K_mean = K_out.mean(dim=0, keepdim=True)
        b_mean = b_out.mean(dim=0, keepdim=True)

        if self.learning:
            if timestep not in self.cumulative:
                self.cumulative[timestep] = {
                    'sum_K': K_mean.clone(),
                    'sum_b': b_mean.clone(),
                    'count': 1
                }
            else:
                self.cumulative[timestep]['sum_K'] += K_mean
                self.cumulative[timestep]['sum_b'] += b_mean
                self.cumulative[timestep]['count'] += 1

            count = self.cumulative[timestep]['count']
            mean_K = self.cumulative[timestep]['sum_K'] / count
            mean_b = self.cumulative[timestep]['sum_b'] / count

            self.solutions[timestep] = (mean_K, mean_b)

        return (K_mean, b_mean)

def blockwise_affine(
    model_real: nn.Module, 
    model_quant: nn.Module, 
    target_device='cuda:0', 
    controlnet_scale=0.7, 
    guidance_scale=7, 
    shuffle=True, 
    config=None, 
    verbose=False,
    pack_per_file=2,
):
    if config is None:
        config = default_affine_config
    
    affiner = BlockwiseAffiner(max_timestep=config['max_timestep'], blocksize=config['blockwise'])
    learning_samples = config['learning_samples']

    if verbose:
        print(f"[BlockwiseAffiner] Init max_timestep[{config['max_timestep']}], blocksize[{config['blockwise']}], learning_samples[{config['learning_samples']}]")
    
    assert next(model_real.parameters()).device == torch.device("cpu")
    assert next(model_quant.parameters()).device == torch.device("cpu")

    noise_sampler = NoiseSampler(controlnet_scale=controlnet_scale, guidance_scale=guidance_scale)

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    
    save_dir = '.'

    def sample_noise(model: nn.Module, t: str):
        file_idx = 0
        outputs = []
        model.to(torch.device(target_device))
        data_loader = dataset.get_dataloader(sampler=SubsetRandomSampler(indices.copy()))
        print(f"[INFO] Sampling noise for {t} model...")  # Log start of sampling
        for sample_data in noise_sampler.sample(
            model,
            data_loader,
            max_timestep=config['max_timestep'],
            sample_size=learning_samples,
            timestep_per_sample=config['max_timestep']
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

            for ts, tv, real, quant in zip(timesteps, timestep_values, reals, quants):
                K = torch.ones_like(real)
                b = torch.zeros_like(real)
                init = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                K, b = affiner.step_learning(ts, quant, real)
                affine = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                if verbose:
                    print(f'BlockwiseAffiner [{ts}] block[{affiner.blocksize}] Learning init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}]')

        # Delete temporary files after processing
        os.remove(os.path.join(save_dir, real_file))
        os.remove(os.path.join(save_dir, quant_file))
        print(f"[INFO] Deleted temporary files: {real_file}, {quant_file}")

    affiner.finish_learning()
    return affiner



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
            "alpha": 0.5,
        },
    }
    calib_args = {
        "max_timestep": 60,
        "sample_size": 16,
        "timestep_per_sample": 30,
        "controlnet_conditioning_scale": 0,
        "guidance_scale": 7,
        "shuffle": True,
    }

    this_affine_config = {
        "blockwise": 128,
        "learning_samples": 8,
        "max_timestep": 30
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

    affiner = blockwise_affine(model_real, model_quant, config=this_affine_config, verbose=True)

    ## perform
    max_num = 1
    model_quant = model_quant.to('cuda')
    latents = torch.load('../latents.pt')
    trace_pic(model_quant, 'affine_pics/blockaffine', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=0, guidance_scale=7, num_inference_steps=30, affiner=affiner)

    trace_pic(model_quant, 'affine_pics/quant', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=0, guidance_scale=7, num_inference_steps=30)
    
    del model_quant
    model_real = model_real.to('cuda')
    trace_pic(model_real, 'affine_pics/real', dataset.get_dataloader(), latents, max_num=max_num, 
              controlnet_conditioning_scale=0, guidance_scale=7, num_inference_steps=30)
