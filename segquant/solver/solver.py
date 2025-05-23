import torch
import torch.nn.functional as F

def affine_with_percentile_scale(quant, K, b=None, percentile=100, greater=True, scale=1, verbose=False):
    with torch.no_grad():
        device = quant.device
        pre_type = quant.dtype

        noise_pred_float = quant.to(dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32)
        if b is None:
            b = torch.zeros_like(K)
        b = b.to(device=device, dtype=torch.float32)

        abs_noise = noise_pred_float.abs()
        
        if greater:
            percentile = 100 - percentile

        threshold = torch.quantile(abs_noise.flatten(), percentile / 100.0)

        if greater:
            mask = abs_noise > threshold
        else:
            mask = abs_noise < threshold
        affine_pred = K * noise_pred_float + b
        blended = noise_pred_float * (1 - scale) + affine_pred * scale
        out = torch.where(mask, blended, noise_pred_float)
        replaced_ratio = mask.sum().item() / mask.numel()

        if verbose:
            print(f"[Affine Replace] Threshold={threshold.item():.4f}, Replace Ratio={replaced_ratio*100:.2f}%")
        return out.to(dtype=pre_type)

def affine_with_threshold(quant, K, b=None, threshold=1, verbose=False):
    with torch.no_grad():
        device = quant.device
        pre_type = quant.dtype

        noise_pred_float = quant.to(dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32)
        if b is None:
            b = torch.zeros_like(K)
        b = b.to(device=device, dtype=torch.float32)
        mask = noise_pred_float > threshold * noise_pred_float.abs().mean()
        affine_pred = K * noise_pred_float + b
        out = torch.where(mask, affine_pred, noise_pred_float)
        replaced_ratio = mask.sum().item() / mask.numel()
        if verbose:
            print(f"[Affine Replace] Threshold={threshold.item():.4f}, Replace Ratio={replaced_ratio*100:.2f}%")
        return out.to(dtype=pre_type)

class BaseSolver:
    def __init__(self):
        pass

    def learn(self, real, quant):
        pass

    def solve(self, quant):
        pass

class MSERelSolver(BaseSolver):
    def __init__(self, config):
        super().__init__()

        if config is None:
            self.blocksize = 1
            self.alpha = 0.5
            self.lambda1 = 0.1
            self.lambda2 = 0.1
            self.sample_mode = 'block'

            self.percentile = 100
            self.greater = True
            self.scale = 1
            self.verbose = True
        else:
            self.blocksize = config['blocksize']
            self.alpha = config['alpha']
            self.lambda1 = config['lambda1']
            self.lambda2 = config['lambda2']
            self.sample_mode = config['sample_mode']

            self.percentile = config['percentile']
            self.greater = config['greater']
            self.scale = config['scale']
            self.verbose = config['verbose']

        self.record = None
        self.solution = None
    
    def _solve_single(self, e, e_hat):
        A = self.alpha + (1 - self.alpha) / (e ** 2 + 1e-8)
        delta = e - e_hat
        denominator = 1 + (self.lambda2 * e_hat ** 2) / self.lambda1 + self.lambda2 / A
        b = delta / denominator
        K = 1 + (self.lambda2 * e_hat / self.lambda1) * b
        return (K, b)
    
    def _sample_block(self, quantized, real):
        B, C, H, W = quantized.shape
        assert H % self.blocksize == 0 and W % self.blocksize == 0, "H and W must be divisible by block size"

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

                    K_block, b_block = self._solve_single(e_mean, e_hat_mean)

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
        
        return (K_out, b_out)

    def _sample_interpolate(self, quantized, real):
        B, C, H, W = quantized.shape
        assert H % self.blocksize == 0 and W % self.blocksize == 0, "H and W must be divisible by block size"

        size = H // self.blocksize
        quantized_sample = F.interpolate(quantized, size=(size, size), mode='bilinear', align_corners=False)
        real_sample = F.interpolate(real, size=(size, size), mode='bilinear', align_corners=False)

        K, b = self._solve_single(real_sample, quantized_sample)
        K = F.interpolate(K, size=(H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, size=(H, W), mode='bilinear', align_corners=False)
        return (K, b)

    def learn(self, real, quant):
        quant = quant.to(torch.float32)
        real = real.to(torch.float32)
        
        if self.sample_mode == 'block':
            K_out, b_out = self._sample_block(quant, real)
        elif self.sample_mode == 'interpolate':
            K_out, b_out = self._sample_interpolate(quant, real)
        
        K_mean = K_out.mean(dim=0, keepdim=True)
        b_mean = b_out.mean(dim=0, keepdim=True)

        if self.record is None:
            self.record = {
                'sum_K': K_mean.clone(),
                'sum_b': b_mean.clone(),
                'count': 1
            }
        else:
            self.record['sum_K'] += K_mean
            self.record['sum_b'] += b_mean
            self.record['count'] += 1
        
        count = self.record['count']
        mean_K = self.record['sum_K'] / count
        mean_b = self.record['sum_b'] / count

        self.solution = (mean_K, mean_b)
    
    def solve(self, quant):
        K, b = self.solution
        return affine_with_percentile_scale(quant, K, b, self.percentile, self.greater, self.scale, self.verbose)