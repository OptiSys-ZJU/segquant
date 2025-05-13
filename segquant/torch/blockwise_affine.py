import torch

from dataset.affine.noise_dataset import NoiseDataset

class BlockwiseAffiner:
    def __init__(self, blocksize=128, alpha=0.5, lambda1=0.1, lambda2=0.1, max_timestep=30):
        self.solutions = {}  # timestep -> (mean_K, mean_b)
        self.cumulative = {}  # timestep -> {'sum_K': ..., 'sum_b': ..., 'count': ...}
        self.blocksize = blocksize
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_timestep = max_timestep

    def loss(self, K, b, quantized, real):
        epsilon_tilde = K * quantized + b
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - quantized) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        rel_loss = numerator / denominator
        k_penalty = torch.mean((K - 1.0) ** 2)
        b_penalty = torch.mean(b ** 2)
        loss = self.alpha * mse_loss + (1 - self.alpha) * rel_loss + self.lambda1 * k_penalty + self.lambda2 * b_penalty
        return loss
    
    def error(self, K, b, quantized, real):
        epsilon_tilde = K * quantized + b
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - quantized) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        rel_loss = numerator / denominator
        loss = self.alpha * mse_loss + (1 - self.alpha) * rel_loss
        return loss

    def get_solution(self, timestep):
        return self.solutions[timestep]

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

                e_hat_mean = e_hat_block.mean(dim=(0, 2, 3))  # shape: [C]
                e_mean = e_block.mean(dim=(0, 2, 3))           # shape: [C]
                A = self.alpha + (1 - self.alpha) / (e_mean ** 2 + 1e-6)  # shape: [C]
                delta = e_mean - e_hat_mean                               # shape: [C]
                denominator = 1 + (self.lambda2 * e_hat_mean ** 2) / self.lambda1 + self.lambda2 / A
                # denominator = torch.clamp(denominator, min=1e-4)          # shape: [C]
                b_block = delta / denominator                             # shape: [C]
                K_block = 1 + (self.lambda2 * e_hat_mean / self.lambda1) * b_block  # shape: [C]
                b_block_full = b_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)
                K_block_full = K_block.view(1, C, 1, 1).expand(B, C, self.blocksize, self.blocksize)
                K_out[:, :, h_start:h_end, w_start:w_end] = K_block_full
                b_out[:, :, h_start:h_end, w_start:w_end] = b_block_full

        K_mean = K_out.mean(dim=0, keepdim=True)
        b_mean = b_out.mean(dim=0, keepdim=True)

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


if __name__ == '__main__':
    blocksizes = [1, 2, 4, 8, 16, 32, 64, 128]
    # blocksizes = [1]
    affiners = [BlockwiseAffiner(max_timestep=60, blocksize=block) for block in blocksizes]
    dataset = NoiseDataset('../dataset/affine_noise')

    ###### learning
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    learning_sample = 8
    step = 0
    for batch in dataloader:
        assert isinstance(batch, list) and len(batch) == dataset.max_timestep
        for data in batch:
            timestep = int(data["timestep"][0].item())
            quant = data["quant"]
            real = data["real"]

            for affiner in affiners:
                K = torch.ones_like(real)
                b = torch.zeros_like(real)
                init = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                K, b = affiner.step_learning(timestep, quant, real)
                affine = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                print(f'[{timestep}] block[{affiner.blocksize}] init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}]')
        
        step += 1
        if step >= learning_sample:
            break
    
    ##### testing
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=True)
    test_sample = 2
    step = 0
    for batch in dataloader:
        assert isinstance(batch, list) and len(batch) == dataset.max_timestep
        for data in batch:
            timestep = int(data["timestep"][0].item())
            quant = data["quant"]
            real = data["real"]

            for affiner in affiners:
                K = torch.ones_like(real)
                b = torch.zeros_like(real)
                init = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                K, b = affiner.get_solution(timestep)
                affine = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                K, b = affiner.step_learning(timestep, quant, real)
                opt = (affiner.loss(K, b, quant, real), affiner.error(K, b, quant, real))

                print(f'[{timestep}] block[{affiner.blocksize}] init: [{init[0]:5f}/{init[1]:5f}], affine: [{affine[0]:5f}/{affine[1]:5f}], opt: [{opt[0]:5f}/{opt[1]:5f}]')
        
        step += 1
        if step >= test_sample:
            break




