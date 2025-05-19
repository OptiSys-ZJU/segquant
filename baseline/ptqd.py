import torch
import scipy.stats as stats
from dataset.affine.noise_dataset import NoiseDataset

class PTQD:
    def __init__(self, lambda1=0.5, max_timestep=30):
        self.solutions = {}
        self.cumulative = {}
        self.lambda1 = lambda1
        self.max_timestep = max_timestep
        self.learning = True

    def loss(self, quantized, real):
        return 0
    
    def error(self, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        epsilon_tilde = quantized / (1 + K)
        noise = torch.randn_like(epsilon_tilde) * std_q + mean_q
        epsilon_tilde -= noise

        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - real) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        qnsr_loss = numerator / denominator
        loss = (1 - self.lambda1) * mse_loss + self.lambda1 * qnsr_loss
        return loss
    
    def get_solution(self, timestep):
        return self.solutions[timestep]
    
    def step(self, noise_pred, timestep):
        device = noise_pred.device
        pre_type = noise_pred.dtype
        B, C, H, W = noise_pred.shape

        k, bias, std_q = self.solutions[timestep]
        
        noise_pred = noise_pred.to(dtype=torch.float32)
        k = k.to(device=device, dtype=torch.float32)
        bias = bias.view(1, C, 1, 1).expand(B, C, H, W)
        bias = bias.to(device=device, dtype=torch.float32)
        noise_pred = noise_pred - bias
        noise_pred = noise_pred / (1 + k)

        return noise_pred.to(dtype=pre_type)
    
    def finish_learning(self):
        self.learning = False

    def step_learning(self, timestep, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)
        error = quantized-real

        B, C, H, W = real.shape
        N = C * H * W

        eps = 1e-7

        mean_q = torch.mean(error, dim=(0, 2, 3))

        flatten_data = real.flatten()
        flatten_error = error.flatten()

        slope, intercept, r_value, p_value, std_err = stats.linregress(flatten_data, flatten_error)
        slope = torch.tensor(slope).to(quantized.device)

        flatten_uncorr = flatten_error - slope * flatten_data

        std_q = flatten_uncorr.std(unbiased=False)

        ######
        if self.learning:
            if timestep not in self.cumulative:
                self.cumulative[timestep] = {
                    'sum_K': slope,
                    'sum_mean_q': mean_q,
                    'sum_std_q': std_q,
                    'count': 1
                }
            else:
                self.cumulative[timestep]['sum_K'] += slope
                self.cumulative[timestep]['sum_mean_q'] += mean_q
                self.cumulative[timestep]['sum_std_q'] += std_q
                self.cumulative[timestep]['count'] += 1

            count = self.cumulative[timestep]['count']
            mean_K = self.cumulative[timestep]['sum_K'] / count
            mean_mean_q = self.cumulative[timestep]['sum_mean_q'] / count
            mean_std_q = self.cumulative[timestep]['sum_std_q'] / count
            self.solutions[timestep] = (mean_K, mean_mean_q, mean_std_q)
        return slope, mean_q, std_q
        



if __name__ == '__main__':
    affiner = PTQD(max_timestep=60)
    dataset = NoiseDataset('../dataset/affine_noise')

    ###### learning
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    learning_sample = 1
    step = 0
    for batch in dataloader:
        assert isinstance(batch, list) and len(batch) == dataset.max_timestep
        for data in batch:
            timestep = int(data["timestep"][0].item())
            quant = data["quant"]
            real = data["real"]

            K = torch.zeros_like(real)
            mean_q = 0
            std_q = 0
            init = (affiner.loss(K, mean_q, std_q, quant, real), affiner.error(K, mean_q, std_q, quant, real))

            K, mean_q, std_q = affiner.step_learning(timestep, quant, real)
            affine = (affiner.loss(K, mean_q, std_q, quant, real), affiner.error(K, mean_q, std_q, quant, real))

            print(f'[{timestep}] init: [{init[0]:5f}/{init[1]:5f}], ptqd: [{affine[0]:5f}/{affine[1]:5f}]')
        
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

            K = 0
            mean_q = 0
            std_q = 0
            init = (affiner.loss(K, mean_q, std_q, quant, real), affiner.error(K, mean_q, std_q, quant, real))

            K, mean_q, std_q = affiner.get_solution(timestep)
            affine = (affiner.loss(K, mean_q, std_q, quant, real), affiner.error(K, mean_q, std_q, quant, real))

            K, mean_q, std_q = affiner.step_learning(timestep, quant, real)
            opt = (affiner.loss(K, mean_q, std_q, quant, real), affiner.error(K, mean_q, std_q, quant, real))

            print(f'[{timestep}] init: [{init[0]:5f}/{init[1]:5f}], ptqd: [{affine[0]:5f}/{affine[1]:5f}], opt: [{opt[0]:5f}/{opt[1]:5f}]')
    
        step += 1
        if step >= test_sample:
            break