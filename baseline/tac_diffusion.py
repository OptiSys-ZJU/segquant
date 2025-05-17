import torch

from dataset.affine.noise_dataset import NoiseDataset

class TACDiffution:
    def __init__(self, lambda1=0.5, lambda2=0.1, max_timestep=30):
        self.solutions = {}
        self.cumulative = {}
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_timestep = max_timestep
        self.learning = True

    def loss(self, K, b, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        epsilon_tilde = K * quantized
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - real) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        qnsr_loss = numerator / denominator
        k_penalty = torch.mean((K - 1.0) ** 2)
        loss = (1 - self.lambda1) * mse_loss + self.lambda1 * qnsr_loss + self.lambda2 * k_penalty
        return loss
    
    def error(self, K, b, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        epsilon_tilde = K * quantized
        mse_loss = torch.mean((epsilon_tilde - real) ** 2)
        numerator = torch.sum((epsilon_tilde - real) ** 2)
        denominator = torch.sum(real ** 2) + 1e-8
        qnsr_loss = numerator / denominator
        loss = (1 - self.lambda1) * mse_loss + self.lambda1 * qnsr_loss
        return loss

    def get_solution(self, timestep, example=None):
        if self.learning:
            if timestep in self.solutions:
                return (self.solutions[timestep], torch.zeros_like(self.solutions[timestep]))
            else:
                print(f'Warning: [{timestep}] get default solution')
                return (torch.ones_like(example), torch.zeros_like(example))
        else:
            return (self.solutions[timestep], torch.zeros_like(self.solutions[timestep]))
    
    def finish_learning(self):
        self.learning = False

    def step_learning(self, timestep, quantized, real):
        quantized = quantized.to(torch.float32)
        real = real.to(torch.float32)

        B, C, H, W = real.shape
        N = C * H * W

        eps = 1e-7
        a = (1 - self.lambda1) * (quantized * real).sum(dim=(2, 3)) + \
            self.lambda1 * N * (quantized / (real + eps)).sum(dim=(2, 3)) + \
            self.lambda2 * N
        b = (1 - self.lambda1) * (quantized ** 2).sum(dim=(2, 3)) + \
            self.lambda1 * N * ((quantized ** 2) / (real ** 2 + eps)).sum(dim=(2, 3)) + \
            self.lambda2 * N
        
        K_out = (a / b + eps).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        K_mean = K_out.mean(dim=0, keepdim=True)

        if self.learning:
            if timestep not in self.cumulative:
                self.cumulative[timestep] = {
                    'sum_K': K_mean.clone(),
                    'count': 1
                }
            else:
                self.cumulative[timestep]['sum_K'] += K_mean
                self.cumulative[timestep]['count'] += 1

            count = self.cumulative[timestep]['count']
            mean_K = self.cumulative[timestep]['sum_K'] / count

            self.solutions[timestep] = mean_K
        return (K_mean, torch.zeros_like(K_mean))


if __name__ == '__main__':
    affiner = TACDiffution(max_timestep=60)
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

            K = torch.ones_like(real)
            init = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.step_learning(timestep, quant, real)
            affine = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            print(f'[{timestep}] init: [{init[0]:5f}/{init[1]:5f}], tac: [{affine[0]:5f}/{affine[1]:5f}]')
        
        step += 1
        if step >= learning_sample:
            break
    
    affiner.finish_learning()

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

            K = torch.ones_like(real)
            init = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.get_solution(timestep)
            affine = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            K = affiner.step_learning(timestep, quant, real)
            opt = (affiner.loss(K, quant, real), affiner.error(K, quant, real))

            print(f'[{timestep}] init: [{init[0]:5f}/{init[1]:5f}], tac: [{affine[0]:5f}/{affine[1]:5f}], opt: [{opt[0]:5f}/{opt[1]:5f}]')
    
        step += 1
        if step >= test_sample:
            break