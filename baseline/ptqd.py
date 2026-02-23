import torch
from scipy import stats
from segquant.solver.recurrent_steper import RecurrentSteper
from segquant.solver.solver import BaseSolver


class PTQDSolver(BaseSolver):
    def __init__(self, _config):
        super().__init__()

        self.record = None

    def learn(self, real, quant):
        quantized = quant.to(torch.float32)
        real = real.to(torch.float32)
        error = quantized - real

        mean_q = torch.mean(error, dim=(0, 2, 3))

        flatten_data = real.flatten()
        flatten_error = error.flatten()

        slope, _intercept, _r_value, _p_value, _std_err = stats.linregress(
            flatten_data.cpu().numpy(),
            flatten_error.cpu().numpy()
        )
        slope = torch.tensor(slope).to(quantized.device)

        flatten_uncorr = flatten_error - slope * flatten_data

        std_q = flatten_uncorr.std(unbiased=False)

        if self.record is None:
            self.record = {
                "sum_K": slope,
                "sum_mean_q": mean_q,
                "sum_std_q": std_q,
                "count": 1,
            }
        else:
            self.record["sum_K"] += slope
            self.record["sum_mean_q"] += mean_q
            self.record["sum_std_q"] += std_q
            self.record["count"] += 1

        count = self.record["count"]
        mean_K = self.record["sum_K"] / count
        mean_mean_q = self.record["sum_mean_q"] / count
        mean_std_q = self.record["sum_std_q"] / count
        self.solution = (mean_K, mean_mean_q, mean_std_q)

    def solve(self, quant):
        device = quant.device
        pre_type = quant.dtype
        B, C, H, W = quant.shape

        k, bias, _std_q = self.solution

        quant = quant.to(dtype=torch.float32)
        k = k.to(device=device, dtype=torch.float32)
        bias = bias.view(1, C, 1, 1).expand(B, C, H, W)
        bias = bias.to(device=device, dtype=torch.float32)
        quant = quant - bias
        quant = quant / (1 + k)

        return quant.to(dtype=pre_type)


class PTQD(RecurrentSteper):
    @classmethod
    def from_config(cls, config, latents=None, device="cuda:0"):
        assert config['type'] == "ptqd", "Config type must be 'ptqd' for PTQD stepper."
        return cls(
            max_timestep=config['config']["max_timestep"],
            sample_size=config['config']["sample_size"],
            latents=latents,
            recurrent=config["config"]["recurrent"],
            noise_target=config['config']["noise_target"],
            enable_timesteps=config['config']["enable_timesteps"],
            device=device,
        )

    def __init__(
        self,
        max_timestep,
        sample_size=1,
        latents=None,
        recurrent=True,
        noise_target="all",
        enable_timesteps=None,
        device="cuda:0",
    ):
        super().__init__(
            max_timestep,
            sample_size,
            PTQDSolver,
            None,
            latents,
            recurrent,
            noise_target,
            False,
            enable_timesteps,
            device,
        )
        self.print_config()

    def dump(self, path):
        """Save the learned state of the PTQD stepper."""
        super_dict = super().dump()
        super_dict['config']['solver_type'] = 'ptqd'
        super_dict['config']['solver_config'] = None
        torch.save(super_dict, path)
        print(f"PTQD stepper record saved to {path}")

    def print_config(self):
        print("PTQD Configuration:")
        print(f"{'=' * 40}")
        print(f"{'Max timestep:':20} {self.max_timestep}")
        print(f"{'Sample size:':20} {self.sample_size}")
        print(f"{'Noise target:':20} {self.noise_target}")
        print(f"{'Recurrent:':20} {self.recurrent}")
        print(f"{'Enabled timesteps:':20} {self.enable_timesteps}")
        print(f"{'Device:':20} {self.device}")
        print(f"{'Noise preds buffer:':20} {len(self.noise_preds_real)} x deque")
        print(f"{'=' * 40}")
