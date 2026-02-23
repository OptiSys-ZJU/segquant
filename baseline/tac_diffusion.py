import torch
from segquant.solver.recurrent_steper import RecurrentSteper
from segquant.solver.solver import BaseSolver, affine_with_threshold


class TACSolver(BaseSolver):
    def __init__(self, config):
        super().__init__()
        if config is None:
            self.lambda1 = 0.5
            self.lambda2 = 0.1
            self.threshold = 1
            self.verbose = True
        else:
            self.lambda1 = config["lambda1"]
            self.lambda2 = config["lambda2"]
            self.threshold = config["threshold"]
            self.verbose = config["verbose"]

        self.record = None

    def learn(self, real, quant):
        quantized = quant.to(torch.float32)
        real = real.to(torch.float32)

        _, C, H, W = real.shape
        N = C * H * W

        eps = 1e-7
        a = (
            (1 - self.lambda1) * (quantized * real).sum(dim=(2, 3))
            + self.lambda1 * N * (quantized / (real + eps)).sum(dim=(2, 3))
            + self.lambda2 * N
        )
        b = (
            (1 - self.lambda1) * (quantized ** 2).sum(dim=(2, 3))
            + self.lambda1 * N * ((quantized ** 2) / (real ** 2 + eps)).sum(dim=(2, 3))
            + self.lambda2 * N
        )

        K_out = (a / b + eps).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        K_mean = K_out.mean(dim=0, keepdim=True)

        if self.record is None:
            self.record = {"sum_K": K_mean.clone(), "count": 1}
        else:
            self.record["sum_K"] += K_mean
            self.record["count"] += 1

        count = self.record["count"]
        mean_K = self.record["sum_K"] / count

        self.solution = mean_K

    def solve(self, quant):
        K = self.solution
        return affine_with_threshold(
            quant, K, threshold=self.threshold, verbose=self.verbose
        )


class TACDiffution(RecurrentSteper):
    @classmethod
    def from_config(cls, config, latents=None, device="cuda:0"):
        assert config["type"] == "tac", "only tac stepper worked"
        return cls(
            max_timestep=config["config"]["max_timestep"],
            sample_size=config["config"]["sample_size"],
            solver_type=config["config"]["solver_type"],
            solver_config=config["config"]["solver_config"],
            latents=latents,
            recurrent=config["config"]["recurrent"],
            noise_target=config["config"]["noise_target"],
            enable_latent_affine=config["config"]["enable_latent_affine"],
            enable_timesteps=config["config"]["enable_timesteps"],
            device=device,
        )

    def __init__(
        self,
        max_timestep,
        sample_size=1,
        solver_type="tac",
        solver_config=None,
        latents=None,
        recurrent=True,
        noise_target="all",
        enable_latent_affine=False,
        enable_timesteps=None,
        device="cuda:0",
    ):
        assert solver_type == "tac", "only tac solver worked"
        if solver_type == "tac":
            super().__init__(
                max_timestep,
                sample_size,
                TACSolver,
                solver_config,
                latents,
                recurrent,
                noise_target,
                enable_latent_affine,
                enable_timesteps,
                device,
            )
            self.solver_config = solver_config
        else:
            raise ValueError()
        self.print_config()
    
    def dump(self, path):
        """Save the learned state of the TACSolver stepper."""
        super_dict = super().dump()
        super_dict['config']['solver_type'] = 'tac'
        super_dict['config']['solver_config'] = self.solver_config
        torch.save(super_dict, path)
        print(f"TACDiffution stepper record saved to {path}")

    def print_config(self):
        print("TACDiffution Configuration:")
        print(f"{'=' * 40}")
        print(f"{'Max timestep:':20} {self.max_timestep}")
        print(f"{'Sample size:':20} {self.sample_size}")
        print(f"{'Noise target:':20} {self.noise_target}")
        print(f"{'Recurrent:':20} {self.recurrent}")
        print(f"{'Enable latent affine:':20} {self.enable_latent_affine}")
        print(f"{'Enabled timesteps:':20} {self.enable_timesteps}")
        print(f"{'Device:':20} {self.device}")
        print(f"{'Noise preds buffer:':20} {len(self.noise_preds_real)} x deque")
        print(f"{'=' * 40}")
