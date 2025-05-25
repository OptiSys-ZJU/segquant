from segquant.solver.recurrent_steper import RecurrentSteper
from segquant.solver.solver import MSERelSolver


class BlockwiseAffiner(RecurrentSteper):
    def __init__(
        self,
        max_timestep,
        blocksize=1,
        sample_size=1,
        solver_type="mserel",
        solver_config=None,
        latents=None,
        recurrent=True,
        noise_target="all",
        enable_latent_affine=False,
        enable_timesteps=None,
        device="cuda:0",
    ):
        if solver_type == "mserel":
            super().__init__(
                max_timestep,
                sample_size,
                MSERelSolver,
                solver_config,
                latents,
                recurrent,
                noise_target,
                enable_latent_affine,
                enable_timesteps,
                device,
            )
        else:
            raise ValueError()
        self.blocksize = blocksize
        self.print_config()

    def print_config(self):
        print("BlockwiseAffiner Configuration:")
        print(f"{'=' * 40}")
        print(f"{'Max timestep:':20} {self.max_timestep}")
        print(f"{'Block size:':20} {self.blocksize}")
        print(f"{'Sample size:':20} {self.sample_size}")
        print(
            f"{'Solver type:':20} {'MSERelSolver' if isinstance(self.solver[0], MSERelSolver) else type(self.solver[0]).__name__}"
        )
        print(f"{'Noise target:':20} {self.noise_target}")
        print(f"{'Recurrent:':20} {self.recurrent}")
        print(f"{'Enable latent affine:':20} {self.enable_latent_affine}")
        print(f"{'Enabled timesteps:':20} {self.enable_timesteps}")
        print(f"{'Device:':20} {self.device}")
        print(f"{'Noise preds buffer:':20} {len(self.noise_preds_real)} x deque")
        print(f"{'=' * 40}")
