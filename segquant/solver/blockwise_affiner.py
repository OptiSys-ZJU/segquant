"""
This module implements the BlockwiseAffiner class, which extends the RecurrentSteper
to perform blockwise affine transformations for latent variables during the sampling process.
"""

from segquant.solver.recurrent_steper import RecurrentSteper
from segquant.solver.solver import MSERelSolver


class BlockwiseAffiner(RecurrentSteper):
    """
    Blockwise Affiner for SegQuant.
    This class extends RecurrentSteper to implement a blockwise affine transformation
    for latent variables during the sampling process.
    It uses a specified solver type, which defaults to MSERelSolver.
    Args:
        max_timestep (int): The maximum timestep for the solver.
        blocksize (int): The size of the blocks to process at a time.
        sample_size (int): The number of samples to generate.
        solver_type (str): The type of solver to use, default is "mserel".
        solver_config (dict, optional): Configuration for the solver.
        latents (list, optional): List of latent variables to process.
        recurrent (bool): Whether to use recurrent processing, default is True.
        noise_target (str): Target for noise processing, default is "all".
        enable_latent_affine (bool): Whether to enable affine transformations on latents.
        enable_timesteps (list, optional): Specific timesteps to enable.
        device (str): Device to run the computations on, default is "cuda:0".
    """
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
        """Print the configuration of the BlockwiseAffiner."""
        print("BlockwiseAffiner Configuration:")
        print(f"{'=' * 40}")
        print(f"{'Max timestep:':20} {self.max_timestep}")
        print(f"{'Block size:':20} {self.blocksize}")
        print(f"{'Sample size:':20} {self.sample_size}")
        solver_type_str = (
            "MSERelSolver"
            if isinstance(self.solver[0], MSERelSolver)
            else type(self.solver[0]).__name__
        )
        print(f"{'Solver type:':20} {solver_type_str}")
        print(f"{'Noise target:':20} {self.noise_target}")
        print(f"{'Recurrent:':20} {self.recurrent}")
        print(f"{'Enable latent affine:':20} {self.enable_latent_affine}")
        print(f"{'Enabled timesteps:':20} {self.enable_timesteps}")
        print(f"{'Device:':20} {self.device}")
        print(f"{'Noise preds buffer:':20} {len(self.noise_preds_real)} x deque")
        print(f"{'=' * 40}")

    # save the dict of affiner
    def save_solver_dict(self, path):
        """Save the configuration of the BlockwiseAffiner."""
        self.solver[0].save_dict(path)

    # load the dict of affiner
    def load_solver_dict(self, path):
        """Load the configuration of the BlockwiseAffiner."""
        self.solver[0].load_dict(path)