"""
This module defines the BaseSteper class, which serves as a base class
for performing a single step in a diffusion process using a registered scheduler.
"""

class BaseSteper:
    """
    Base class for a steper in a diffusion process.
    This class is responsible for performing a single step of the diffusion process
    using a registered scheduler.
    """
    def __init__(self, max_timestep):
        self.max_timestep = max_timestep
        self.scheduler = None

    def register_scheduler(self, scheduler):
        """
        Register a scheduler for the steper.
        Args:
            scheduler: Scheduler instance to be registered.
        """
        self.scheduler = scheduler

    def step_forward(
        self,
        latents: any,
        noise_pred: any,
        _i: int,
        t: float,
        num_inference_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
    ) -> any:
        """
        Perform a single step of the diffusion process.
        Args:
            latents (any): Current latent representation.
            noise_pred (any): Predicted noise.
            t (float): Current timestep.
            num_inference_steps (int): Total number of inference steps.
            do_classifier_free_guidance (bool): Whether to apply classifier-free guidance.
            guidance_scale (float): Scale for guidance.
        Returns:
            any: Updated latent representation after the step.
        Raises:
            ValueError: If the scheduler is not registered or if the max_timestep
                        does not match num_inference_steps.
        """
        if self.scheduler is None:
            raise ValueError("[Error] BaseSolver: scheduler not register")

        if self.max_timestep != num_inference_steps:
            print(
                f"[Warning] BaseSolver: max_timestep[{self.max_timestep}] "
                f"!= num_inference_steps[{num_inference_steps}]"
            )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = self.scheduler.step(noise_pred, t, latents)[0]

        return latents
