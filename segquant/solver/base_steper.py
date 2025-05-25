class BaseSteper:
    def __init__(self, max_timestep):
        self.max_timestep = max_timestep
        self.scheduler = None

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler

    def step_forward(
        self,
        latents,
        noise_pred,
        i,
        t,
        num_inference_steps,
        do_classifier_free_guidance,
        guidance_scale,
    ):
        if self.scheduler is None:
            raise ValueError("[Error] BaseSolver: scheduler not register")

        if self.max_timestep != num_inference_steps:
            print(
                f"[Warning] BaseSolver: max_timestep[{self.max_timestep}] != num_inference_steps[{num_inference_steps}]"
            )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = self.scheduler.step(noise_pred, t, latents)[0]

        return latents
