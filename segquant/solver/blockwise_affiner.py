
import torch
from segquant.solver.base_steper import BaseSteper
from segquant.solver.solver import MSERelSolver
from segquant.solver.state import Stage, StateMachine, solver_trans
from collections import deque

class BlockwiseAffiner(BaseSteper):
    def __init__(self, max_timestep, blocksize=1, sample_size=1, solver_type='mserel', solver_config=None, latents=None, recurrent=True, noise_target='all', enable_latent_affine=False, enable_timesteps=None, device='cuda:0'):
        super().__init__(max_timestep)
        self.blocksize = blocksize

        if solver_type == 'mserel':
            self.solver = [MSERelSolver(solver_config) for _ in range(max_timestep)]
        
        self.fsm = StateMachine() # 'init' -> ['learning_real' -> 'learning_quant']+ -> 'final'

        self.sample_size = sample_size

        self.noise_target = noise_target # 'all' 'uncond' 'text'
        self.noise_preds_real = [deque() for _ in range(max_timestep)] # (real, quant)

        self.recurrent = recurrent
        self.recurrent_index = 0
        self.recurrent_latents = [(latents, latents) for _ in range(self.sample_size)]
        self.replay = False

        self.enable_latent_affine = enable_latent_affine
        assert not self.enable_latent_affine, 'enable_latent_affine not work now'
        self.latent_record = [[], []] # real / quant

        if enable_timesteps is None:
            self.enable_timesteps = [i for i in range(max_timestep)]
        else:
            self.enable_timesteps = enable_timesteps
        
        self.device = device

        self.print_config()
    
    def print_config(self):
        print("BlockwiseAffiner Configuration:")
        print(f"{'=' * 40}")
        print(f"{'Max timestep:':20} {self.max_timestep}")
        print(f"{'Block size:':20} {self.blocksize}")
        print(f"{'Sample size:':20} {self.sample_size}")
        print(f"{'Solver type:':20} {'MSERelSolver' if isinstance(self.solver[0], MSERelSolver) else type(self.solver[0]).__name__}")
        print(f"{'Noise target:':20} {self.noise_target}")
        print(f"{'Recurrent:':20} {self.recurrent}")
        print(f"{'Enable latent affine:':20} {self.enable_latent_affine}")
        print(f"{'Enabled timesteps:':20} {self.enable_timesteps}")
        print(f"{'Device:':20} {self.device}")
        print(f"{'Noise preds buffer:':20} {len(self.noise_preds_real)} x deque")
        print(f"{'=' * 40}")

    def learning(self, model, data_loader, **kwargs):
        model.to(torch.device(self.device))
        self.sample_count = 0

        def run():
            for batch in data_loader:
                batch_size = len(batch)
                for i in range(batch_size):
                    if self.sample_count >= self.sample_size:
                        return
                    with torch.no_grad():
                        prompt, _, control = batch[i]
                        if not self.recurrent:
                            model(prompt=prompt, control_image=control, num_inference_steps=self.max_timestep, steper=self, **kwargs)
                        else:
                            if 'latents' in kwargs:
                                del kwargs['latents']
                            model(prompt=prompt, 
                                    control_image=control, 
                                    num_inference_steps=self.max_timestep, 
                                    steper=self,
                                    latents=self.recurrent_latents[self.sample_count][0],
                                    replay_timestep=self.recurrent_index,
                                    early_stop=self.recurrent_index+1,
                                    **kwargs)
                    self.sample_count += 1
        run()
        model.to(torch.device('cpu'))

    @solver_trans([Stage.INIT, Stage.WAIT_QUANT], Stage.WAIT_REAL)
    def learning_real(self, model_real, data_loader, **kwargs):
        self.learning(model_real, data_loader, **kwargs)

    @solver_trans([Stage.WAIT_REAL], Stage.WAIT_QUANT)
    def learning_quant(self, model_quant, data_loader, **kwargs):
        self.learning(model_quant, data_loader, **kwargs)

    def _replay(self, model, data_loader, **kwargs):
        model.to(torch.device(self.device))
        if self.recurrent:
            if 'latents' in kwargs:
                del kwargs['latents']
            self.sample_count = 0
            def run():
                for batch in data_loader:
                    batch_size = len(batch)
                    for i in range(batch_size):
                        if self.sample_count >= self.sample_size:
                            return
                        with torch.no_grad():
                            prompt, _, control = batch[i]
                            model(prompt=prompt, 
                                        control_image=control, 
                                        num_inference_steps=self.max_timestep, 
                                        steper=self,
                                        latents=self.recurrent_latents[self.sample_count][0],
                                        replay_timestep=self.recurrent_index,
                                        early_stop=self.recurrent_index+1,
                                        **kwargs)
                        self.sample_count += 1
            run()
        else:
            print('[Warning] Recurrent disabled, replay not work.')
        
        model.to(torch.device('cpu'))

    @solver_trans([Stage.WAIT_QUANT], Stage.WAIT_REAL)
    def replay_real(self, model_real, data_loader, **kwargs):
        self.replay = True
        self._replay(model_real, data_loader, **kwargs)
    
    @solver_trans([Stage.WAIT_REAL], Stage.WAIT_QUANT)
    def replay_quant(self, model_quant, data_loader, **kwargs):
        self._replay(model_quant, data_loader, **kwargs)
        self.replay = False
        self.recurrent_index += 1

    @solver_trans([Stage.WAIT_QUANT], Stage.FINAL)
    def finish_learning(self):
        print('[INFO] BlockwiseAffiner: Finish learning')

    def _get_noise(self, noise_pred, noise_target, do_classifier_free_guidance, guidance_scale):
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if noise_target == 'all':
                return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            elif noise_target == 'uncond':
                return noise_pred_uncond
            elif noise_target == 'text':
                return noise_pred_text
            else:
                raise ValueError(f'[Error] _get_noise({noise_target})')
        else:
            if noise_target == 'all':
                return noise_pred
            else:
                raise ValueError(f'[Error] _get_noise({noise_target})')
    
    def _reconstruct_noise(self, i, noise_pred, noise_target, do_classifier_free_guidance, guidance_scale):
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if noise_target == 'all':
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                return self.solver[i].solve(noise_pred)
            elif noise_target == 'uncond':
                noise_pred_uncond = self.solver[i].solve(noise_pred_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                return noise_pred
            elif noise_target == 'text':
                noise_pred_text = self.solver[i].solve(noise_pred_text)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                return noise_pred
            else:
                raise ValueError(f'[Error] _reconstruct_noise({noise_target})')
        else:
            if noise_target == 'all':
                return self.solver[i].solve(noise_pred)
            else:
                raise ValueError(f'[Error] _reconstruct_noise({noise_target})')

    def step_forward(self, latents, noise_pred, i, t, num_inference_steps, do_classifier_free_guidance, guidance_scale):
        if self.scheduler is None:
            raise ValueError('[Error] BlockwiseAffiner: scheduler not register')
        
        if self.max_timestep != num_inference_steps:
            print(f'[Warning] BlockwiseAffiner: max_timestep[{self.max_timestep}] != num_inference_steps[{num_inference_steps}]')
        
        recon = False
        if i in self.enable_timesteps:
            if self.fsm.state == Stage.FINAL or (self.recurrent and self.replay):
                recon = True
                noise_pred = self._reconstruct_noise(i, noise_pred=noise_pred, noise_target=self.noise_target, do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale)
                latents = self.scheduler.step(noise_pred, t, latents)[0]
            else:
                this_noise_pred = self._get_noise(noise_pred, noise_target=self.noise_target, do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale)
                if self.fsm.state == Stage.WAIT_REAL:
                    self.noise_preds_real[i].append(this_noise_pred)
                elif self.fsm.state == Stage.WAIT_QUANT:
                    self.solver[i].learn(self.noise_preds_real[i].popleft(), this_noise_pred)
                else:
                    raise RuntimeError(f'[BlockwiseAffiner] invalid FSM state[{self.fsm.state}]')

        if not recon:
            # get normal result
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        if self.recurrent and self.replay:
            if self.fsm.state == Stage.WAIT_REAL:
                self.recurrent_latents[self.sample_count] = (latents, self.recurrent_latents[self.sample_count][1])
            elif self.fsm.state == Stage.WAIT_QUANT:
                self.recurrent_latents[self.sample_count] = (self.recurrent_latents[self.sample_count][0], latents)

        return latents
