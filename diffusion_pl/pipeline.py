import torch
from tqdm import tqdm
from functools import partial
from diffusers import DDIMScheduler

class SR3scheduler(DDIMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, beta_schedule: str = 'linear', diff_chns=3):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, prediction_type="sample")
        self.diff_chns = diff_chns  # Number of channels to apply noise to (RGB = 3)

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        # check 
        noise = torch.clamp(noise, -1.0, 1.0)
        # original_samples = torch.clamp(original_samples, 0.0, 1.0)
        
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # check
        sqrt_alpha_prod = torch.clamp(alphas_cumprod[timesteps] ** 0.5, 1e-8, 1.0)
        sqrt_one_minus_alpha_prod = torch.clamp((1 - alphas_cumprod[timesteps]) ** 0.5, 1e-8, 1.0)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Selective noise addition to the last `diff_chns` channels (e.g., RGB)
        num_channels = original_samples.shape[1]
        if num_channels > self.diff_chns:
            original_samples_select = original_samples[:, -self.diff_chns:].contiguous()
            noise_select = noise[:, -self.diff_chns:].contiguous()
            noisy_samples_select = sqrt_alpha_prod * original_samples_select + sqrt_one_minus_alpha_prod * noise_select
            noisy_samples = original_samples.clone()
            noisy_samples[:, -self.diff_chns:] = noisy_samples_select
        else:
            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

def create_SR3scheduler(opt, phase):
    steps = opt.num_train_timesteps if phase == "train" else opt.sampling_timesteps
    scheduler = SR3scheduler(
        num_train_timesteps=steps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        diff_chns=3  # Fixed to 3 for RGB channels
    )
    return scheduler

class SR3Sampler:
    def __init__(self, model: torch.nn.Module, scheduler: SR3scheduler, eta: float = 0.0):
        self.model = model
        self.scheduler = scheduler
        self.eta = eta

    def sample_high_res(self, low_res: torch.Tensor, conditions=None, clean_structure=None, guidance=None, train=False, quality_condition=None, raw_quality_scores=None):
        device = next(self.model.parameters()).device
        eta_ddim = self.eta  # Assuming eta is a class attribute; adjust if it's passed differently
        b, c, h, w = low_res.shape  # e.g., [4, 3, 256, 256]
        
        # Create 6-channel input: concatenate low_res with zero residual
        zero_residual = torch.zeros_like(low_res, device=device)
        combined_input = torch.cat([low_res, zero_residual], dim=1)  # [4, 6, 256, 256]
        
        # Initialize noisy image: low_res + random noise for residual
        img = torch.zeros_like(combined_input, device=device)
        img[:, :3] = low_res  # First 3 channels are the low-res image
        noise = torch.randn_like(low_res, device=device)
        noise = torch.clamp(noise, -1.0, 1.0)  # 限制噪声范围
        img[:, 3:6] = torch.randn_like(low_res, device=device)  # Last 3 channels are noise
        
        # Set timesteps for the scheduler
        self.scheduler.set_timesteps(self.scheduler.num_train_timesteps if train else len(self.scheduler.timesteps))
        
        # Sampling loop
        for t in tqdm(self.scheduler.timesteps, desc="Sampling" if not train else "Training Sampling"):
            timesteps = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Model predicts the residual (noise) for the last 3 channels
            model_output_dict = self.model(
                inp=img,
                time=timesteps,
                quality_condition=quality_condition,
                raw_quality_scores=raw_quality_scores,
            )
            predicted_residual = model_output_dict.get('predicted_residual')  # [4, 3, 256, 256]
            
            # Extract the residual part of img to pass to the scheduler
            img_residual = img[:, 3:6]  # [4, 3, 256, 256]
            
            # Update residual using the scheduler
            scheduler_output = self.scheduler.step(
                model_output=predicted_residual,  # [4, 3, 256, 256] - predicted noise/residual
                timestep=t,
                sample=img_residual,  # [4, 3, 256, 256] - current residual
                eta=eta_ddim,
                return_dict=True
            )
            
            # Update img with the new residual, keeping low_res intact
            img[:, 3:6] = scheduler_output.prev_sample  # Update only the residual channels
        
        # Final output: the residual part after sampling
        final_residual = img[:, 3:6]  # [4, 3, 256, 256]
        
        # Handle prompt loss if applicable
        prompt_loss = model_output_dict.get('prompt_loss', torch.tensor(0.0, device=device))
        avg_prompt_loss = prompt_loss if train else prompt_loss / len(self.scheduler.timesteps)
        
        return {'sample': final_residual, 'prompt_loss': avg_prompt_loss}


def create_SR3Sampler(model, opt):
    scheduler = create_SR3scheduler(opt, "test")
    scheduler.set_timesteps(opt.sampling_timesteps)
    sampler = SR3Sampler(
        model=model,
        scheduler=scheduler,
        eta=opt.eta
    )
    return sampler
