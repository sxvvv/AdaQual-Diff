import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import lpips
from torch.utils.data import DataLoader
from DeQAScore.src.evaluate.scorer import Scorer
from loss.CL1 import PSNRLoss
from models_prompt.modules.DenoisingNAFNet_arch import NAFNet
from Utils.Allweather import Allweather, Snow100kTest, Test1
from Utils.AGAN_data import AGAN_Dataset
from Utils.imgqual_utils import batch_PSNR, batch_SSIM
from Utils.save_image import save_colormapped_image
import diffusion_pl.pipeline as pipeline
from einops import rearrange, reduce
import math
import numpy as np
import torch.optim as optim
from diffusion_pl.ema import EMA
from peft import LoraConfig, get_peft_model

def gaussian_blur(tensor, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to a tensor."""
    b, c, h, w = tensor.shape
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=tensor.device)
    kernel = torch.exp(-((torch.arange(kernel_size, device=tensor.device) - kernel_size // 2) ** 2) / (2 * sigma ** 2))
    kernel = kernel.view(1, 1, kernel_size, 1) * kernel.view(1, 1, 1, kernel_size)
    kernel = kernel / kernel.sum()
    
    padding = kernel_size // 2
    tensor_padded = F.pad(tensor, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(tensor_padded, kernel, groups=c)

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))

class MaskingQualityModule(nn.Module):
    def __init__(self, channels, mask_ratio_min=0.3, mask_ratio_max=0.8, boundary_sigma=0.1, 
                 deqa_scorer=None, smooth_kernel_size=5, smooth_sigma=1.0, use_lora=False, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.channels = channels
        self.boundary_sigma = boundary_sigma
        self.deqa_scorer = deqa_scorer
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        
        self.input_adapt = nn.ModuleDict({
            'c3': nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1), nn.SiLU()),
            'c6': nn.Sequential(nn.Conv2d(6, channels, kernel_size=3, padding=1), nn.SiLU())
        })
        # 定义 quality_net
        self.quality_net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels//2, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(channels//2, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )
        
        # 添加 LoRA（可选）
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["0", "2"],  # 针对 quality_net 中的第 0 和第 2 个 Conv2d 层
                lora_dropout=lora_dropout,
                task_type="FEATURE_EXTRACTOR"  # 自定义任务类型
            )
            self.quality_net = get_peft_model(self.quality_net, lora_config)
            print("LoRA applied to quality_net:")
            self.quality_net.print_trainable_parameters()
    
    def extract_deqa_quality_map(self, x):
        """Extract and smooth a spatial quality map from DeQA."""
        b, c, h, w = x.shape
        
        
        if c != 3:
            print(f"Warning: Input tensor to extract_deqa_quality_map has {c} channels. Converting to compatible format.")
            
            default_quality = 0.5  
            deqa_quality = torch.ones((b, 1, h, w), device=x.device) * default_quality
            return deqa_quality
        
        # 处理 RGB 图像
        try:
            pil_images = [Image.fromarray((x[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)) for i in range(b)]
            with torch.no_grad():
                # Use DeQA scalar score and expand spatially
                deqa_scores = torch.tensor(self.deqa_scorer(pil_images), device=x.device).view(b, 1, 1, 1)
                # Expand to spatial dimensions and smooth
                deqa_quality = deqa_scores.expand(b, 1, h, w)
                deqa_quality = gaussian_blur(deqa_quality, self.smooth_kernel_size, self.smooth_sigma)
            return deqa_quality
        except Exception as e:
            print(f"Error in extract_deqa_quality_map: {e}")
            print(f"Input shape: {x.shape}, values range: {x.min().item()}-{x.max().item()}")
            
            deqa_quality = torch.ones((b, 1, h, w), device=x.device) * 0.5
            return deqa_quality

    def compute_gradient(self, quality_map):
        """Compute smoothed gradients of the quality map."""
        grad_x = torch.abs(F.conv2d(quality_map, torch.tensor([[-1, 0, 1]], dtype=torch.float, device=quality_map.device).view(1, 1, 1, 3), padding=(0, 1)))
        grad_y = torch.abs(F.conv2d(quality_map, torch.tensor([[-1], [0], [1]], dtype=torch.float, device=quality_map.device).view(1, 1, 3, 1), padding=(1, 0)))
        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gaussian_blur(gradient, self.smooth_kernel_size, self.smooth_sigma)

    def forward(self, x, timestep=None):
        #
        original_x = x
        original_channels = x.shape[1]
        is_rgb = (original_channels == 3)
        
        in_channels = x.shape[1]
        if in_channels != self.channels:
            if in_channels == 3:
                x = self.input_adapt['c3'](x)
            elif in_channels == 6:
                x = self.input_adapt['c6'](x)
            else:
                adapter = nn.Conv2d(in_channels, self.channels, kernel_size=1).to(x.device)
                x = adapter(x)
        
        # 
        if is_rgb:
            deqa_quality = self.extract_deqa_quality_map(original_x)
        else:
            # 
            b, _, h, w = x.shape
            deqa_quality = torch.ones((b, 1, h, w), device=x.device) * 0.5
        
        x_with_deqa = torch.cat([x, deqa_quality], dim=1)
        quality_map = self.quality_net(x_with_deqa).float()
        quality_map = gaussian_blur(quality_map, self.smooth_kernel_size, self.smooth_sigma)  # Additional smoothing
        
        # Time step adjustment
        if timestep is not None:
            t_factor = 1.0 - timestep.float() / 1000.0 if isinstance(timestep, torch.Tensor) else torch.tensor(1.0 - timestep / 1000.0).to(x.device)
            mask_ratio = self.mask_ratio_min + (self.mask_ratio_max - self.mask_ratio_min) * t_factor
        else:
            mask_ratio = self.mask_ratio_max
        
        batch_size = x.shape[0]
        attention_masks = []
        boundary_weights = []
        
        for i in range(batch_size):
            inverted_quality = (1.0 - quality_map[i]).float()
            mask_ratio_val = mask_ratio.mean().item() if isinstance(mask_ratio, torch.Tensor) else float(mask_ratio)
            
            if mask_ratio_val < 1.0:
                threshold = torch.quantile(inverted_quality.flatten(), 1.0 - mask_ratio_val)
                mask = (inverted_quality > threshold).float()
            else:
                mask = torch.ones_like(inverted_quality)
            
            gradient = self.compute_gradient(quality_map[i:i+1])
            boundary_weight = 1.0 - torch.exp(-gradient**2 / self.boundary_sigma**2)
            boundary_weights.append(boundary_weight)
            attention_masks.append(mask)
        
        attention_mask = torch.stack(attention_masks, dim=0)
        boundary_weight = torch.stack(boundary_weights, dim=0)
        
        return attention_mask, quality_map, boundary_weight


class QualityAwareDiffusionSampler(pipeline.SR3Sampler):
    """Diffusion sampler with mask-guided autoregressive acceleration and quality-aware enhancements"""
    def __init__(self, model, scheduler, quality_threshold=0.7, max_steps=5, ar_steps=3, adaptive_steps=True):
        super().__init__(model, scheduler)
        self.quality_threshold = quality_threshold
        self.max_steps = max_steps
        self.ar_steps = ar_steps
        self.adaptive_steps = adaptive_steps

    def sample_high_res(self, x, train=False, quality_score=None, mask=None, boundary_weight=None):
        """Quality-adaptive sampling with Mask-ARDA and dynamic enhancements"""
        device = x.device
        b, c, h, w = x.shape
        model_output = torch.randn((b, c * 2, h, w), device=device)

        # Mask generation and region partitioning (Mask-ARDA)
        if mask is not None and quality_score is not None:
            high_quality_mask = (quality_score > self.quality_threshold).float().view(b, 1, 1, 1) * (1 - mask)
            low_quality_mask = mask
            boundary_mask = boundary_weight * (1 - high_quality_mask) * (1 - low_quality_mask)
        else:
            high_quality_mask = torch.zeros_like(model_output[:, :c, :, :])
            low_quality_mask = torch.ones_like(model_output[:, :c, :, :])
            boundary_mask = torch.zeros_like(model_output[:, :c, :, :])

        prompt_loss = 0
        scorer = self.model.deqa_scorer if hasattr(self.model, 'deqa_scorer') and self.adaptive_steps and not train else None

        # High-quality region: 1-step diffusion
        if not train:
            self.scheduler.set_timesteps(3)  # 基础扩散步数
            high_quality_output = model_output.clone()
            for timestep in self.scheduler.timesteps:
                res_output, step_loss = self.model(high_quality_output, timestep, quality_score=quality_score)
                high_quality_output = self.scheduler.step(res_output, timestep, high_quality_output).prev_sample
                prompt_loss += step_loss
            
            # 添加 1-2 步自回归精炼
            for _ in range(1):  # 或 2
                res_output, step_loss = self.model(high_quality_output, timestep, quality_score=quality_score)
                step_size = 0.05  # 较小的步长，避免过度调整
                high_quality_output[:, c:, :, :] += step_size * res_output[:, c:, :, :]
                prompt_loss += step_loss
            high_quality_residual = high_quality_output[:, c:, :, :] * high_quality_mask

        else:
            high_quality_residual = torch.zeros_like(model_output[:, c:, :, :])

        # Low-quality region: Multi-step diffusion + Autoregressive refinement
        low_quality_output = model_output.clone()
        if not train:
            self.scheduler.set_timesteps(self.max_steps)
            timesteps = self.scheduler.timesteps
            for t in timesteps:
                timestep = torch.full((b,), t, device=device, dtype=torch.long)
                res_output, step_loss = self.model(
                    low_quality_output, 
                    timestep, 
                    quality_score=quality_score
                )
                low_quality_output = self.scheduler.step(res_output, t, low_quality_output).prev_sample
                prompt_loss += step_loss

                # Adaptive early stopping based on quality score
                if scorer and t == timesteps[len(timesteps)//2] and self.adaptive_steps:
                    temp_output = x + low_quality_output[:, c:, :, :]
                    temp_pils = [self.model.to_pil(img.cpu()) for img in temp_output]
                    with torch.no_grad():
                        mid_quality = torch.tensor(scorer(temp_pils), device=device)
                    if mid_quality.mean() / 5.0 > self.quality_threshold:
                        break

            # Autoregressive refinement
            for _ in range(self.ar_steps):
                res_output, step_loss = self.model(low_quality_output, timestep, quality_score=quality_score)
                step_size = 0.1 + 0.2 * (1 - quality_score.mean() / 5.0)  # 动态步长
                low_quality_output[:, c:, :, :] += step_size * res_output[:, c:, :, :]
                prompt_loss += step_loss
        else:
            self.scheduler.set_timesteps(1)
            timestep = self.scheduler.timesteps[0]
            res_output, step_loss = self.model(
                low_quality_output, 
                torch.full((b,), timestep, device=device, dtype=torch.long), 
                quality_score=quality_score
            )
            low_quality_output = self.scheduler.step(res_output, timestep, low_quality_output).prev_sample
            prompt_loss += step_loss

        low_quality_residual = low_quality_output[:, c:, :, :] * low_quality_mask

        # Boundary region: Dynamic steps based on boundary weight
        boundary_output = model_output.clone()
        if not train and boundary_mask.sum() > 0:
            boundary_steps = max(1, int(self.max_steps * boundary_weight.mean().item()))
            self.scheduler.set_timesteps(boundary_steps)
            timesteps = self.scheduler.timesteps
            for t in timesteps:
                timestep = torch.full((b,), t, device=device, dtype=torch.long)
                res_output, step_loss = self.model(
                    boundary_output, 
                    timestep, 
                    quality_score=quality_score
                )
                boundary_output = self.scheduler.step(res_output, t, boundary_output).prev_sample
                prompt_loss += step_loss
        else:
            boundary_residual = torch.zeros_like(model_output[:, c:, :, :])

        boundary_residual = boundary_output[:, c:, :, :] * boundary_mask

        # Combine results
        residual = high_quality_residual + low_quality_residual + boundary_residual
        return residual, prompt_loss

class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, config):
        super(DenoisingDiffusion, self).__init__()
        self.config = config
        self.loss_psnr = PSNRLoss()
        
        self.model = NAFNet(
            img_channel=config.model.img_channel,
            out_channel=config.model.out_channel,
            width=config.model.width,
            middle_blk_num=config.model.middle_blk_num,
            enc_blk_nums=config.model.enc_blk_nums,
            dec_blk_nums=config.model.dec_blk_nums,
            is_prompt_pool=True,
        )
        
        self.deqa_scorer = Scorer(pretrained="/data/suxin/DM/DeQA-Score-Mix3", device="cuda:0")
        self.masking_module = MaskingQualityModule(
                channels=config.model.width,
                deqa_scorer=self.deqa_scorer, 
                smooth_kernel_size=5,  
                smooth_sigma=1.0 )
                
        self.DiffSampler = QualityAwareDiffusionSampler(
            model=self.model,
            scheduler=pipeline.create_SR3scheduler(self.config.diffusion, 'train'),
            quality_threshold=0.7,
            max_steps=10,
            ar_steps=3
        )
        self.DiffSampler.scheduler.set_timesteps(self.config.sampling_timesteps)
        self.lpips_fn = lpips.LPIPS(net='alex')
        
        self.automatic_optimization = True
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.save_path = self.config.image_folder
        self.max_steps = self.config.Trainer.max_steps
        self.epochs = self.config.Trainer.max_epochs
        
        self.save_hyperparameters()
    
    def to_pil(self, img_tensor):
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def extract_deqa_features(self, images):
        batch_size = images.shape[0]
        pil_images = [self.to_pil(img.cpu()) for img in images]
        with torch.no_grad():
            scores = torch.tensor(self.deqa_scorer(pil_images), device=images.device).view(batch_size, 1)
        return scores
    
    def closest_multiple_of_14(self, n):
        return round(n / 14.0) * 14
    
    def lpips_score_fn(self, x, gt):
        self.lpips_fn.to(self.device)
        x = x.to(self.device)
        gt = gt.to(self.device)
        lp_score = self.lpips_fn(gt * 2 - 1, x * 2 - 1)
        return torch.mean(lp_score).item()
    
    def configure_optimizers(self):
        parameters = [
            {'params': self.model.parameters()},
            {'params': self.masking_module.parameters()},
        ]
        optimizer = get_optimizer(self.config, parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_steps // len(self.train_dataloader())*2, 
            eta_min=self.config.optim.lr * 0.1
        )
        self.optimizer = optimizer
        return [optimizer], [scheduler]
    
    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_psnr',
            filename='QA-Diffusion-epoch{epoch:02d}-PSNR-{val_psnr:.3f}-SSIM-{val_ssim:.4f}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=6,
            mode="max",
            save_last=True
        )
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        ema_callback = EMA(decay=0.995, every_n_steps=1)
        return [checkpoint_callback, lr_monitor_callback, ema_callback]
    
    def training_step(self, batch, batch_idx):
        x, gt, img_id = batch
        b, c, h, w = x.shape
        
        gt_residual = gt - x
        quality_scores = self.extract_deqa_features(x)
        batch = torch.cat([x, gt_residual], dim=1)
        
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (b,), device=self.device).long()
        noise = torch.randn(batch.shape).to(self.device)
        noisy_images = self.DiffSampler.scheduler.add_noise(batch, timesteps=timesteps, noise=noise)
        
        attention_mask, quality_map, boundary_weight = self.masking_module(noisy_images, timesteps)
        residual_train, prompt_loss_train = self.model(noisy_images, timesteps, quality_score=quality_scores)
        pred_residual = residual_train[:, c:, :, :]
        loss_noise = self.loss_psnr(pred_residual + x, gt)
        
        sample_residual, prompt_loss_sample = self.DiffSampler.sample_high_res(
            x, train=True, quality_score=quality_scores, mask=attention_mask, boundary_weight=boundary_weight
        )
        samples = x + sample_residual
        
        # Debug: Check shapes of samples and gt
        # print(f"samples shape: {samples.shape}")
        # print(f"gt shape: {gt.shape}")
        
        # Fix shapes if necessary
        if samples.dim() == 5:
            samples = samples[:, 0, :, :, :]  # Reduce to 4D
        if gt.dim() == 5:
            gt = gt[:, 0, :, :, :]  # Reduce to 4D
        
        psnr_value = batch_PSNR(samples.float(), gt.float(), ycbcr=True)
        ssim_value = batch_SSIM(samples.float(), gt.float(), ycbcr=True)
        loss_samples_psnr = self.loss_psnr(samples, gt)
        loss_samples = loss_samples_psnr
        # 计算区域残差一致性损失
        high_quality_residual = sample_residual * (quality_scores > 0.7).float().view(b, 1, 1, 1) * (1 - attention_mask)
        low_quality_residual = sample_residual * attention_mask
        consistency_loss = F.mse_loss(high_quality_residual.mean(dim=[2, 3]), low_quality_residual.mean(dim=[2, 3]))
        
        loss_prompt_contrast = (prompt_loss_train + prompt_loss_sample) * 0.5
        loss = 2.0 * loss_noise + 2.0 * loss_samples + loss_prompt_contrast + 0.1 * consistency_loss
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    
    def validation_step(self, batch, batch_idx):
        input_x, target, img_id = batch
        input_x = input_x.float()
        target = target.float()
        
        # 确保使用原始 RGB 图像计算 DeQA 质量分数
        quality_scores = self.extract_deqa_features(input_x)
        
        attention_mask, quality_map, boundary_weight = self.masking_module(input_x, None)
        
        samples_residual, _ = self.DiffSampler.sample_high_res(
            input_x, train=False, quality_score=quality_scores, 
            mask=attention_mask, boundary_weight=boundary_weight
        )
        
        if samples_residual.dim() == 5:
            samples_residual = samples_residual[:, 0, :, :, :]
        
        if samples_residual.shape[1] != input_x.shape[1]:
            if samples_residual.shape[1] > input_x.shape[1]:
                samples_residual = samples_residual[:, :input_x.shape[1]]
            else:
                pad_channels = input_x.shape[1] - samples_residual.shape[1]
                samples_residual = torch.cat([
                    samples_residual, 
                    torch.zeros(samples_residual.shape[0], pad_channels, 
                                samples_residual.shape[2], samples_residual.shape[3], 
                                device=samples_residual.device)
                ], dim=1)
        
        if (samples_residual.shape[2] != input_x.shape[2] or 
            samples_residual.shape[3] != input_x.shape[3]):
            samples_residual = F.interpolate(
                samples_residual,
                size=(input_x.shape[2], input_x.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        
        samples = input_x + samples_residual
        
        
        if samples.shape[1] != 3:
            if samples.shape[1] > 3:
                samples_to_save = samples[:, :3]
            else:
                samples_to_save = torch.cat([
                    samples, 
                    torch.zeros(samples.shape[0], 3-samples.shape[1], 
                                samples.shape[2], samples.shape[3], 
                                device=samples.device)
                ], dim=1)
        else:
            samples_to_save = samples
        
        
        max_batch_items = min(5, samples_to_save.shape[0])
        
        if self.config.train_type == True:
            if batch_idx == 0:
                filename = f"sample_{self.current_epoch}.png"
                save_image(samples_to_save[:max_batch_items], os.path.join(self.save_path, filename))
                
                
                if samples_residual.shape[1] != 3:
                    if samples_residual.shape[1] > 3:
                        residual_to_save = samples_residual[:, :3]
                    else:
                        residual_to_save = torch.cat([
                            samples_residual,
                            torch.zeros(samples_residual.shape[0], 3-samples_residual.shape[1],
                                    samples_residual.shape[2], samples_residual.shape[3],
                                    device=samples_residual.device)
                        ], dim=1)
                else:
                    residual_to_save = samples_residual
                
                filename = f"sample_degraded_{self.current_epoch}.png"
                save_colormapped_image(residual_to_save[:max_batch_items], os.path.join(self.save_path, filename))
                
                filename = f"quality_map_{self.current_epoch}.png"
                save_colormapped_image(quality_map[:max_batch_items], os.path.join(self.save_path, filename))
                
                filename = f"target_{self.current_epoch}.png"
                save_image(target[:max_batch_items], os.path.join(self.save_path, filename))
        else:
            filename = f"sample_{img_id[0]}.png"
            save_image(samples_to_save[:1], os.path.join(self.save_path, filename))
        
        psnr = batch_PSNR(samples.float(), target.float(), ycbcr=True)
        ssim = batch_SSIM(samples.float(), target.float(), ycbcr=True)
        lpips_score = self.lpips_score_fn(samples.float(), target.float())
        
        with torch.no_grad():
            input_quality = self.extract_deqa_features(input_x)
            output_quality = self.extract_deqa_features(samples)
            quality_gain = (output_quality - input_quality).mean()
        
        self.log('val_psnr', psnr, sync_dist=True)
        self.log('val_ssim', ssim, sync_dist=True)
        self.log('val_lpips', lpips_score)
        self.log('val_quality_gain', quality_gain)
        
        return {"psnr": psnr, "ssim": ssim, "lpips": lpips_score, "quality_gain": quality_gain}

    
    def train_dataloader(self):
        train_set = AGAN_Dataset(
            self.config.data.data_dir,
            train=True,
            size=self.config.data.image_size,
            crop=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        return train_loader
    
    def val_dataloader(self):
        if self.config.data.dataset == 'Test1':
            val_set = Test1(self.config.data.val_data_dir, train=False, size=256, crop=True)
        elif self.config.data.dataset == 'Raindrop':
            val_set = AGAN_Dataset(self.config.data.val_data_dir, train=False, size=256, crop=True)
        elif self.config.data.dataset == 'Snow100k-S':
            val_set = Snow100kTest(self.config.data.val_data_dir, train=False, size=256, crop=True)
        elif self.config.data.dataset == 'Snow100k-L':
            val_set = Snow100kTest(self.config.data.val_data_dir, train=False, size=256, crop=True)
        
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        return val_loader
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['quality_config'] = {
            'deqa_version': getattr(self.deqa_scorer, 'version', 'unknown'),
            'masking_threshold_min': self.masking_module.mask_ratio_min,
            'masking_threshold_max': self.masking_module.mask_ratio_max,
        }
