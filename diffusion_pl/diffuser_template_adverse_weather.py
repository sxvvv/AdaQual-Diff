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
from models_prompt.modules.prompt import AdaptiveQualityPrompt
import Utils


class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, config):
        super(DenoisingDiffusion, self).__init__()
        self.save_hyperparameters() # Save configuration
        self.config = config

        self.loss_psnr = PSNRLoss()
        self.model = NAFNet(
            img_channel=config.model.img_channel,
            out_channel=config.model.out_channel,
            width=config.model.width,
            middle_blk_num=config.model.middle_blk_num,
            enc_blk_nums=config.model.enc_blk_nums,
            dec_blk_nums=config.model.dec_blk_nums,
            is_prompt_pool=True, # Enable prompt pool
            prompt_embed_dim=config.model.width # Set prompt embedding dimension
        )
        
        self.deqa_scorer = Scorer(pretrained='/data/suxin/DM/DeQA-Score-Mix3', device="cuda:1") 

        # Create directory for saving images
        self.save_path = self.config.image_folder
        os.makedirs(self.save_path, exist_ok=True)

        # Get training parameters from config
        self.max_steps = self.config.Trainer.max_steps
        self.epochs = self.config.Trainer.max_epochs

        self.val_crop = True

        # Setup diffusion sampler
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler=pipeline.create_SR3scheduler(self.config.diffusion, 'train')
        )
        self.DiffSampler.scheduler.set_timesteps(self.config.sampling_timesteps)

        # Setup LPIPS loss
        self.lpips_fn = lpips.LPIPS(net='alex')

        # Module to convert quality score to embedding
        self.condition_fusion = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, config.model.width)
        )

        self.automatic_optimization = True

        # Basic transform for images
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def lpips_score_fn(self, x, gt):
        # Move LPIPS model to current device
        self.lpips_fn.to(self.device)
        x = x.to(self.device)
        gt = gt.to(self.device)
        # Calculate LPIPS (input range [-1, 1])
        lp_score = self.lpips_fn(gt * 2 - 1, x * 2 - 1)
        return torch.mean(lp_score).item()

    def tensor_to_pil(self, tensor):
        """Convert tensor batch to list of PIL images"""
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.detach().cpu()
        images = []
        for img_tensor in tensor:
            img = torchvision.transforms.functional.to_pil_image(img_tensor)
            images.append(img)
        return images

    def configure_optimizers(self):
        # Combine parameters from model and condition fusion
        parameters = [
            {'params': self.model.parameters()},
            {'params': self.condition_fusion.parameters()}
        ]
        # Get optimizer from config
        optimizer = Utils.optimize.get_optimizer(self.config, parameters)

        # Setup learning rate scheduler
        if hasattr(self, 'trainer') and self.trainer.datamodule:
            num_batches = len(self.trainer.datamodule.train_dataloader())
            t_max_steps = (self.max_steps // num_batches) if num_batches > 0 else self.max_steps
        else:
            t_max_steps = self.max_steps
            print("Warning: Using max_steps for scheduler T_max calculation.")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_steps,
            eta_min=self.config.optim.lr * 1e-2 # Minimum learning rate
        )
        self.optimizer = optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        # Setup model checkpoint saving
        checkpoint_callback = ModelCheckpoint(
            monitor='psnr',
            filename='epoch{epoch:02d}-PSNR{psnr:.3f}-SSIM{ssim:.4f}',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=5,
            mode="max",
            save_last=True
        )
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        ema_callback = EMA(decay=0.995, every_n_steps=1)
        return [checkpoint_callback, lr_monitor_callback, ema_callback]

    def training_step(self, batch, batch_idx):
        # Get inputs from batch
        x, gt, img_id = batch
        if torch.isnan(x).any() or torch.isinf(x).any() or torch.isnan(gt).any() or torch.isinf(gt).any():
            print(f"NaN/Inf detected in input batch {batch_idx} at step {self.global_step}")
            return None # Skip invalid batch

        b, c, h, w = x.shape
        gt_residual = gt - x # Calculate residual
        if torch.isnan(gt_residual).any() or torch.isinf(gt_residual).any():
             print(f"NaN/Inf detected in gt_residual at step {self.global_step}")
             return None

        # Prepare input for diffusion
        batch_combined = torch.cat([x, gt_residual], dim=1)
        if torch.isnan(batch_combined).any() or torch.isinf(batch_combined).any():
             print(f"NaN/Inf detected in batch_combined at step {self.global_step}")
             return None

        # Calculate image quality score
        x_pil = self.tensor_to_pil(x)
        with torch.no_grad():
             deqa_score = self.deqa_scorer(x_pil).to(self.device)
             if torch.isnan(deqa_score).any() or torch.isinf(deqa_score).any():
                  print(f"NaN/Inf detected in deqa_score at step {self.global_step}")
                  deqa_score = torch.ones_like(deqa_score) * 2.5 # Default mid-range score

        # Convert quality score to embedding
        deqa_score_for_fusion = deqa_score.clone()
        if self.training:
             deqa_score_for_fusion.requires_grad_(True)
        quality_embedding = self.condition_fusion(deqa_score_for_fusion.view(-1, 1))
        if torch.isnan(quality_embedding).any() or torch.isinf(quality_embedding).any():
             print(f"NaN/Inf detected in quality_embedding at step {self.global_step}")
             return None

        # Apply diffusion process
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (b,), device=self.device).long()
        noise = torch.randn(batch_combined.shape, device=self.device)
        noisy_images = self.DiffSampler.scheduler.add_noise(batch_combined, timesteps=timesteps, noise=noise)
        if torch.isnan(noisy_images).any() or torch.isinf(noisy_images).any():
             print(f"NaN/Inf detected in noisy_images at step {self.global_step}")
             return None

        # Model prediction (denoising)
        output_dict_train = self.model(
            inp=noisy_images,
            time=timesteps,
            quality_condition=quality_embedding,
            raw_quality_scores=deqa_score
        )
        residual_train = output_dict_train['predicted_residual']
        prompt_loss_train = output_dict_train.get('prompt_loss', torch.tensor(0.0, device=self.device))
        if torch.isnan(residual_train).any() or torch.isinf(residual_train).any():
             print(f"NaN/Inf detected in model output residual_train at step {self.global_step}")
             return None
        if torch.isnan(prompt_loss_train).any() or torch.isinf(prompt_loss_train).any():
             print(f"NaN/Inf detected in model output prompt_loss_train at step {self.global_step}")
             prompt_loss_train = torch.tensor(0.0, device=self.device)

        # Reconstruct image and calculate loss
        reconstructed_gt = residual_train + x
        if torch.isnan(reconstructed_gt).any() or torch.isinf(reconstructed_gt).any():
            print(f"NaN/Inf detected in reconstructed_gt at step {self.global_step}")
            return None

        # Clamp values for stability
        reconstructed_gt_clamped = torch.clamp(reconstructed_gt, 1e-8, 1.0)
        gt_clamped = torch.clamp(gt, 1e-8, 1.0)
        loss_noise = self.loss_psnr(reconstructed_gt_clamped, gt_clamped)
        if torch.isnan(loss_noise).any() or torch.isinf(loss_noise).any():
            print(f"NaN/Inf detected in loss_noise at step {self.global_step}. Clamped inputs: recon={reconstructed_gt_clamped.min().item(), reconstructed_gt_clamped.max().item()}, gt={gt_clamped.min().item(), gt_clamped.max().item()}")
            return None

        # Optional online sampling during training (currently disabled)
        run_online_sampling = False
        if run_online_sampling:
            with torch.no_grad():
                 sample_output_dict = self.DiffSampler.sample_high_res(
                     low_res=x,
                     train=True,
                     quality_condition=quality_embedding.detach(),
                     raw_quality_scores=deqa_score
                 )
                 sample_residual = sample_output_dict['sample']
                 prompt_loss_sample = sample_output_dict.get('prompt_loss', torch.tensor(0.0, device=self.device))
                 if torch.isnan(sample_residual).any() or torch.isinf(sample_residual).any():
                     print(f"NaN/Inf detected in sample_residual at step {self.global_step}")
                     sample_residual = torch.zeros_like(sample_residual)
                     prompt_loss_sample = torch.tensor(0.0, device=self.device)
                 if torch.isnan(prompt_loss_sample).any() or torch.isinf(prompt_loss_sample).any():
                      print(f"NaN/Inf detected in prompt_loss_sample at step {self.global_step}")
                      prompt_loss_sample = torch.tensor(0.0, device=self.device)

            samples = x + sample_residual
            if torch.isnan(samples).any() or torch.isinf(samples).any():
                 print(f"NaN/Inf detected in samples (from sampler) at step {self.global_step}")
                 loss_samples = torch.tensor(0.0, device=self.device)
                 psnr_train_online = torch.tensor(0.0, device=self.device)
            else:
                # Calculate online PSNR
                psnr_train_online = batch_PSNR(samples.detach().float(), gt.float(), ycbcr=True)
                if torch.isnan(psnr_train_online).any() or torch.isinf(psnr_train_online).any():
                     print(f"NaN/Inf detected in psnr_train_online at step {self.global_step}")
                     psnr_train_online = torch.tensor(0.0, device=self.device)

                # Calculate sample loss
                samples_clamped = torch.clamp(samples, 1e-8, 1.0)
                loss_samples = self.loss_psnr(samples_clamped, gt_clamped)
                if torch.isnan(loss_samples).any() or torch.isinf(loss_samples).any():
                     print(f"NaN/Inf detected in loss_samples_psnr at step {self.global_step}. Clamped inputs: samples={samples_clamped.min().item(), samples_clamped.max().item()}, gt={gt_clamped.min().item(), gt_clamped.max().item()}")
                     loss_samples = torch.tensor(0.0, device=self.device)
        else:
            # Default values when sampling is disabled
            loss_samples = torch.tensor(0.0, device=self.device)
            prompt_loss_sample = torch.tensor(0.0, device=self.device)
            psnr_train_online = torch.tensor(0.0, device=self.device)

        # Calculate prompt loss with quality weighting
        quality_weight = deqa_score.view(-1) / 5.0
        if torch.isnan(quality_weight).any() or torch.isinf(quality_weight).any() or quality_weight.min() < 0:
             print(f"Problem detected in quality_weight at step {self.global_step}: min={quality_weight.min()}, max={quality_weight.max()}")
             quality_weight = torch.ones_like(quality_weight) * 0.5

        loss_prompt_contrast = (prompt_loss_train + prompt_loss_sample) * 0.5
        if torch.isnan(loss_prompt_contrast).any() or torch.isinf(loss_prompt_contrast).any():
             print(f"NaN/Inf detected in loss_prompt_contrast at step {self.global_step}")
             loss_prompt_contrast = torch.tensor(0.0, device=self.device)

        # Combine all losses
        loss = loss_noise + loss_samples + loss_prompt_contrast * quality_weight.mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
             print(f"Final loss is NaN/Inf at step {self.global_step}. Loss components: noise={loss_noise.item()}, samples={loss_samples.item()}, prompt={loss_prompt_contrast.item()}, weight_mean={quality_weight.mean().item()}")
             return None

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_psnr_online", psnr_train_online, prog_bar=True, sync_dist=True)
        self.log("loss_noise", loss_noise, prog_bar=False, sync_dist=True)
        self.log("loss_samples", loss_samples, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_train", prompt_loss_train, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_sample", prompt_loss_sample, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_combined", loss_prompt_contrast, prog_bar=False, sync_dist=True)
        self.log("mean_quality_weight", quality_weight.mean(), prog_bar=False, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_x, target, img_id = batch
        b, c, h, w = input_x.shape

        # Calculate quality score and embedding
        input_x_pil = self.tensor_to_pil(input_x)
        with torch.no_grad():
            deqa_score = self.deqa_scorer(input_x_pil).to(self.device)
            quality_embedding_cond = self.condition_fusion(deqa_score.view(-1, 1))

        # Run sampling inference
        with torch.no_grad():
            sample_output = self.DiffSampler.sample_high_res(
                low_res=input_x,
                train=False,
                quality_condition=quality_embedding_cond,
                raw_quality_scores=deqa_score
            )
            samples_residual = sample_output['sample']
            prompt_loss_val = sample_output.get('prompt_loss', torch.tensor(0.0, device=self.device))

        # Create final output image
        samples = torch.clamp(samples_residual + input_x, 0, 1)

        # Save validation images
        if batch_idx == 0 and self.trainer.is_global_zero:
            save_dir = os.path.join(self.save_path, f"epoch_{self.current_epoch}")
            os.makedirs(save_dir, exist_ok=True)

            save_image(samples[:8], os.path.join(save_dir, f"samples_batch{batch_idx}.png"))
            save_image(input_x[:8], os.path.join(save_dir, f"inputs_batch{batch_idx}.png"))
            save_image(target[:8], os.path.join(save_dir, f"targets_batch{batch_idx}.png"))
            save_colormapped_image(samples_residual[:8], os.path.join(save_dir, f"residuals_batch{batch_idx}.png"))

        # Calculate quality metrics
        psnr = batch_PSNR(samples.float(), target.float(), ycbcr=True)
        ssim = batch_SSIM(samples.float(), target.float(), ycbcr=True)
        lpips_score = self.lpips_score_fn(samples.float(), target.float())

        # Calculate quality score of output
        samples_pil = self.tensor_to_pil(samples)
        with torch.no_grad():
             output_deqa_scores = self.deqa_scorer(samples_pil).to(self.device)
        deqa_score_val = torch.mean(output_deqa_scores).item()

        # Log validation metrics
        self.log('psnr', psnr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('ssim', ssim, on_step=False, on_epoch=True, sync_dist=True)
        self.log('lpips', lpips_score, on_step=False, on_epoch=True, sync_dist=True)
        self.log('deqa_score_output', deqa_score_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('prompt_loss_val', prompt_loss_val, on_step=False, on_epoch=True, sync_dist=True)

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips_score, "deqa_score": deqa_score_val}

    def train_dataloader(self):
        # Setup training data loader
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
            pin_memory=True,
            persistent_workers=True if self.config.data.num_workers > 0 else False
        )
        return train_loader

    def val_dataloader(self):
        # Select appropriate validation dataset
        dataset_name = self.config.data.dataset
        val_dir = self.config.data.val_data_dir
        image_size = self.config.data.image_size

        if dataset_name == 'Test1':
            val_set = Test1(val_dir, train=False, size=image_size, crop=self.val_crop)
        elif dataset_name == 'Raindrop':
            val_set = AGAN_Dataset(val_dir, train=False, size=image_size, crop=self.val_crop)
        elif dataset_name in ['Snow100k-S', 'Snow100k-L']:
            val_set = Snow100kTest(val_dir, train=False, size=image_size, crop=self.val_crop)
        elif dataset_name == 'Allweather':
             val_set = Allweather(val_dir, train=False, size=image_size, crop=self.val_crop)
        else:
            raise ValueError(f"Unsupported validation dataset: {dataset_name}")

        val_loader = DataLoader(
            val_set,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.data.num_workers > 0 else False
        )
        return val_loader
