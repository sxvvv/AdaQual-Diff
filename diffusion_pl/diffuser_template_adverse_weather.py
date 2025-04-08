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
from Utils.Allweather import AllWeather, Snow100kTest, Test1
from Utils.AGAN_data import AGAN_Dataset
from Utils.imgqual_utils import batch_PSNR, batch_SSIM
from Utils.save_image import save_colormapped_image
import diffusion_pl.pipeline as pipeline
from einops import rearrange, reduce
import math
import numpy as np
import torch.optim as optim
from diffusion_pl.ema import EMA
from models_prompt.modules.prompt import AdaptiveQualityPrompt
import Utils
import gc
from collections import OrderedDict
import time

class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, config):
        super(DenoisingDiffusion, self).__init__()
        self.save_hyperparameters()
        self.config = config

        # GPU-optimized scorer management
        self.deqa_scorer = None
        self.scorer_initialized = False
        
        # Efficient scoring cache system
        self.score_cache = OrderedDict()
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        self.max_cache_size = 10000
        
        # Performance optimization parameters
        self.score_batch_size = 16
        self.scoring_call_count = 0
        self.memory_cleanup_interval = 100
        
        # GPU memory optimization
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Performance statistics
        self.scoring_time_total = 0.0
        self.scoring_calls = 0
        
        # Model initialization
        self.loss_psnr = PSNRLoss()
        self.model = NAFNet(
            img_channel=config.model.img_channel,
            out_channel=config.model.out_channel,
            width=config.model.width,
            middle_blk_num=config.model.middle_blk_num,
            enc_blk_nums=config.model.enc_blk_nums,
            dec_blk_nums=config.model.dec_blk_nums,
            is_prompt_pool=True, 
            prompt_embed_dim=config.model.width
        )
        
        self.deqa_pretrained_path = '/data/suxin/DeQA-Score-Mix3'

        # Create image save directory
        self.save_path = self.config.image_folder
        os.makedirs(self.save_path, exist_ok=True)

        # Get training parameters from config
        self.max_steps = self.config.Trainer.max_steps
        self.epochs = self.config.Trainer.max_epochs

        self.val_crop = True

        # Set up diffusion sampler
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler=pipeline.create_SR3scheduler(self.config.diffusion, 'train')
        )
        self.DiffSampler.scheduler.set_timesteps(self.config.sampling_timesteps)

        # Set up LPIPS loss
        self.lpips_fn = lpips.LPIPS(net='alex')

        # Quality score to embedding module
        self.condition_fusion = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, config.model.width)
        )

        self.automatic_optimization = True

        # Basic image transforms
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_persistent_scorer(self):
        """Get or create a persistent GPU scorer instance"""
        if self.scorer_initialized and self.deqa_scorer is not None:
            try:
                if hasattr(self.deqa_scorer, 'model'):
                    return self.deqa_scorer
            except:
                self.scorer_initialized = False
                self.deqa_scorer = None
        
        if hasattr(self, 'deqa_scorer') and self.deqa_scorer is not None:
            try:
                del self.deqa_scorer
                torch.cuda.empty_cache()
            except:
                pass
        
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            scorer = Scorer(pretrained=self.deqa_pretrained_path, device=str(self.device))
            scorer.eval()
            
            dummy_img = Image.new('RGB', (224, 224), color='white')
            with torch.no_grad():
                scorer([dummy_img])
            
            self.deqa_scorer = scorer
            self.scorer_initialized = True
            return scorer
                
        except Exception:
            self.scorer_initialized = False
            return None

    def safe_deqa_score(self, images_pil, img_ids=None):
        """Efficient scoring function with GPU optimization and caching"""
        start_time = time.time()
        self.scoring_call_count += 1
        self.scoring_calls += 1
        
        result_scores = torch.ones(len(images_pil), device=self.device) * 3.0
        
        if img_ids is not None:
            cache_keys = [str(id.item()) if torch.is_tensor(id) else str(id) for id in img_ids]
        else:
            cache_keys = [f"img_{i}" for i in range(len(images_pil))]
        
        images_to_score = []
        indices_to_score = []
        
        for i, key in enumerate(cache_keys):
            if key in self.score_cache:
                result_scores[i] = self.score_cache[key]
                self.score_cache_hits += 1
                score = self.score_cache.pop(key)
                self.score_cache[key] = score
            else:
                images_to_score.append(images_pil[i])
                indices_to_score.append(i)
                self.score_cache_misses += 1
        
        if not images_to_score:
            end_time = time.time()
            self.scoring_time_total += (end_time - start_time)
            return result_scores
        
        try:
            scorer = self.get_persistent_scorer()
            if scorer is None:
                return result_scores
            
            for i in range(0, len(images_to_score), self.score_batch_size):
                batch_images = images_to_score[i:i+self.score_batch_size]
                batch_indices = indices_to_score[i:i+self.score_batch_size]
                
                with torch.no_grad():
                    batch_scores = scorer(batch_images)
                    
                    if torch.isnan(batch_scores).any() or torch.isinf(batch_scores).any():
                        batch_scores = torch.ones(len(batch_images), device=batch_scores.device) * 3.0
                    
                    batch_scores = torch.clamp(batch_scores, 1.0, 5.0)
                    
                    for j, (idx, key) in enumerate(zip(batch_indices, [cache_keys[idx] for idx in batch_indices])):
                        if j < len(batch_scores):
                            score_value = batch_scores[j].item()
                            result_scores[idx] = score_value
                            self.score_cache[key] = score_value
                            
                            if len(self.score_cache) > self.max_cache_size:
                                self.score_cache.popitem(last=False)
            
            if self.scoring_call_count % self.memory_cleanup_interval == 0:
                torch.cuda.empty_cache()
            
        except Exception:
            pass
        
        end_time = time.time()
        self.scoring_time_total += (end_time - start_time)
        
        return result_scores

    def lpips_score_fn(self, x, gt):
        """Calculate LPIPS score between generated and ground truth images"""
        self.lpips_fn.to(self.device)
        x = x.to(self.device)
        gt = gt.to(self.device)
        lp_score = self.lpips_fn(
            gt * 2 - 1, x * 2 - 1
        )
        return torch.mean(lp_score).item()

    def tensor_to_pil(self, tensor):
        """Convert tensor batch to PIL image list"""
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.detach().cpu()
        images = []
        for img_tensor in tensor:
            img = torchvision.transforms.functional.to_pil_image(img_tensor)
            images.append(img)
        return images

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        parameters = [
            {'params': self.model.parameters()},
            {'params': self.condition_fusion.parameters()}
        ]
        optimizer = Utils.optimize.get_optimizer(self.config, parameters)

        if hasattr(self, 'trainer') and self.trainer.datamodule:
            num_batches = len(self.trainer.datamodule.train_dataloader())
            t_max_steps = (self.max_steps // num_batches) if num_batches > 0 else self.max_steps
        else:
            t_max_steps = self.max_steps

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_steps,
            eta_min=self.config.optim.lr * 1e-2
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
        """Configure training callbacks"""
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
        x, gt, img_id = batch
        if torch.isnan(x).any() or torch.isinf(x).any() or torch.isnan(gt).any() or torch.isinf(gt).any():
            return None

        b, c, h, w = x.shape
        gt_residual = gt - x
        if torch.isnan(gt_residual).any() or torch.isinf(gt_residual).any():
            return None

        batch_combined = torch.cat([x, gt_residual], dim=1)
        if torch.isnan(batch_combined).any() or torch.isinf(batch_combined).any():
            return None

        x_pil = self.tensor_to_pil(x)
        with torch.no_grad():
            try:
                deqa_score = self.safe_deqa_score(x_pil, img_id)
                if torch.isnan(deqa_score).any() or torch.isinf(deqa_score).any():
                    deqa_score = torch.ones(b, device=self.device) * 3.0
            except Exception:
                deqa_score = torch.ones(b, device=self.device) * 3.0
        
        deqa_score_for_fusion = deqa_score.clone()
        if self.training:
            deqa_score_for_fusion.requires_grad_(True)
        quality_embedding = self.condition_fusion(deqa_score_for_fusion.view(-1, 1))
        if torch.isnan(quality_embedding).any() or torch.isinf(quality_embedding).any():
            return None

        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (b,), device=self.device).long()
        noise = torch.randn(batch_combined.shape, device=self.device)
        noisy_images = self.DiffSampler.scheduler.add_noise(batch_combined, timesteps=timesteps, noise=noise)
        if torch.isnan(noisy_images).any() or torch.isinf(noisy_images).any():
            return None

        output_dict_train = self.model(
            inp=noisy_images,
            time=timesteps,
            quality_condition=quality_embedding,
            raw_quality_scores=deqa_score
        )
        residual_train = output_dict_train['predicted_residual']
        prompt_loss_train = output_dict_train.get('prompt_loss', torch.tensor(0.0, device=self.device))
        if torch.isnan(residual_train).any() or torch.isinf(residual_train).any():
            return None
        if torch.isnan(prompt_loss_train).any() or torch.isinf(prompt_loss_train).any():
            prompt_loss_train = torch.tensor(0.0, device=self.device)

        reconstructed_gt = residual_train + x
        if torch.isnan(reconstructed_gt).any() or torch.isinf(reconstructed_gt).any():
            return None

        reconstructed_gt_clamped = torch.clamp(reconstructed_gt, 1e-8, 1.0)
        gt_clamped = torch.clamp(gt, 1e-8, 1.0)
        loss_noise = self.loss_psnr(reconstructed_gt_clamped, gt_clamped)
        if torch.isnan(loss_noise).any() or torch.isinf(loss_noise).any():
            return None

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
                    sample_residual = torch.zeros_like(sample_residual)
                    prompt_loss_sample = torch.tensor(0.0, device=self.device)
                if torch.isnan(prompt_loss_sample).any() or torch.isinf(prompt_loss_sample).any():
                    prompt_loss_sample = torch.tensor(0.0, device=self.device)

            samples = x + sample_residual
            if torch.isnan(samples).any() or torch.isinf(samples).any():
                loss_samples = torch.tensor(0.0, device=self.device)
                psnr_train_online = torch.tensor(0.0, device=self.device)
            else:
                psnr_train_online = batch_PSNR(samples.detach().float(), gt.float(), ycbcr=True)
                if torch.isnan(psnr_train_online).any() or torch.isinf(psnr_train_online).any():
                    psnr_train_online = torch.tensor(0.0, device=self.device)

                samples_clamped = torch.clamp(samples, 1e-8, 1.0)
                loss_samples = self.loss_psnr(samples_clamped, gt_clamped)
                if torch.isnan(loss_samples).any() or torch.isinf(loss_samples).any():
                    loss_samples = torch.tensor(0.0, device=self.device)
        else:
            loss_samples = torch.tensor(0.0, device=self.device)
            prompt_loss_sample = torch.tensor(0.0, device=self.device)
            psnr_train_online = torch.tensor(0.0, device=self.device)

        quality_weight = deqa_score.view(-1) / 5.0
        if torch.isnan(quality_weight).any() or torch.isinf(quality_weight).any() or quality_weight.min() < 0:
            quality_weight = torch.ones_like(quality_weight) * 0.5

        loss_prompt_contrast = (prompt_loss_train + prompt_loss_sample) * 0.5
        if torch.isnan(loss_prompt_contrast).any() or torch.isinf(loss_prompt_contrast).any():
            loss_prompt_contrast = torch.tensor(0.0, device=self.device)

        loss = loss_noise + loss_samples + loss_prompt_contrast * quality_weight.mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return None

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_psnr_online", psnr_train_online, prog_bar=True, sync_dist=True)
        self.log("loss_noise", loss_noise, prog_bar=False, sync_dist=True)
        self.log("loss_samples", loss_samples, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_train", prompt_loss_train, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_sample", prompt_loss_sample, prog_bar=False, sync_dist=True)
        self.log("prompt_loss_combined", loss_prompt_contrast, prog_bar=False, sync_dist=True)
        self.log("mean_quality_weight", quality_weight.mean(), prog_bar=False, sync_dist=True)
        self.log("deqa_mean_score", deqa_score.mean(), prog_bar=False, sync_dist=True)
        
        hit_rate = 0
        if (self.score_cache_hits + self.score_cache_misses) > 0:
            hit_rate = self.score_cache_hits / (self.score_cache_hits + self.score_cache_misses) * 100
        self.log("score_cache_hit_rate", hit_rate, prog_bar=False, sync_dist=True)
        
        if self.scoring_calls > 0:
            avg_scoring_time = self.scoring_time_total / self.scoring_calls
            self.log("avg_scoring_time", avg_scoring_time, prog_bar=False, sync_dist=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_x, target, img_id = batch
        b, c, h, w = input_x.shape

        input_x_pil = self.tensor_to_pil(input_x)
        with torch.no_grad():
            try:
                deqa_score = self.safe_deqa_score(input_x_pil, img_id)
                if torch.isnan(deqa_score).any() or torch.isinf(deqa_score).any():
                    deqa_score = torch.ones(b, device=self.device) * 3.0
            except Exception:
                deqa_score = torch.ones(b, device=self.device) * 3.0
                
            quality_embedding_cond = self.condition_fusion(deqa_score.view(-1, 1))

        with torch.no_grad():
            sample_output = self.DiffSampler.sample_high_res(
                low_res=input_x,
                train=False,
                quality_condition=quality_embedding_cond,
                raw_quality_scores=deqa_score
            )
            samples_residual = sample_output['sample']
            prompt_loss_val = sample_output.get('prompt_loss', torch.tensor(0.0, device=self.device))

        samples = torch.clamp(samples_residual + input_x, 0, 1)

        if batch_idx == 0 and self.trainer.is_global_zero:
            save_dir = os.path.join(self.save_path, f"epoch_{self.current_epoch}")
            os.makedirs(save_dir, exist_ok=True)

            save_image(samples[:8], os.path.join(save_dir, f"samples_batch{batch_idx}.png"))
            save_image(input_x[:8], os.path.join(save_dir, f"inputs_batch{batch_idx}.png"))
            save_image(target[:8], os.path.join(save_dir, f"targets_batch{batch_idx}.png"))
            save_colormapped_image(samples_residual[:8], os.path.join(save_dir, f"residuals_batch{batch_idx}.png"))

        psnr = batch_PSNR(samples.float(), target.float(), ycbcr=True)
        ssim = batch_SSIM(samples.float(), target.float(), ycbcr=True)
        lpips_score = self.lpips_score_fn(samples.float(), target.float())

        samples_pil = self.tensor_to_pil(samples)
        with torch.no_grad():
            try:
                output_deqa_scores = self.safe_deqa_score(samples_pil)
                deqa_score_val = torch.mean(output_deqa_scores).item()
            except Exception:
                deqa_score_val = 3.0

        self.log('psnr', psnr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('ssim', ssim, on_step=False, on_epoch=True, sync_dist=True)
        self.log('lpips', lpips_score, on_step=False, on_epoch=True, sync_dist=True)
        self.log('deqa_score_output', deqa_score_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('prompt_loss_val', prompt_loss_val, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.scoring_calls > 0:
            avg_scoring_time = self.scoring_time_total / self.scoring_calls
            self.log('avg_scoring_time', avg_scoring_time, on_step=False, on_epoch=True, sync_dist=True)
        
        if (self.score_cache_hits + self.score_cache_misses) > 0:
            hit_rate = self.score_cache_hits / (self.score_cache_hits + self.score_cache_misses) * 100
            self.log('cache_hit_rate', hit_rate, on_step=False, on_epoch=True, sync_dist=True)

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips_score, "deqa_score": deqa_score_val}

    def train_dataloader(self):
        """Configure training data loader"""
        train_set = AllWeather(
            self.config.data.data_dir,
            train=True,
            size=self.config.data.image_size,
            crop=True
        )
        train_loader = DataLoader(train_set, 
                                  batch_size=self.config.training.batch_size, 
                                  shuffle=True, 
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True)
        return train_loader

    def val_dataloader(self):
        """Configure validation data loader"""
        dataset_name = self.config.data.dataset
        val_dir = self.config.data.val_data_dir
        image_size = self.config.data.image_size

        if dataset_name == 'Test1':
            val_set = Test1(val_dir, train=False, size=image_size, crop=self.val_crop)
        elif dataset_name == 'Raindrop':
            val_set = AGAN_Dataset(val_dir, train=False, size=image_size, crop=self.val_crop)
        elif dataset_name in ['Snow100k-S', 'Snow100k-L']:
            val_set = Snow100kTest(val_dir, train=False, size=image_size, crop=self.val_crop)
        else:
            val_set = AllWeather(val_dir, train=False, size=image_size, crop=self.val_crop)

        val_loader = DataLoader(val_set,
                               batch_size=self.config.training.batch_size,
                               shuffle=False,
                               num_workers=self.config.data.num_workers,
                               pin_memory=True)
        return val_loader
