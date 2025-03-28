import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from .module_util import LayerNorm
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

class QualityFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=768, quality_features_dim=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_proj = nn.Linear(quality_features_dim, embed_dim)
        self.local_proj = nn.Sequential(
            nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, embed_dim)
        )
        self.quality_level_embeddings = nn.Parameter(torch.randn(5, embed_dim) * 0.02)
    
    def forward(self, quality_score, quality_variance=None):
        batch_size = quality_score.shape[0]
        device = quality_score.device  # Use the device of the input tensor
        normalized_scores = (quality_score - 1) / 4  # Map [1,5] to [0,1]
        level_weights = torch.zeros(batch_size, 5, device=device)
        for b in range(batch_size):
            score = quality_score[b].item()
            lower_idx = max(0, min(3, int(score - 1)))
            upper_idx = lower_idx + 1
            alpha = score - (lower_idx + 1)
            level_weights[b, lower_idx] = 1 - alpha
            level_weights[b, upper_idx] = alpha
        # Ensure quality_level_embeddings is on the same device
        quality_embeddings = torch.matmul(level_weights, self.quality_level_embeddings.to(device))
        # local_proj will automatically use its parameters' device, assuming the model is on the correct device
        quality_features = self.local_proj(quality_score.to(self.local_proj[0].weight.device, dtype=self.local_proj[0].weight.dtype))
        return quality_embeddings + quality_features.to(quality_embeddings.device)


class Attention_cross(nn.Module):
    def __init__(self, dim, text_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(text_dim, 2 * dim // self.num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, guidance):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(guidance).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 2, 1).reshape(B, C, H, W)

class QualityAwarePrompt(nn.Module):
    """Quality-Driven Dynamic Prompt Distillation"""
    def __init__(self, length_min=5, length_max=64, embed_dim=512, pool_size=20, top_k=5, beta=0.5):
        super().__init__()
        self.length_min = length_min
        self.length_max = length_max
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.top_k = top_k
        self.beta = beta  # Quality influence factor
        
        self.prompt_keys = nn.Parameter(torch.randn(pool_size, embed_dim))
        self.prompt_embeddings = nn.Parameter(torch.randn(pool_size, length_max, embed_dim))
        self.quality_proj = nn.Sequential(
            nn.Linear(1, embed_dim//2), nn.LayerNorm(embed_dim//2), nn.ReLU(), nn.Linear(embed_dim//2, embed_dim)
        )
        nn.init.xavier_uniform_(self.prompt_keys)
        nn.init.xavier_uniform_(self.prompt_embeddings)
    
    def compute_dynamic_length(self, quality_score):
        mu = quality_score.mean().item()  # [1,5]
        length = self.length_min + (self.length_max - self.length_min) * (1 - mu / 5)
        return max(self.length_min, min(self.length_max, int(length)))
    
    def update_frequency_table(self, selected_idx):
        # Increment the frequency table based on selected indices
        for idx in selected_idx:
            self.prompt_frequency_table[idx] += 1

    def penalize_frequent_prompts(self, similarity_scores):
        # Penalize prompts based on frequency by subtracting the normalized frequency
        # from the similarity scores
        self.prompt_frequency_table = self.prompt_frequency_table.to(similarity_scores.device)
        penalties = self.prompt_frequency_table / self.prompt_frequency_table.max()
        adjusted_scores = similarity_scores - penalties.unsqueeze(0)  # Broadcasting the penalties
        return adjusted_scores

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, quality_score=None):
        batch_size = x_embed.shape[0]
        query_features = self.quality_proj(quality_score) if quality_score is not None else x_embed.mean(1)
        
        similarities = F.cosine_similarity(
            query_features.unsqueeze(1).expand(-1, self.pool_size, -1),
            self.prompt_keys.unsqueeze(0).expand(batch_size, -1, -1),
            dim=2
        )
        weights = torch.softmax(similarities * (1 + self.beta * quality_score.mean()), dim=1)
        
        prompted_embeddings = []
        max_length = self.length_max  # Ensure all tensors have the same length
        for b in range(batch_size):
            length = self.compute_dynamic_length(quality_score[b:b+1])
            topk_indices = torch.topk(weights[b], self.top_k)[1]
            selected_prompts = self.prompt_embeddings[topk_indices, :length]
            padded_prompts = F.pad(selected_prompts, (0, 0, 0, max_length - length))  # Pad to max_length
            weighted_prompt = (padded_prompts * weights[b, topk_indices].view(-1, 1, 1)).sum(0)
            prompted_embeddings.append(weighted_prompt)
        
        return {
            'prompted_embedding_e': torch.stack(prompted_embeddings, dim=0),
            'prompt_loss': torch.tensor(0.0, device=x_embed.device)
        }

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)) if time_emb_dim else None
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
    
    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        return rearrange(time_emb, 'b c -> b c 1 1').chunk(4, dim=1)
    
    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)
        x = self.norm1(inp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return x + y * self.gamma, time

class QualityEnhancer(nn.Module):
    """Quality-aware feature enhancement module"""
    def __init__(self, channels):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True)
        )
        self.weight_proj = nn.Linear(768, channels)
    
    def forward(self, feature, quality_embedding):
        weight = torch.sigmoid(self.weight_proj(quality_embedding)).view(-1, feature.shape[1], 1, 1)
        enhanced = self.enhance(feature)
        return feature * (1 + weight * enhanced)

class NAFNet(nn.Module):
    """NAFNet with Quality-Aware Multi-Scale Feature Enhancement"""
    def __init__(self, img_channel=6, out_channel=3, text_dim=512, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 18], dec_blk_nums=[1, 1, 1, 1], is_prompt_pool=True):
        super().__init__()
        self.is_prompt_pool = is_prompt_pool
        time_dim = width * 4
        self.time_mlp = nn.Sequential(
            Timesteps(width, flip_sin_to_cos=True, downscale_freq_shift=0.),
            nn.Linear(width, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_channel, 3, padding=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enhancers = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            self.enhancers.append(QualityEnhancer(chan))
            chan *= 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)])
        
        if self.is_prompt_pool:
            self.prompt_pool = QualityAwarePrompt(length_min=5, length_max=64, embed_dim=chan, pool_size=20, top_k=5)
            self.atten_list = nn.ModuleList([Attention_cross(chan, chan) for _ in range(2)])
        
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)]))
        
        self.padder_size = 2 ** len(self.encoders)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    
    def forward(self, inp, time=1., depth_feature=None, quality_score=None):
        timesteps = torch.tensor([time], dtype=torch.long, device=inp.device) if not torch.is_tensor(time) else time
        timesteps = timesteps * torch.ones(inp.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        time_mlp_device = next(self.time_mlp.parameters()).device
        t = self.time_mlp(timesteps.to(time_mlp_device))
        
        x = self.check_image_size(inp)
        x = self.intro(x)
        
        encs = [x]
        quality_embedding = QualityFeatureExtractor(embed_dim=768)(quality_score) if quality_score is not None else None
        
        for idx, (encoder, down, enhancer) in enumerate(zip(self.encoders, self.downs, self.enhancers)):
            x, _ = encoder([x, t])
            if quality_embedding is not None:
                x = enhancer(x, quality_embedding)
            encs.append(x)
            x = down(x)
        
        x, _ = self.middle_blks([x, t])
        b, c, h, w = x.shape
        
        prompt_loss = torch.tensor(0.0, device=inp.device)
        if self.is_prompt_pool and quality_score is not None:
            prompt = self.prompt_pool(x.reshape(b, c, h * w).permute(0, 2, 1), quality_score)
            prompt_e = prompt['prompted_embedding_e']
            x = self.atten_list[0](x, prompt_e)
            prompt_loss = prompt['prompt_loss']
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
        
        x = self.ending(x + encs[0])
        input_img_channels = inp.shape[1] // 2
        if input_img_channels * 2 == inp.shape[1]:
            orig_img = inp[:, :input_img_channels]
            return torch.cat([orig_img, x], dim=1), prompt_loss
        return x, prompt_loss

# Example usage and testing
if __name__ == "__main__":
    model = NAFNet(img_channel=6, out_channel=3, width=64, enc_blk_nums=[1, 1, 1, 18], dec_blk_nums=[1, 1, 1, 1], is_prompt_pool=True).cuda()
    x = torch.ones([1, 6, 256, 256]).cuda()
    quality_score = torch.tensor([[3.5]], device='cuda')
    result, prompt_loss = model(x, time=1., quality_score=quality_score)
    print(result.shape, prompt_loss.item())
