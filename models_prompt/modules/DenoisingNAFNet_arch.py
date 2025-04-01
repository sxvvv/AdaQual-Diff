import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .module_util import LayerNorm
from diffusers.models.embeddings import Timesteps
from models_prompt.modules.prompt import AdaptiveQualityPrompt

class Attention_cross(nn.Module):
    """ Cross-attention module for fusing prompts with image features """
    def __init__(self, dim, context_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x: Input features [B, C, H, W]
            context: Context features [B, SeqLen, context_dim]
        Returns:
            Attention-fused features [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        q = self.q(x_flat)
        q = q.view(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(context)
        k, v = kv.chunk(2, dim=-1)

        k = k.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attended_x = (attn @ v)
        attended_x = attended_x.transpose(1, 2).reshape(B, H * W, C)

        output = self.proj(attended_x)
        output = self.proj_drop(output)

        return output.permute(0, 2, 1).view(B, C, H, W)


class SimpleGate(nn.Module):
    """ Simple gating mechanism """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """ NAFNet basic block with time embedding support """
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        ffn_channel = c * FFN_Expand
        
        self.mlp = nn.Sequential(SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)) if time_emb_dim else None
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel//2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True))
        self.sg = SimpleGate()
        
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        """ Process time embedding for AdaLN modulation """
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x_time_tuple):
        """ Forward pass with time conditioning """
        inp, time_emb = x_time_tuple

        # Time-dependent modulation parameters
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time_emb, self.mlp)

        # Attention block
        x = inp
        x_norm1 = self.norm1(x)
        x = x_norm1 * (scale_att + 1) + shift_att

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN block
        x_norm2 = self.norm2(y)
        x = x_norm2 * (scale_ffn + 1) + shift_ffn

        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        out = y + x * self.gamma

        return out, time_emb


class NAFNet(nn.Module):
    """ NAFNet for diffusion models with time embedding and quality prompting """
    def __init__(self, img_channel=6, out_channel=3, width=64, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 18], dec_blk_nums=[1, 1, 1, 1],
                 upscale=1, trans_out=False, is_prompt_pool=True,
                 prompt_embed_dim=256, prompt_pool_size=10,
                 prompt_max_length=32, prompt_min_length=5,
                 attn_num_heads=8, attn_qkv_bias=False, attn_drop=0., proj_drop=0.
                 ):
        super().__init__()
        self.upscale = upscale
        self.trans_out = trans_out
        self.is_prompt_pool = is_prompt_pool

        # Time embedding
        fourier_dim = width
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            Timesteps(fourier_dim, flip_sin_to_cos=True, downscale_freq_shift=0.),
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        # Input and output convolutions
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, out_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        if self.trans_out:
            self.ending_trans = nn.Sequential(
                nn.Conv2d(width, 1, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
                nn.Sigmoid()
            )

        # Network structure
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Encoder path
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan *= 2

        # Middle blocks
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
        )

        # Quality prompting components
        if self.is_prompt_pool:
            self.prompt_pool = AdaptiveQualityPrompt(
                embed_dim=prompt_embed_dim,
                pool_size=prompt_pool_size,
                max_length=prompt_max_length,
                min_length=prompt_min_length
            )
            self.attn_local = Attention_cross(
                dim=chan,
                context_dim=prompt_embed_dim,
                num_heads=attn_num_heads, qkv_bias=attn_qkv_bias,
                attn_drop=attn_drop, proj_drop=proj_drop
            )
            self.attn_global = Attention_cross(
                dim=chan,
                context_dim=prompt_embed_dim,
                num_heads=attn_num_heads, qkv_bias=attn_qkv_bias,
                attn_drop=attn_drop, proj_drop=proj_drop
            )

        # Decoder path
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan // 2 * 4, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)]))

        # Calculate padding size
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        """ Ensure input size is divisible by padder_size """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, inp, time=1., quality_condition=None, raw_quality_scores=None):
        """ Forward pass with time and quality conditioning """
        time = torch.tensor([time], dtype=torch.long, device=inp.device) if not torch.is_tensor(time) else time
        if time.ndim == 0: time = time.unsqueeze(0)
        if time.shape[0] != inp.shape[0]: time = time.expand(inp.shape[0])
        t_emb = self.time_mlp(time)
        
        H_orig, W_orig = inp.shape[-2:]
        x = self.check_image_size(inp)
        x = self.intro(x)
        encs = [x]
        
        # Encoder path
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x, _ = encoder([x, t_emb])
            x = down(x)
            encs.append(x)
        
        # Middle blocks
        x, _ = self.middle_blks([x, t_emb])

        # Prompt integration
        prompt_loss = torch.tensor(0.0, device=inp.device)
        if self.is_prompt_pool and raw_quality_scores is not None:
            b, c_mid, h_mid, w_mid = x.shape
            x_flat = x.view(b, c_mid, h_mid * w_mid).permute(0, 2, 1)

            quality_prompt_dict = self.prompt_pool(x_flat, raw_quality_scores)

            prompt_e = quality_prompt_dict.get('prompted_embedding_e')
            prompt_g = quality_prompt_dict.get('prompted_embedding_g')
            prompt_loss = quality_prompt_dict.get('prompt_loss', torch.tensor(0.0, device=inp.device))

            # Fuse prompts with features
            if prompt_e is not None:
                 x = self.attn_local(x, prompt_e)
            if prompt_g is not None:
                 x = self.attn_global(x, prompt_g)

        # Decoder path with skip connections
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1][1:])):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t_emb])
        
        output = self.ending(x + encs[0])
        output = output[..., :H_orig, :W_orig]
        
        return_dict = {'predicted_residual': output, 'prompt_loss': prompt_loss}
        if self.trans_out:
            trans_map = self.ending_trans(x + encs[0])[..., :H_orig, :W_orig]
            return_dict['trans_map'] = trans_map
        return return_dict
