import torch
import torch.nn as nn

class AdaptiveQualityPrompt(nn.Module):
    def __init__(self, embed_dim=256, pool_size=10, max_length=32, min_length=5, quality_threshold=3.0, prompt_init='uniform'):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.max_length = max_length
        self.min_length = min_length
        self.quality_threshold = quality_threshold  # DeQAScore threshold (range 1-5)
        self.input_projections = nn.ModuleDict()
        
        # Global quality prompt (fixed length)
        self.global_prompt_length = 8
        self.global_quality_prompt = nn.Parameter(torch.randn(1, self.global_prompt_length, embed_dim))
        if prompt_init == 'uniform':
            nn.init.xavier_uniform_(self.global_quality_prompt)
        elif prompt_init == 'zero':
            nn.init.zeros_(self.global_quality_prompt)

        # Local repair prompt pool (variable length)
        self.local_prompt_pool = nn.Parameter(torch.randn(pool_size, max_length, embed_dim))
        if prompt_init == 'uniform':
            nn.init.xavier_uniform_(self.local_prompt_pool)
        elif prompt_init == 'zero':
            nn.init.zeros_(self.local_prompt_pool)

        # Prompt keys (for selection)
        self.prompt_keys = nn.Parameter(torch.randn(pool_size, embed_dim))
        nn.init.uniform_(self.prompt_keys, -1, 1)

        # Lightweight fusion module
        self.quality_fusion = nn.Sequential(
            nn.Linear(1, embed_dim),  # Maps DeQAScore scalar to embedding dimension
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """L2 normalization"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def adjust_prompt_length(self, quality_scores):
        """Dynamically adjust local prompt length based on quality score"""
        # quality_scores: [batch_size], range 1-5
        normalized_quality = quality_scores / 5.0  # Normalize to 0-1
        length_range = self.max_length - self.min_length
        # Higher quality = shorter prompts, lower quality = longer prompts
        prompt_lengths = self.min_length + torch.round(length_range * (1 - normalized_quality)).int()
        return torch.clamp(prompt_lengths, self.min_length, self.max_length)

    def forward(self, x_embed, quality_scores):
        batch_size = x_embed.shape[0]
        input_embed_dim = x_embed.shape[-1]
        
        # Ensure quality_scores has gradients if in training mode
        quality_scores = quality_scores.clone()
        if self.training:
            quality_scores.requires_grad_(True)
            
        # Handle dimension mismatch with projection if needed
        if input_embed_dim != self.embed_dim:
            proj_key = f"proj_{input_embed_dim}"
            if proj_key not in self.input_projections:
                print(f"AdaptiveQualityPrompt: Input embedding dim ({input_embed_dim}) != expected embed_dim ({self.embed_dim}). Adding projection layer.")
                self.input_projections[proj_key] = nn.Linear(input_embed_dim, self.embed_dim).to(x_embed.device)
            x_embed = self.input_projections[proj_key](x_embed)

        out = {}

        # Expand global prompt to batch size
        global_prompt = self.global_quality_prompt.expand(batch_size, -1, -1)  # [batch_size, global_prompt_length, embed_dim]

        # Get adaptive prompt lengths based on quality
        prompt_lengths = self.adjust_prompt_length(quality_scores)  # [batch_size]

        # Calculate mean embedding
        x_embed_mean = torch.mean(x_embed, dim=1)  # [batch_size, embed_dim]
        
        # Select prompts from pool based on quality
        batched_local_prompt = []
        
        for i in range(batch_size):
            # Get appropriate prompt length for this item
            current_len = max(self.min_length, min(int(prompt_lengths[i].item()), self.max_length))
            
            # Select different prompts based on quality threshold
            if quality_scores[i] > self.quality_threshold:
                # Use first 2 prompts for high-quality images
                selected_prompt = self.local_prompt_pool[:2, :current_len, :]
            else:
                # Use prompts 2-6 for low-quality images
                end_idx = min(6, self.pool_size)
                selected_prompt = self.local_prompt_pool[2:end_idx, :current_len, :]
            
            # Average the selected prompts
            local_prompt = torch.mean(selected_prompt, dim=0)  # [current_len, embed_dim]
            batched_local_prompt.append(local_prompt)
        
        # Pad prompts to same length for batching
        batched_local_prompt = torch.nn.utils.rnn.pad_sequence(
            batched_local_prompt, batch_first=True, padding_value=0.0
        )  # [batch_size, max_current_len, embed_dim]
        
        # Convert quality scores to embeddings
        quality_embedding = self.quality_fusion(quality_scores.view(-1, 1).float())  # [batch_size, embed_dim]
        quality_embedding = quality_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Add quality embeddings to prompts
        global_prompt = global_prompt + quality_embedding  # [batch_size, global_prompt_length, embed_dim]
        local_prompt = batched_local_prompt + quality_embedding  # [batch_size, max_current_len, embed_dim]
        
        # Combine global and local prompts
        batched_prompt = torch.cat([global_prompt, local_prompt], dim=1)  # [batch_size, total_length, embed_dim]
        
        # Simple placeholder prompt loss
        prompt_loss = torch.tensor(0.1, device=x_embed.device)
        
        # Prepare return values
        out['prompt_loss'] = prompt_loss
        out['prompted_embedding'] = batched_prompt
        out['total_prompt_len'] = batched_prompt.shape[1]
        
        return out
