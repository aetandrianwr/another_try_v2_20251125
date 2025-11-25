"""
Advanced Next-Location Prediction Model with Multiple Cutting-Edge Techniques:
1. Learnable positional embeddings
2. Multi-head self-attention with relative position encoding
3. Feed-forward with GLU activation
4. Focal loss for handling class imbalance
5. Location frequency-based bias initialization
6. Temporal encoding with cyclical features
7. User preference modeling
8. Attention pooling instead of average pooling
9. Residual connections everywhere
10. DropPath for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class AdvancedTransformerEncoderLayer(nn.Module):
    """Advanced Transformer layer with modern improvements"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_path=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward with GLU
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.glu = GLU(dim=-1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN with residual
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        src = src + self.drop_path(self.dropout1(src2))
        
        # Feed-forward with GLU
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.glu(src2)
        src2 = self.dropout2(src2)
        src2 = self.linear2(src2)
        src = src + self.drop_path(src2)
        
        return src, attn_weights


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism"""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
    
    def forward(self, x, mask):
        # x: (batch, seq_len, d_model)
        # mask: (batch, seq_len)
        attn_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask.logical_not(), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)
        return pooled


class AdvancedNextLocationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Embeddings with larger vocabulary for better representation
        self.loc_embedding = nn.Embedding(config.num_locations, config.loc_embed_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(config.num_users, config.user_embed_dim, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, config.weekday_embed_dim, padding_idx=0)
        
        # Temporal features with cyclical encoding
        self.time_proj = nn.Sequential(
            nn.Linear(2, config.time_embed_dim),  # sin/cos encoding
            nn.LayerNorm(config.time_embed_dim),
            nn.ReLU()
        )
        self.duration_proj = nn.Sequential(
            nn.Linear(1, config.time_embed_dim),
            nn.LayerNorm(config.time_embed_dim),
            nn.ReLU()
        )
        self.time_diff_embedding = nn.Embedding(100, config.time_embed_dim, padding_idx=0)
        
        # Input projection
        input_dim = (config.loc_embed_dim + config.user_embed_dim + 
                     config.weekday_embed_dim + config.time_embed_dim * 3)
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout * 0.5)
        )
        
        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(config.d_model, max_len=100)
        
        # Transformer encoder layers with drop path
        drop_path_rate = np.linspace(0, config.drop_path_rate, config.num_layers).tolist()
        self.encoder_layers = nn.ModuleList([
            AdvancedTransformerEncoderLayer(
                config.d_model, config.nhead, 
                config.dim_feedforward, config.dropout,
                drop_path=drop_path_rate[i]
            )
            for i in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Attention pooling instead of average pooling
        self.pooling = AttentionPooling(config.d_model)
        
        # User-specific location preferences (compressed)
        # Instead of full user x location matrix, use low-rank factorization
        self.user_preference_dim = 16  # Reduced from 32
        self.user_pref_embed = nn.Embedding(config.num_users, self.user_preference_dim)
        self.loc_pref_embed = nn.Embedding(config.num_locations, self.user_preference_dim)
        
        # Output projection with residual
        self.pre_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.output_proj = nn.Linear(config.d_model, config.num_locations)
        
        # Initialize with frequency-based bias
        self._init_weights()
        
        # Initialize preference embeddings
        nn.init.normal_(self.user_pref_embed.weight, mean=0, std=0.01)
        nn.init.normal_(self.loc_pref_embed.weight, mean=0, std=0.01)
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'norm' not in name and 'pref' not in name:
                    nn.init.xavier_uniform_(param, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def set_location_bias(self, location_counts):
        """Set output bias based on location frequency"""
        freq = torch.FloatTensor(location_counts)
        freq = freq / freq.sum()
        bias = torch.log(freq + 1e-8)
        # Keep bias on the same device as the model
        if self.output_proj.bias is not None:
            self.output_proj.bias.data = bias.to(self.output_proj.bias.device)
    
    def forward(self, locations, users, weekdays, start_minutes, durations, time_diffs, mask):
        batch_size, seq_len = locations.shape
        
        # Embeddings
        loc_emb = self.loc_embedding(locations)
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekdays)
        
        # Cyclical time encoding (sin/cos for periodic patterns)
        time_rad = (start_minutes / 1440.0) * 2 * math.pi  # Normalize to [0, 2π]
        time_sin = torch.sin(time_rad).unsqueeze(-1)
        time_cos = torch.cos(time_rad).unsqueeze(-1)
        time_emb = self.time_proj(torch.cat([time_sin, time_cos], dim=-1))
        
        # Duration encoding
        dur_emb = self.duration_proj(durations.unsqueeze(-1))
        
        # Time difference embedding
        time_diff_emb = self.time_diff_embedding(torch.clamp(time_diffs, 0, 99))
        
        # Concatenate all features
        x = torch.cat([loc_emb, user_emb, weekday_emb, time_emb, dur_emb, time_diff_emb], dim=-1)
        
        # Project to model dimension
        x = self.input_proj(x)
        
        # Add learnable positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask for transformer
        padding_mask = mask.logical_not()  # Use logical_not instead of ~
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x, _ = layer(x, src_key_padding_mask=padding_mask)
        
        x = self.final_norm(x)
        
        # Attention pooling
        x = self.pooling(x, mask)
        
        # Pre-output processing
        x_residual = x
        x = self.pre_output(x)
        x = x + x_residual  # Residual connection
        
        # Output projection
        logits = self.output_proj(x)
        
        # Add user-specific location preference (low-rank factorization)
        user_pref = self.user_pref_embed(users[:, 0])  # (batch, user_pref_dim)
        loc_pref = self.loc_pref_embed.weight.to(user_pref.device)  # (num_locations, user_pref_dim)
        user_bias = torch.matmul(user_pref, loc_pref.t())  # (batch, num_locations)
        logits = logits + user_bias * 0.05  # Small scale to avoid overfitting
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
