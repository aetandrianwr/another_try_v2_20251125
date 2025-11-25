"""
Next-Location Prediction Model with Transformer Architecture
Incorporates modern techniques: Layer Normalization, Residual connections,
Multi-head attention, Position-wise FFN, Positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer with Pre-LN"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN: Normalize first, then apply attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Pre-LN: Normalize first, then apply FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class NextLocationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.loc_embedding = nn.Embedding(config.num_locations, config.loc_embed_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(config.num_users, config.user_embed_dim, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, config.weekday_embed_dim, padding_idx=0)
        
        # Time encodings
        self.time_proj = nn.Linear(1, config.time_embed_dim)
        self.duration_proj = nn.Linear(1, config.time_embed_dim)
        self.time_diff_embedding = nn.Embedding(100, config.time_embed_dim, padding_idx=0)
        
        # Input projection
        input_dim = (config.loc_embed_dim + config.user_embed_dim + 
                     config.weekday_embed_dim + config.time_embed_dim * 3)
        
        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=100)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.nhead, 
                                   config.dim_feedforward, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Output layers with bottleneck
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_locations)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, locations, users, weekdays, start_minutes, durations, time_diffs, mask):
        batch_size, seq_len = locations.shape
        
        # Embeddings
        loc_emb = self.loc_embedding(locations)
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekdays)
        
        # Time features
        time_emb = self.time_proj(start_minutes.unsqueeze(-1))
        dur_emb = self.duration_proj(durations.unsqueeze(-1))
        time_diff_emb = self.time_diff_embedding(torch.clamp(time_diffs, 0, 99))
        
        # Concatenate all features
        x = torch.cat([loc_emb, user_emb, weekday_emb, time_emb, dur_emb, time_diff_emb], dim=-1)
        
        # Project to model dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask for transformer (True for padding positions)
        padding_mask = ~mask
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        
        x = self.final_norm(x)
        
        # Global average pooling with masking
        mask_expanded = mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
