"""Time2Vec、相对位置编码与可学习时间衰减注意力。"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class Time2Vec(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.randn(max(dim - 1, 1)))
        self.periodic_bias = nn.Parameter(torch.zeros(max(dim - 1, 1)))
        self.dim = dim

    def forward(self, time):
        value = time.unsqueeze(-1)
        linear = value * self.linear_weight + self.linear_bias
        if self.dim == 1:
            return linear
        periodic = torch.sin(value * self.periodic_weight + self.periodic_bias)
        return torch.cat([linear, periodic[..., :self.dim-1]], dim=-1)


class TimeDecayAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, num_kernels: int = 4):
        super().__init__()
        self.heads = heads
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        # 每个注意力头学习多个时间尺度及其混合系数。
        base = torch.logspace(0, -math.log10(720.0), num_kernels)
        initial = torch.log(torch.expm1(base.clamp_min(1e-6))).repeat(heads, 1)
        self.decay_rates = nn.Parameter(initial)
        self.mixture_logits = nn.Parameter(torch.zeros(heads, num_kernels))
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x, intervals, padding_mask):
        batch, length, _ = x.shape
        cumulative = torch.cumsum(intervals, dim=1)
        delta = torch.abs(cumulative.unsqueeze(2) - cumulative.unsqueeze(1))
        rates = F.softplus(self.decay_rates).view(1, self.heads, -1, 1, 1)
        mixtures = torch.softmax(self.mixture_logits, dim=-1).view(1, self.heads, -1, 1, 1)
        # K(Δt)=Σ_k α_k exp(-λ_k Δt)，以log K作为注意力加性偏置。
        kernel = (mixtures * torch.exp(-rates * delta.unsqueeze(1).unsqueeze(2))).sum(2)
        bias = torch.log(kernel.clamp_min(1e-8))
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        bias = bias.masked_fill(causal.view(1, 1, length, length), float("-inf"))
        bias = bias.expand(batch, -1, -1, -1).reshape(batch * self.heads, length, length)
        padding_bias = torch.zeros_like(padding_mask, dtype=x.dtype).masked_fill(padding_mask, float("-inf"))
        attended, _ = self.attention(x, x, x, attn_mask=bias,
                                     key_padding_mask=padding_bias, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))
