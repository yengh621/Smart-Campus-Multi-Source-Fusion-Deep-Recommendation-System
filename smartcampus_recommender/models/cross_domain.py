"""个体跨域映射：根据同一global_user_id的学业表征预测其行为表征。"""
from __future__ import annotations

from torch import nn


class PersonalizedMapper(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, input_dim: int | None = None):
        super().__init__()
        input_dim = input_dim or dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.LayerNorm(dim),
        )

    def forward(self, z_stu):
        return self.network(z_stu)
