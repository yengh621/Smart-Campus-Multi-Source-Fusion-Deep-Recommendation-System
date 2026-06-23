"""兴趣漂移感知：长期兴趣、AUGRU短期兴趣、多兴趣胶囊与动态门控。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AUGRUShortTerm(nn.Module):
    """AUGRU风格：候选相关注意力直接调制GRU状态更新幅度。"""
    def __init__(self, dim: int):
        super().__init__()
        self.input_gate = nn.Linear(dim, dim * 3)
        self.hidden_gate = nn.Linear(dim, dim * 3)
        self.query = nn.Linear(dim, dim)

    def forward(self, sequence, mask, query, window: int):
        batch, length, dim = sequence.shape
        lengths = mask.sum(1)
        start = (lengths - window).clamp_min(0)
        positions = torch.arange(length, device=sequence.device).unsqueeze(0)
        active = mask & (positions >= start.unsqueeze(1)) & (positions < lengths.unsqueeze(1))
        relevance = torch.sigmoid((sequence * self.query(query).unsqueeze(1)).sum(-1) / dim ** 0.5)
        # 越新的行为权重越高，时间相关性与候选相关性共同控制更新门。
        recency = ((positions - start.unsqueeze(1) + 1).float() / window).clamp(0, 1)
        update_attention = relevance * recency
        hidden = sequence.new_zeros(batch, dim)
        for step in range(length):
            ir, iz, inn = self.input_gate(sequence[:, step]).chunk(3, -1)
            hr, hz, hn = self.hidden_gate(hidden).chunk(3, -1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            # 标准 AUGRU：注意力直接缩放 GRU 的更新门，而非在 GRU 输出后插值。
            alpha = (update_attention[:, step] * active[:, step].float()).unsqueeze(-1)
            attended_update = alpha * update
            hidden = (1.0 - attended_update) * hidden + attended_update * candidate
        return hidden, update_attention


class MultiInterestExtractor(nn.Module):
    """使用多个可学习兴趣查询从行为序列提取K个兴趣向量（MIND/ComiRec思想）。"""
    def __init__(self, dim: int, num_interests: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_interests, dim) * 0.02)
        self.projection = nn.Linear(dim, dim)

    def forward(self, sequence, mask):
        projected = self.projection(sequence)
        scores = torch.einsum("bld,kd->bkl", projected, self.queries) / projected.size(-1) ** 0.5
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=-1)
        interests = torch.einsum("bkl,bld->bkd", weights, sequence)
        effective_count = mask.sum(1).clamp_max(self.queries.size(0))
        slot_mask = (torch.arange(self.queries.size(0), device=sequence.device).unsqueeze(0) <
                     effective_count.unsqueeze(1))
        slot = slot_mask.unsqueeze(-1).to(interests.dtype)
        return F.normalize(interests, dim=-1) * slot, weights * slot, slot_mask


class InterestDriftModule(nn.Module):
    def __init__(self, dim: int, num_interests: int, short_window: int, dropout: float,
                 recent_hours: float = 168.0, history_hours: float = 720.0):
        super().__init__()
        self.short_window = short_window
        self.recent_hours = recent_hours
        self.history_hours = history_hours
        self.short_encoder = AUGRUShortTerm(dim)
        self.multi_interest = MultiInterestExtractor(dim, num_interests)
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 2, dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def window_drift(sequence, mask, age_hours, recent_hours, history_hours):
        """比较序列前后半窗兴趣余弦距离，0稳定、2高度漂移。"""
        batch, length, _ = sequence.shape
        lengths = mask.sum(1).clamp_min(1)
        split = (lengths // 2).clamp_min(1)
        positions = torch.arange(length, device=sequence.device).unsqueeze(0)
        first_mask = mask & (age_hours > recent_hours) & (age_hours <= history_hours)
        second_mask = mask & (age_hours <= recent_hours)
        # 稀疏用户没有两个真实时间窗时，才退化为序列前后半段。
        fallback = ~(first_mask.any(1) & second_mask.any(1))
        half_first = mask & (positions < split.unsqueeze(1))
        half_second = mask & (positions >= split.unsqueeze(1))
        first_mask = torch.where(fallback.unsqueeze(1), half_first, first_mask)
        second_mask = torch.where(fallback.unsqueeze(1), half_second, second_mask)
        first = (sequence * first_mask.unsqueeze(-1)).sum(1) / first_mask.sum(1, keepdim=True).clamp_min(1)
        second = (sequence * second_mask.unsqueeze(-1)).sum(1) / second_mask.sum(1, keepdim=True).clamp_min(1)
        valid = (first_mask.any(1) & second_mask.any(1)).float()
        return (1.0 - F.cosine_similarity(first, second, dim=-1)).clamp(0, 2) * valid

    def forward(self, long_interest, sequence, mask, query, weekly_frequency, age_hours):
        short_interest, short_attention = self.short_encoder(
            sequence, mask, query, self.short_window)
        multi_interests, multi_weights, multi_mask = self.multi_interest(sequence, mask)
        drift = self.window_drift(sequence, mask, age_hours, self.recent_hours, self.history_hours)
        activity = (weekly_frequency * mask.float()).sum(1) / mask.sum(1).clamp_min(1)
        gate = self.gate(torch.cat([long_interest, short_interest,
                                    drift.unsqueeze(-1), activity.unsqueeze(-1)], dim=-1))
        dynamic = self.norm(gate * short_interest + (1.0 - gate) * long_interest)
        return {
            "dynamic_interest": dynamic,
            "short_interest": short_interest,
            "multi_interests": multi_interests,
            "multi_interest_mask": multi_mask,
            "drift_score": drift,
            "short_gate": gate.squeeze(-1),
            "short_attention": short_attention,
            "multi_interest_weights": multi_weights,
        }
