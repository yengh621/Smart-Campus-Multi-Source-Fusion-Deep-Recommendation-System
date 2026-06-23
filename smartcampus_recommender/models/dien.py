"""DIEN风格行为编码器：兴趣GRU + 查询注意力 + 兴趣演化GRU。"""
import torch
from torch import nn


class DIENEncoder(nn.Module):
    def __init__(self, num_items: int, dim: int, dropout: float,
                 use_internal_item_embedding: bool = True):
        super().__init__()
        self.item_emb = (nn.Embedding(num_items, dim, padding_idx=0)
                         if use_internal_item_embedding else None)
        self.meal_emb = nn.Embedding(4, dim)
        self.hour_emb = nn.Embedding(24, dim)
        self.weekday_emb = nn.Embedding(7, dim)
        self.holiday_emb = nn.Embedding(2, dim)
        self.numeric = nn.Sequential(nn.Linear(2, dim), nn.GELU())
        self.interest_gru = nn.GRU(dim, dim, batch_first=True)
        self.evolution_gru = nn.GRU(dim, dim, batch_first=True)
        self.query = nn.Linear(dim, dim)
        self.target_attention = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.target_input_gate = nn.Linear(dim, dim * 3)
        self.target_hidden_gate = nn.Linear(dim, dim * 3)
        self.target_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, items, meal, late, weekly, hour, weekday, holiday, mask, query,
                shared_item_weight=None):
        numeric = torch.stack([late, weekly], dim=-1)
        item_embedding = (self.item_emb(items) if shared_item_weight is None else
                          torch.nn.functional.embedding(items, shared_item_weight, padding_idx=0))
        x = (item_embedding + self.meal_emb(meal.clamp(0, 3)) + self.numeric(numeric) +
             self.hour_emb(hour.clamp(0, 23)) + self.weekday_emb(weekday.clamp(0, 6)) +
             self.holiday_emb(holiday.clamp(0, 1)))
        interests, _ = self.interest_gru(self.dropout(x))
        scores = (interests * self.query(query).unsqueeze(1)).sum(-1) / (interests.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, -1e4)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        evolved, _ = self.evolution_gru(interests * attention)
        lengths = mask.sum(1).clamp_min(1) - 1
        z = evolved[torch.arange(evolved.size(0), device=evolved.device), lengths]
        valid = mask.any(1).float().unsqueeze(-1)
        return self.norm(z) * valid, attention.squeeze(-1) * valid, interests * mask.unsqueeze(-1)

    def evolve_for_candidates(self, sequence, mask, candidates):
        """Candidate-aware AUGRU evolution used by the ranking stage."""
        batch, length, dim = sequence.shape
        count = candidates.size(1)
        seq = sequence.unsqueeze(1).expand(-1, count, -1, -1)
        query = candidates.unsqueeze(2).expand(-1, -1, length, -1)
        features = torch.cat([seq, query, seq - query, seq * query], dim=-1)
        scores = self.target_attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attention = torch.softmax(scores, dim=-1)
        valid = mask.any(1).view(batch, 1, 1)
        attention = attention * valid
        hidden = sequence.new_zeros(batch, count, dim)
        for step in range(length):
            current = sequence[:, step].unsqueeze(1).expand(-1, count, -1)
            ir, iz, inn = self.target_input_gate(current).chunk(3, -1)
            hr, hz, hn = self.target_hidden_gate(hidden).chunk(3, -1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            alpha = (attention[:, :, step] * mask[:, step].float().unsqueeze(1)).unsqueeze(-1)
            attended_update = alpha * update
            hidden = (1.0 - attended_update) * hidden + attended_update * candidate
        return self.target_norm(hidden) * valid, attention
