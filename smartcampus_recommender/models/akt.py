"""AKT-style temporal learning encoder with causal attention and time decay."""
import torch
from torch import nn

from models.time_encoding import Time2Vec, TimeDecayAttentionLayer


class AKTEncoder(nn.Module):
    def __init__(self, num_questions: int, num_concepts: int, dim: int, heads: int,
                 layers: int, dropout: float, max_len: int, time_decay_kernels: int = 4):
        super().__init__()
        self.question_emb = nn.Embedding(num_questions, dim, padding_idx=0)
        self.concept_emb = nn.Embedding(num_concepts, dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, dim)
        self.numeric = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.time2vec = Time2Vec(dim)
        self.relative_position = nn.Embedding(max_len + 1, dim)
        self.layers = nn.ModuleList([
            TimeDecayAttentionLayer(dim, heads, dropout, time_decay_kernels)
            for _ in range(layers)
        ])
        self.kt_head = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, questions, concepts, correct, intervals, night, wrong_streak,
                relative_position, mask, kg_concept_embedding=None):
        numeric = torch.stack([night, wrong_streak], dim=-1)
        x = (
            self.question_emb(questions)
            + self.concept_emb(concepts)
            + self.correct_emb(correct.long().clamp(0, 1))
            + self.numeric(numeric)
            + self.time2vec(torch.log1p(intervals))
            + self.relative_position(
                relative_position.clamp_max(self.relative_position.num_embeddings - 1)
            )
        )
        if kg_concept_embedding is not None:
            x = x + kg_concept_embedding

        hidden = x
        safe_mask = mask.clone()
        empty = ~safe_mask.any(1)
        if empty.any():
            safe_mask[empty, 0] = True

        for layer in self.layers:
            hidden = layer(hidden, intervals, ~safe_mask)

        hidden = self.norm(hidden)
        kt_logits = self.kt_head(hidden).squeeze(-1)
        weights = mask.float()
        z_stu = (
            hidden * weights.unsqueeze(-1)
        ).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.0)
        return z_stu, kt_logits
