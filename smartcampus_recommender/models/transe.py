"""TransE：h + r ≈ t，使用负采样间隔损失训练知识图谱表征。"""
import torch
from torch import nn
from torch.nn import functional as F


class TransEEncoder(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, margin: float = 1.0):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim, padding_idx=0)
        self.relation = nn.Embedding(num_relations, dim, padding_idx=0)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def distance(self, triples: torch.Tensor) -> torch.Tensor:
        h, r, t = triples.unbind(-1)
        return torch.linalg.vector_norm(self.entity(h) + self.relation(r) - self.entity(t), ord=1, dim=-1)

    def loss(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if positive.numel() == 0:
            return self.entity.weight.sum() * 0.0
        return F.relu(self.margin + self.distance(positive) - self.distance(negative)).mean()

