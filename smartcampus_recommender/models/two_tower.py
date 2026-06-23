"""双塔召回：用户塔与三类候选物品塔，使用余弦相似度完成Top-N召回。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerRetriever(nn.Module):
    TASKS = ("knowledge", "course", "consume")

    def __init__(self, user_input_dim: int, dim: int, candidate_sizes: dict[str, int],
                 temperature: float, dropout: float):
        super().__init__()
        self.temperature = temperature
        self.user_towers = nn.ModuleDict({task: nn.Sequential(
            nn.Linear(user_input_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim)
        ) for task in self.TASKS})
        self.candidates = nn.ModuleDict({task: nn.Embedding(size, dim, padding_idx=0)
                                         for task, size in candidate_sizes.items()})
        for embedding in self.candidates.values():
            nn.init.xavier_uniform_(embedding.weight)

    def candidate_matrix(self, task: str, graph_embedding=None):
        matrix = self.candidates[task].weight
        return matrix if graph_embedding is None else matrix + graph_embedding

    def forward(self, user_input, graph_candidates: dict[str, torch.Tensor | None], topk: int,
                 targets: dict[str, torch.Tensor] | None = None, force_target: bool = False,
                 multi_interests: torch.Tensor | None = None,
                 multi_interest_mask: torch.Tensor | None = None,
                 exclusion_indices: dict[str, torch.Tensor] | None = None,
                 exclusion_masks: dict[str, torch.Tensor] | None = None):
        full_scores, indices, embeddings = {}, {}, {}
        for task in self.TASKS:
            task_user_input = user_input[task] if isinstance(user_input, dict) else user_input
            user = F.normalize(self.user_towers[task](task_user_input), dim=-1)
            matrix = self.candidate_matrix(task, graph_candidates.get(task))
            normalized_items = F.normalize(matrix, dim=-1)
            score = user @ normalized_items.T / self.temperature
            if task == "consume" and multi_interests is not None:
                interest_score = torch.einsum("bkd,nd->bkn", F.normalize(multi_interests, dim=-1),
                                               normalized_items) / self.temperature
                if multi_interest_mask is not None:
                    interest_score = interest_score.masked_fill(
                        ~multi_interest_mask.unsqueeze(-1), -1e4)
                interest_score = interest_score.max(1).values
                has_interest = (multi_interest_mask.any(1) if multi_interest_mask is not None
                                else torch.ones(score.size(0), dtype=torch.bool, device=score.device))
                score = torch.where(has_interest.unsqueeze(1), torch.maximum(score, interest_score), score)
            if exclusion_indices is not None and task in exclusion_indices:
                blocked = torch.zeros_like(score, dtype=torch.bool)
                indices_to_block = exclusion_indices[task].clamp(0, score.size(1) - 1)
                valid_block = indices_to_block.gt(1)
                if exclusion_masks is not None and task in exclusion_masks:
                    valid_block = valid_block & exclusion_masks[task]
                blocked.scatter_(1, indices_to_block, valid_block)
                score = score.masked_fill(blocked, -1e4)
            if score.size(1) > 2:
                score[:, :2] = -1e4
            k = min(topk, score.size(1))
            ranked_k = min(k + (1 if force_target else 0), score.size(1))
            ranked = torch.topk(score, k=ranked_k, dim=-1).indices
            candidate_idx = ranked[:, :k]
            if force_target and targets is not None and k > 0 and ranked_k > k:
                target = targets[task]
                present = candidate_idx.eq(target.unsqueeze(1)).any(1)
                # Keep every naturally retrieved hard negative. The extra slot
                # is the positive when missed, otherwise the next ranked item.
                extra = torch.where(present, ranked[:, k], target).unsqueeze(1)
                candidate_idx = torch.cat([candidate_idx, extra], dim=1)
            full_scores[task] = score
            indices[task] = candidate_idx
            embeddings[task] = matrix[candidate_idx]
        return full_scores, indices, embeddings
