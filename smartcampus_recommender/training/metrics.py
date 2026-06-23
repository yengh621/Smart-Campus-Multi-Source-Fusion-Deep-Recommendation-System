from __future__ import annotations

import math

import numpy as np
import torch


def ranking_metrics(logits: torch.Tensor, targets: torch.Tensor, ks=(5, 10), weights=None) -> dict[str, float]:
    """单正样本Top-K指标；AUC按每个用户目标类别与全部负类别成对比较。"""
    scores = logits.detach().float().cpu().clone()
    y = targets.detach().cpu().long()
    w = (torch.ones(len(y)) if weights is None else weights.detach().float().cpu())
    w = w / w.sum().clamp_min(1e-8)
    if scores.size(1) > 2:
        scores[:, :2] = -1e9  # PAD/UNK 不参与推荐
    target_scores = scores.gather(1, y[:, None])
    greater = (scores > target_scores).sum(1).float()
    equal = (scores == target_scores).sum(1).float() - 1.0
    negatives = max(scores.size(1) - 1, 1)
    auc = (((negatives - greater - 0.5 * equal) / negatives)*w).sum().item()
    order = torch.argsort(scores, dim=1, descending=True)
    result = {"auc": auc}
    for k in ks:
        top = order[:, :min(k, order.size(1))]
        match = top.eq(y[:, None])
        hit = match.any(1).float()
        positions = torch.argmax(match.float(), dim=1) + 1
        ndcg = torch.where(hit.bool(), 1.0 / torch.log2(positions.float() + 1.0), torch.zeros_like(hit))
        result[f"recall@{k}"] = (hit*w).sum().item()
        result[f"ndcg@{k}"] = (ndcg*w).sum().item()
    return result


def merge_metric_batches(logits_batches, target_batches, ks=(5, 10), weight_batches=None):
    weights = torch.cat(weight_batches) if weight_batches else None
    return ranking_metrics(torch.cat(logits_batches), torch.cat(target_batches), ks, weights)
