"""
论文总损失：
L = λkg LTransE + λkt LAKT + λp (LInfoNCE+LVICReg) + LPLE-uncertainty
    + λr Lretrieval + λd Ldrift

其中跨域项使用对称InfoNCE保持个体可分性，并以VICReg方差/协方差约束防止坍缩；
三项PLE任务由可学习同方差不确定性自动加权。
"""
import torch
from torch.nn import functional as F


def weighted_mean(values, weights):
    weights = weights.to(values.dtype)
    return (values*weights).sum() / weights.sum().clamp_min(1e-8)


def recommendation_loss(outputs, batch, config):
    targets = {
        "knowledge": batch["target_concept"],
        "course": batch["target_course"],
        "consume": batch["target_consume"],
    }
    losses = {}
    validity = {"knowledge": batch["knowledge_valid"], "course": batch["course_valid"],
                "consume": batch["consume_valid"]}
    active = {}
    for task, target in targets.items():
        logits = outputs[task].clone()
        if logits.size(1) > 2:
            logits[:, :2] = torch.finfo(logits.dtype).min
        # During evaluation a missed target has a deliberately very low final
        # logit and must count as a hard error instead of disappearing from loss.
        valid = validity[task]
        active[task] = bool(valid.any())
        if valid.any():
            raw = F.cross_entropy(logits[valid], target[valid], reduction="none")
            weight = batch["sample_weight"][valid]
            if task == "knowledge":
                confidence = torch.where(
                    batch["target_direct_concept"][valid],
                    torch.full_like(weight, config.direct_label_confidence),
                    torch.where(
                        batch["target_resolved_concept"][valid],
                        torch.full_like(weight, config.resolved_label_confidence),
                        torch.full_like(weight, config.weak_label_confidence),
                    ),
                )
                weight = weight * confidence
            losses[task] = weighted_mean(raw, weight)
        else:
            losses[task] = logits.sum() * 0.0
    # 同方差不确定性自动加权：0.5*exp(-s_t)*L_t + 0.5*s_t。
    total = 0.0
    active_weight = outputs["task_weights"].new_zeros(())
    for index, task in enumerate(("knowledge", "course", "consume")):
        if not active[task]:
            continue
        total = total + outputs["task_weights"][index] * losses[task]
        active_weight = active_weight + outputs["task_weights"][index]
    return total / active_weight.clamp_min(1e-8), losses


def retrieval_loss(outputs, batch):
    targets = {"knowledge": batch["target_concept"], "course": batch["target_course"],
               "consume": batch["target_consume"]}
    losses = []
    for task, target in targets.items():
        logits = outputs["retrieval_logits"][task]
        valid = batch[{"knowledge": "knowledge_valid", "course": "course_valid",
                       "consume": "consume_valid"}[task]]
        if valid.any():
            raw = F.cross_entropy(logits[valid], target[valid], reduction="none")
            losses.append(weighted_mean(raw, batch["sample_weight"][valid]))
    return torch.stack(losses).mean() if losses else outputs["z_stu"].sum()*0.0


def kt_loss(outputs, batch):
    # h_t 预测下一次答题 y_{t+1}，避免把当前 correct embedding 直接泄露给当前标签。
    logits = outputs["kt_logits"][:, :-1]
    target = batch["correct"][:, 1:]
    mask = batch["learning_mask"][:, 1:] & batch["knowledge_valid"].unsqueeze(1)
    if not mask.any():
        return logits.sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = batch["sample_weight"].unsqueeze(1).expand_as(raw)
    return weighted_mean(raw[mask], weights[mask])


def _variance_covariance_loss(z):
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    centered = z - z.mean(0)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / (z.size(0) - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return variance, off_diagonal.pow(2).sum() / z.size(1)


def personalization_loss(outputs, batch, config):
    z, z_real = outputs["z_behavior"], outputs["z_real"]
    available = ((batch["consume_mask"].any(1) | batch["door_mask"].any(1)) &
                 batch["alignment_valid"])
    z, z_real = z[available], z_real[available]
    groups = batch["split_group"][available]
    sample_weights = batch["sample_weight"][available]
    if z.size(0) >= 2:
        left, right = F.normalize(z, dim=-1), F.normalize(z_real, dim=-1)
        similarity = left @ right.T / config.contrastive_temperature
        positive = groups.unsqueeze(1).eq(groups.unsqueeze(0))
        log_prob_lr = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        log_prob_rl = similarity.T - torch.logsumexp(similarity.T, dim=1, keepdim=True)
        negative_inf = torch.finfo(similarity.dtype).min
        loss_lr = -torch.logsumexp(log_prob_lr.masked_fill(~positive, negative_inf), dim=1)
        loss_rl = -torch.logsumexp(log_prob_rl.masked_fill(~positive.T, negative_inf), dim=1)
        info_nce = 0.5*(weighted_mean(loss_lr, sample_weights) +
                        weighted_mean(loss_rl, sample_weights))
        invariance = F.mse_loss(z, z_real)
        var_left, cov_left = _variance_covariance_loss(z)
        var_right, cov_right = _variance_covariance_loss(z_real)
        vicreg = (config.vicreg_invariance_weight * invariance +
                  config.vicreg_variance_weight * 0.5 * (var_left + var_right) +
                  config.vicreg_covariance_weight * 0.5 * (cov_left + cov_right))
    else:
        info_nce = outputs["z_behavior"].sum() * 0.0
        vicreg = info_nce
    aux_logits = outputs["behavior_aux_logits"].clone()
    if aux_logits.size(1) > 2:
        aux_logits[:, :2] = torch.finfo(aux_logits.dtype).min
    valid = batch["consume_valid"]
    dien_aux = (weighted_mean(
                    F.cross_entropy(aux_logits[valid], batch["target_consume"][valid], reduction="none"),
                    batch["sample_weight"][valid])
                if valid.any() else aux_logits.sum() * 0.0)
    return config.info_nce_weight * info_nce + vicreg + 0.1 * dien_aux


def drift_smoothing_loss(outputs, batch=None):
    """稳定用户保持长短期连续；检测到明显漂移时自动放松平滑约束。"""
    distance = (outputs["short_interest"] - outputs["z_long"]).pow(2).mean(-1)
    stable_weight = (1.0 - outputs["drift_score"].detach() / 2.0).clamp(0, 1)
    values = stable_weight*distance
    return weighted_mean(values, batch["sample_weight"]) if batch is not None else values.mean()


def interest_regularization(outputs, batch):
    """Penalize collapsed interest capsules and imbalanced routing."""
    interests = outputs["multi_interests"]
    routing = outputs["multi_interest_weights"]
    slot_mask = outputs["multi_interest_mask"]
    if not (slot_mask.sum(1) >= 2).any():
        zero = interests.sum() * 0.0
        return zero, zero
    normalized = F.normalize(interests, dim=-1)
    gram = normalized @ normalized.transpose(1, 2)
    pair_mask = slot_mask.unsqueeze(2) & slot_mask.unsqueeze(1)
    diagonal = torch.eye(gram.size(1), device=gram.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal
    diversity = gram.masked_select(pair_mask).pow(2).mean()
    usage = routing.sum(-1)
    usage = usage / usage.sum(1, keepdim=True).clamp_min(1e-8)
    target = slot_mask.float() / slot_mask.sum(1, keepdim=True).clamp_min(1)
    balance = (((usage - target).pow(2) * slot_mask).sum() /
               slot_mask.sum().clamp_min(1))
    return diversity, balance
