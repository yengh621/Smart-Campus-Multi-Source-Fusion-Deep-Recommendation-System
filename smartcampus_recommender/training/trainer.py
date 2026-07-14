from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from config import ExperimentConfig
from data.preprocessing import DataArtifacts
from models.full_model import SmartCampusRecommender
from training.losses import (drift_smoothing_loss, interest_regularization, kt_loss,
                             personalization_loss, recommendation_loss, retrieval_loss)
from training.metrics import merge_metric_batches


TASKS = ("knowledge", "course", "consume")


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class Trainer:
    def __init__(self, model: SmartCampusRecommender, artifacts: DataArtifacts,
                 config: ExperimentConfig, logger, checkpoint_name: str = "best_model.pth"):
        self.model, self.artifacts, self.config, self.logger = model, artifacts, config, logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.amp_enabled = bool(config.use_amp and self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=config.lr_factor, patience=config.lr_patience)
        self.checkpoint = config.output_path / checkpoint_name
        checkpoint_path = Path(checkpoint_name)
        if checkpoint_path.name == "best_model.pth":
            latest_name = "last_checkpoint.pth"
        else:
            latest_name = checkpoint_path.name.replace("_best.pth", "_last.pth")
        self.last_checkpoint = self.checkpoint.with_name(latest_name)
        self.kg = torch.tensor(artifacts.kg_triples, dtype=torch.long) if artifacts.kg_triples else torch.empty((0, 3), dtype=torch.long)
        self.history: list[dict] = []
        self.popularity_ready = False
        self.training_stage = None
        self.freeze_kg_in_joint = False
        self.task_label_confidence = torch.ones(3, device=self.device)
        self.gradient_conflict = torch.zeros(3, device=self.device)
        self.previous_validation_ce = None

    def _training_state(self, epoch, best, stale, mode, warmup_epochs):
        state = {
            "format_version": 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "best_mean_ndcg10": best,
            "stale_epochs": stale,
            "mode": mode,
            "warmup_epochs": warmup_epochs,
            "history": self.history,
            "popularity_ready": self.popularity_ready,
            "task_label_confidence": self.task_label_confidence.detach().cpu(),
            "gradient_conflict": self.gradient_conflict.detach().cpu(),
            "previous_validation_ce": (
                None if self.previous_validation_ce is None
                else self.previous_validation_ce.detach().cpu()
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _atomic_save(state, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def _resolve_resume_path(self, resume_from):
        path = Path(resume_from)
        return path if path.is_absolute() else self.config.output_path / path

    def _load_training_state(self, resume_from, expected_mode, warmup_epochs):
        path = self._resolve_resume_path(resume_from)
        if not path.is_file():
            raise FileNotFoundError(f"训练断点不存在：{path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        required = {"model_state", "optimizer_state", "scheduler_state", "epoch"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"{path} 不是可续训断点，缺少字段：{', '.join(missing)}。"
                "旧版 best_model.pth 只能推理，不能恢复优化器状态。"
            )
        saved_mode = state.get("mode", expected_mode)
        if saved_mode != expected_mode:
            raise ValueError(f"断点训练模式为 {saved_mode!r}，当前模式为 {expected_mode!r}")
        saved_warmup = state.get("warmup_epochs", warmup_epochs)
        if saved_warmup != warmup_epochs:
            raise ValueError(
                f"断点预热轮数为 {saved_warmup}，当前配置计算为 {warmup_epochs}；"
                "请保持总 epochs/预热配置一致。"
            )
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        self.history = list(state.get("history", []))
        self.popularity_ready = bool(state.get("popularity_ready", True))
        self.task_label_confidence = state.get(
            "task_label_confidence", self.task_label_confidence).to(self.device)
        self.gradient_conflict = state.get(
            "gradient_conflict", self.gradient_conflict).to(self.device)
        previous_ce = state.get("previous_validation_ce")
        self.previous_validation_ce = None if previous_ce is None else previous_ce.to(self.device)
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state.get("best_mean_ndcg10", -1.0))
        stale = int(state.get("stale_epochs", 0))
        self.logger.info(
            "恢复训练断点：%s | 已完成 epoch %d | best mean_NDCG@10 %.4f | stale %d",
            path, start_epoch - 1, best, stale,
        )
        return start_epoch, best, stale

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16,
                              enabled=self.amp_enabled)

    @staticmethod
    def set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def set_training_stage(self, stage):
        if stage == self.training_stage:
            return
        self.training_stage = stage
        if stage == "representation":
            self.set_requires_grad(self.model.ple, False)
            self.set_requires_grad(self.model.transe, True)
        else:
            self.set_requires_grad(self.model.ple, True)
            freeze_kg = self.config.freeze_kg_after_warmup and self.freeze_kg_in_joint
            self.set_requires_grad(self.model.transe, not freeze_kg)
        self.logger.info("training stage: %s", stage)

    @torch.no_grad()
    def update_task_weights(self, validation_metrics):
        """Adapt task priors from validation NDCG with EMA and hard bounds."""
        if not self.config.adaptive_task_weights:
            return
        difficulty = torch.tensor([
            1.0 - validation_metrics[f"{task}_ndcg@10"] for task in TASKS
        ], device=self.device).clamp_min(0.05)
        current_ce = torch.tensor([
            validation_metrics[f"{task}_normalized_ce"] for task in TASKS
        ], device=self.device)
        if self.previous_validation_ce is None:
            progress_factor = torch.ones_like(current_ce)
        else:
            progress_factor = (current_ce / self.previous_validation_ce.clamp_min(1e-6)).clamp(0.5, 1.5)
        conflict_factor = torch.exp(
            -self.config.gradient_conflict_strength * self.gradient_conflict)
        priority = (difficulty * progress_factor * self.task_label_confidence *
                    conflict_factor).clamp_min(1e-6)
        proposal = torch.softmax(
            torch.log(priority) / self.config.task_weight_temperature, dim=0)
        proposal = proposal.clamp(self.config.task_weight_min, self.config.task_weight_max)
        proposal = proposal / proposal.sum()
        updated = (self.config.task_weight_ema * self.model.task_weights +
                   (1.0 - self.config.task_weight_ema) * proposal)
        updated = updated.clamp(self.config.task_weight_min, self.config.task_weight_max)
        self.model.task_weights.copy_(updated / updated.sum())
        self.previous_validation_ce = current_ce
        self.logger.info("task weights: knowledge=%.3f course=%.3f consume=%.3f",
                         *self.model.task_weights.tolist())

    def configure_popularity(self, train_loader):
        """只使用训练用户目标统计热门度，验证/测试目标不参与。"""
        v = self.artifacts.vocabs
        counts = {
            "knowledge": torch.ones(len(v["concept"]), device=self.device),
            "course": torch.ones(len(v["course"]), device=self.device),
            "consume": torch.ones(len(v["consume"]), device=self.device),
        }
        for sample in train_loader.dataset.samples:
            sample_weight = float(sample.sample_weight)
            if sample.knowledge_valid:
                for item in sample.concepts:
                    counts["knowledge"][v["concept"].encode(item)] += sample_weight
                counts["knowledge"][v["concept"].encode(sample.target_concept)] += sample_weight
            if sample.course_valid:
                for item in sample.course_history:
                    counts["course"][v["course"].encode(item)] += sample_weight
                counts["course"][v["course"].encode(sample.target_course)] += sample_weight
            for item in sample.consume_items:
                token = item[len("CONSUME::"):] if item.startswith("CONSUME::") else item
                counts["consume"][v["consume"].encode(token)] += sample_weight
            if sample.consume_valid:
                counts["consume"][v["consume"].encode(sample.target_consume)] += sample_weight
        self.model.popularity_knowledge.copy_(counts["knowledge"])
        self.model.popularity_course.copy_(counts["course"])
        self.model.popularity_consume.copy_(counts["consume"])
        self.popularity_ready = True

    def configure_task_confidence(self, train_loader):
        weighted_confidence, total_weight = 0.0, 0.0
        for sample in train_loader.dataset.samples:
            if not sample.knowledge_valid:
                continue
            if sample.target_knowledge_source == "direct_concept_problem":
                confidence = self.config.direct_label_confidence
            elif sample.target_knowledge_source == "fallback_problem_exercise_course_concept":
                confidence = self.config.resolved_label_confidence
            else:
                confidence = self.config.weak_label_confidence
            weighted_confidence += confidence * float(sample.sample_weight)
            total_weight += float(sample.sample_weight)
        knowledge_confidence = weighted_confidence / max(total_weight, 1e-8)
        self.task_label_confidence = torch.tensor(
            [knowledge_confidence, 1.0, 1.0], device=self.device)

    def update_gradient_conflict(self, task_losses):
        if self.training_stage != "joint":
            return
        parameters = [p for p in self.model.ple.levels[0].shared_experts.parameters()
                      if p.requires_grad]
        gradients = {}
        for task in TASKS:
            grads = torch.autograd.grad(
                task_losses[task], parameters, retain_graph=True, allow_unused=True)
            gradients[task] = grads
        conflict = torch.zeros(3, device=self.device)
        for i, left in enumerate(TASKS):
            values = []
            for j, right in enumerate(TASKS):
                if i == j:
                    continue
                zero = task_losses[left].new_zeros(())
                dot = sum(((a.float()*b.float()).sum() for a, b in zip(gradients[left], gradients[right])
                           if a is not None and b is not None), zero)
                norm_left = sum((a.float().pow(2).sum() for a in gradients[left] if a is not None), zero).sqrt()
                norm_right = sum((b.float().pow(2).sum() for b in gradients[right] if b is not None), zero).sqrt()
                if norm_left > 0 and norm_right > 0:
                    values.append(torch.relu(-dot / (norm_left*norm_right + 1e-8)))
            if values:
                conflict[i] = torch.stack(values).mean()
        self.gradient_conflict.mul_(0.8).add_(0.2 * conflict.detach())

    def apply_modality_dropout(self, batch):
        """模拟零行为、仅学习、仅行为用户，使画像与跨域映射具备冷启动能力。"""
        if not self.model.training:
            return batch
        size = batch["user_id"].size(0)
        draw = torch.rand(size, device=self.device)
        profile_only = draw < self.config.dropout_profile_only_prob
        no_behavior = profile_only | ((draw >= self.config.dropout_profile_only_prob) &
            (draw < self.config.dropout_profile_only_prob + self.config.dropout_behavior_prob))
        start = self.config.dropout_profile_only_prob + self.config.dropout_behavior_prob
        no_learning = profile_only | ((draw >= start) &
            (draw < start + self.config.dropout_learning_prob))
        batch["learning_mask"] = batch["learning_mask"] & ~no_learning.unsqueeze(1)
        batch["consume_mask"] = batch["consume_mask"] & ~no_behavior.unsqueeze(1)
        batch["door_mask"] = batch["door_mask"] & ~no_behavior.unsqueeze(1)
        batch["modality_behavior_available"] = ~no_behavior
        return batch

    @torch.no_grad()
    def update_behavior_base(self, loader):
        self.model.eval()
        total = torch.zeros_like(self.model.global_behavior_base)
        count = 0
        for batch in loader:
            batch = move_batch(batch, self.device)
            batch = self.apply_modality_dropout(batch)
            with self.autocast():
                z = self.model.encode_behavior(batch)
            valid = batch["alignment_valid"]
            if valid.any():
                total += z[valid].sum(0)
                count += int(valid.sum())
        self.model.global_behavior_base.copy_(total / max(count, 1))

    def sample_kg(self, size=512):
        if self.kg.numel() == 0:
            empty = torch.empty((0, 3), dtype=torch.long, device=self.device)
            return empty, empty
        index = torch.randint(0, len(self.kg), (min(size, len(self.kg)),))
        positive = self.kg[index].to(self.device)
        negative = positive.clone()
        negative[:, 2] = torch.randint(2, self.model.transe.entity.num_embeddings, (len(negative),), device=self.device)
        return positive, negative

    def batch_loss(self, outputs, batch):
        ple, task_losses = recommendation_loss(outputs, batch, self.config)
        kt = kt_loss(outputs, batch)
        personal = personalization_loss(outputs, batch, self.config)
        retrieval = retrieval_loss(outputs, batch)
        drift = drift_smoothing_loss(outputs, batch)
        interest_diversity, interest_balance = interest_regularization(outputs, batch)
        positive, negative = self.sample_kg()
        kg = self.model.transe.loss(positive, negative)
        if self.training_stage == "representation":
            total = (self.config.representation_retrieval_weight * retrieval +
                     self.config.kt_weight * kt + self.config.transe_weight * kg +
                     self.config.personalization_weight * personal +
                     self.config.drift_smoothing_weight * drift)
        else:
            kg_weight = 0.0 if (self.config.freeze_kg_after_warmup and self.freeze_kg_in_joint) else self.config.transe_weight
            total = (self.config.recommendation_weight * ple +
                     self.config.retrieval_weight * retrieval +
                     self.config.kt_weight * kt + kg_weight * kg +
                     self.config.personalization_weight * personal +
                     self.config.drift_smoothing_weight * drift)
        total = (total + self.config.interest_diversity_weight * interest_diversity +
                 self.config.interest_balance_weight * interest_balance)
        details = {"loss": total.item(), "ple": ple.item(), "kt": kt.item(),
                   "retrieval": retrieval.item(), "transe": kg.item(), "personal": personal.item(),
                   "drift": drift.item(), "interest_diversity": interest_diversity.item(),
                   "interest_balance": interest_balance.item(),
                   **{f"loss_{k}": v.item() for k, v in task_losses.items()}}
        return total, details

    def train_epoch(self, loader, mode):
        self.model.train()
        totals = defaultdict(float)
        batches = 0
        for batch in tqdm(loader, desc="train", leave=False):
            batch = move_batch(batch, self.device)
            batch = self.apply_modality_dropout(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self.autocast():
                outputs = self.model(batch, mode=mode)
                if batches == 0 and self.training_stage == "joint":
                    _, conflict_losses = recommendation_loss(outputs, batch, self.config)
                    self.update_gradient_conflict(conflict_losses)
                loss, details = self.batch_loss(outputs, batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, value in details.items(): totals[key] += value
            batches += 1
        return {k: v / max(batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader, mode="full", collect_embeddings=False):
        self.model.eval()
        logits = {t: [] for t in TASKS}; targets = {t: [] for t in TASKS}
        metric_weights = {t: [] for t in TASKS}
        loss_total, batches = 0.0, 0
        embeddings = defaultdict(list)
        retrieval_hits = {t: [] for t in TASKS}
        retrieval_weights = {t: [] for t in TASKS}
        direct_logits, direct_targets = [], []
        direct_weights = []
        resolved_logits, resolved_targets = [], []
        resolved_weights = []
        for batch in loader:
            batch = move_batch(batch, self.device)
            with self.autocast():
                outputs = self.model(batch, mode=mode)
            direct = batch["target_direct_concept"] & batch["knowledge_valid"]
            if direct.any():
                direct_logits.append(outputs["knowledge"][direct].cpu())
                direct_targets.append(batch["target_concept"][direct].cpu())
                direct_weights.append(batch["sample_weight"][direct].cpu())
            resolved = batch["target_resolved_concept"] & batch["knowledge_valid"]
            if resolved.any():
                resolved_logits.append(outputs["knowledge"][resolved].cpu())
                resolved_targets.append(batch["target_concept"][resolved].cpu())
                resolved_weights.append(batch["sample_weight"][resolved].cpu())
            with self.autocast():
                loss, _ = self.batch_loss(outputs, batch)
            loss_total += loss.item(); batches += 1
            target_map = {"knowledge": "target_concept", "course": "target_course", "consume": "target_consume"}
            for task in TASKS:
                valid = batch[{"knowledge": "knowledge_valid", "course": "course_valid",
                               "consume": "consume_valid"}[task]]
                if valid.any():
                    logits[task].append(outputs[task][valid].cpu())
                    targets[task].append(batch[target_map[task]][valid].cpu())
                    metric_weights[task].append(batch["sample_weight"][valid].cpu())
                    retrieval_hits[task].append(outputs["semantic_candidate_indices"][task][valid]
                                                .eq(batch[target_map[task]][valid].unsqueeze(1)).any(1).float().cpu())
                    retrieval_weights[task].append(batch["sample_weight"][valid].cpu())
            if collect_embeddings:
                for key in ("z_stu", "z_behavior", "drift_score", "short_gate",
                            "consume_drift_score", "door_drift_score", "behavior_fusion_gate"):
                    embeddings[key].append(outputs[key].cpu())
                embeddings["subject"].append(batch["subject"].cpu())
                embeddings["user_id"].append(batch["user_id"].cpu())
        metrics = {"loss": loss_total / max(batches, 1)}
        for task in TASKS:
            merged_logits = torch.cat(logits[task])
            merged_targets = torch.cat(targets[task])
            merged_weights = torch.cat(metric_weights[task])
            for name, value in merge_metric_batches(
                    logits[task], targets[task], self.config.topk, metric_weights[task]).items():
                metrics[f"{task}_{name}"] = value
            raw_ce = F.cross_entropy(merged_logits, merged_targets, reduction="none").clamp_max(50.0)
            normalized_ce = raw_ce / np.log(max(merged_logits.size(1), 2))
            metrics[f"{task}_normalized_ce"] = float(
                (normalized_ce * merged_weights).sum() / merged_weights.sum().clamp_min(1e-8))
            hit, weight = torch.cat(retrieval_hits[task]), torch.cat(retrieval_weights[task])
            metrics[f"{task}_retrieval_recall@{self.config.retrieval_topk}"] = \
                (hit*weight).sum().div(weight.sum().clamp_min(1e-8)).item()
        if direct_logits:
            for name, value in merge_metric_batches(
                    direct_logits, direct_targets, self.config.topk, direct_weights).items():
                metrics[f"knowledge_direct_concept_{name}"] = value
            metrics["knowledge_direct_concept_count"] = sum(len(x) for x in direct_targets)
        if resolved_logits:
            for name, value in merge_metric_batches(
                    resolved_logits, resolved_targets, self.config.topk, resolved_weights).items():
                metrics[f"knowledge_resolved_concept_{name}"] = value
            metrics["knowledge_resolved_concept_count"] = sum(len(x) for x in resolved_targets)
        if collect_embeddings:
            metrics["embeddings"] = {k: torch.cat(v).numpy() for k, v in embeddings.items()}
        return metrics

    def fit(self, train_loader, val_loader, mode="full", epochs=None, resume_from=None):
        epochs = epochs or self.config.epochs
        best, stale = -1.0, 0
        warmup_epochs = min(self.config.representation_warmup_epochs, max(epochs - 1, 0))
        self.freeze_kg_in_joint = warmup_epochs > 0
        start_epoch = 1
        if resume_from:
            start_epoch, best, stale = self._load_training_state(
                resume_from, mode, warmup_epochs)
        if not self.popularity_ready:
            self.configure_popularity(train_loader)
        if not resume_from:
            self.configure_task_confidence(train_loader)
        if stale >= self.config.early_stopping_patience:
            self.logger.info("该断点已经满足早停条件，不再继续训练")
            start_epoch = epochs + 1
        elif start_epoch > epochs:
            self.logger.info("断点已完成 %d 轮，目标总轮数为 %d，无需继续训练", start_epoch - 1, epochs)
        for epoch in range(start_epoch, epochs + 1):
            stage = "representation" if epoch <= warmup_epochs else "joint"
            self.set_training_stage(stage)
            start = time.time()
            self.update_behavior_base(train_loader)
            train = self.train_epoch(train_loader, mode)
            val = self.evaluate(val_loader, mode)
            ndcg = np.mean([val[f"{t}_ndcg@10"] for t in TASKS])
            if stage == "joint":
                self.scheduler.step(ndcg)
            row = {"epoch": epoch, "seconds": time.time()-start,
                   **{f"task_weight_{task}": float(self.model.task_weights[i])
                      for i, task in enumerate(TASKS)},
                   **{f"gradient_conflict_{task}": float(self.gradient_conflict[i])
                      for i, task in enumerate(TASKS)},
                   **{f"label_confidence_{task}": float(self.task_label_confidence[i])
                      for i, task in enumerate(TASKS)},
                   **{f"train_{k}": v for k, v in train.items()},
                   **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(row)
            self.logger.info("epoch %03d | %.1fs | train_loss %.4f | val_loss %.4f | mean_NDCG@10 %.4f | lr %.2e",
                             epoch, row["seconds"], train["loss"], val["loss"], ndcg,
                             self.optimizer.param_groups[0]["lr"])
            if stage == "joint" and ndcg > best + 1e-6:
                best, stale = ndcg, 0
                self._atomic_save(
                    {"model_state": self.model.state_dict(), "config": self.config.to_dict(),
                     "epoch": epoch, "best_mean_ndcg10": best},
                    self.checkpoint,
                )
            elif stage == "joint":
                stale += 1
                if stale >= self.config.early_stopping_patience:
                    self._atomic_save(
                        self._training_state(epoch, best, stale, mode, warmup_epochs),
                        self.last_checkpoint,
                    )
                    self.logger.info("早停：连续 %d 轮验证集未提升", stale)
                    break
            if stage == "joint":
                self.update_task_weights(val)
            self._atomic_save(
                self._training_state(epoch, best, stale, mode, warmup_epochs),
                self.last_checkpoint,
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"最佳模型检查点不存在：{self.checkpoint}")
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        return self.history


def run_ablations(artifacts, loaders, config, logger, full_metrics):
    results = {}
    modes = (("仅全局行为均值", "global_mean"), ("无跨域映射", "no_cross"))
    for label, mode in modes:
        logger.info("开始消融重训练：%s", label)
        model = SmartCampusRecommender(artifacts, config)
        trainer = Trainer(model, artifacts, config, logger, checkpoint_name=f"result/{mode}_best.pth")
        trainer.fit(loaders["train"], loaders["val"], mode=mode, epochs=config.ablation_epochs)
        results[label] = trainer.evaluate(loaders["test"], mode=mode)
    results["完整模型"] = full_metrics
    return results
