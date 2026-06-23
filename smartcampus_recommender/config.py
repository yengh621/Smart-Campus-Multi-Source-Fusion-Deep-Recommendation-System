"""集中管理全部实验超参数，论文实验设置以此文件为唯一事实来源。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    data_root: str = "../my_output"
    output_root: str = "."
    seed: int = 2026
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_learning_len: int = 100
    max_behavior_len: int = 100
    min_learning_events: int = 3
    min_consume_history: int = 3
    max_consume_windows_per_user: int = 10
    consume_window_stride: int = 1
    max_course_windows_per_user: int = 5
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    akt_layers: int = 2
    num_shared_experts: int = 2
    num_task_experts: int = 2
    dropout: float = 0.20
    batch_size: int = 64
    epochs: int = 60
    ablation_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 4
    early_stopping_patience: int = 15
    gradient_clip: float = 5.0
    use_amp: bool = True
    num_workers: int = 0
    topk: tuple[int, int] = (5, 10)
    transe_margin: float = 1.0
    transe_weight: float = 0.10
    kt_weight: float = 0.20
    personalization_weight: float = 0.05
    recommendation_weight: float = 1.0
    representation_warmup_epochs: int = 5
    representation_retrieval_weight: float = 1.0
    freeze_kg_after_warmup: bool = True
    task_knowledge_weight: float = 0.35
    task_course_weight: float = 0.40
    task_consume_weight: float = 0.25
    adaptive_task_weights: bool = True
    task_weight_ema: float = 0.80
    task_weight_temperature: float = 0.50
    task_weight_min: float = 0.15
    task_weight_max: float = 0.60
    direct_label_confidence: float = 1.0
    resolved_label_confidence: float = 0.65
    weak_label_confidence: float = 0.35
    gradient_conflict_strength: float = 1.0
    personalization_margin: float = 0.5
    retrieval_topk: int = 100
    retrieval_temperature: float = 0.07
    retrieval_weight: float = 0.30
    course_rule_candidates: int = 30
    course_score_temperature_floor: float = 0.10
    short_interest_window: int = 20
    num_interests: int = 4
    drift_smoothing_weight: float = 0.02
    popularity_penalty: float = 0.0
    inverse_popularity_alpha: float = 0.0
    mmr_diversity_weight: float = 0.25
    rerank_pool_size: int = 50
    interest_quota_per_vector: int = 3
    cold_start_popularity_mix: float = 0.35
    dropout_profile_only_prob: float = 0.10
    dropout_behavior_prob: float = 0.30
    dropout_learning_prob: float = 0.20
    contrastive_temperature: float = 0.10
    info_nce_weight: float = 1.0
    interest_diversity_weight: float = 0.05
    interest_balance_weight: float = 0.02
    vicreg_invariance_weight: float = 5.0
    vicreg_variance_weight: float = 5.0
    vicreg_covariance_weight: float = 1.0
    time_decay_kernels: int = 4
    course_logic_scale: float = 2.0
    course_explicit_mapping_weight: float = 0.45
    course_weakness_weight: float = 0.15
    course_interest_weight: float = 0.15
    course_prerequisite_weight: float = 0.15
    course_difficulty_weight: float = 0.10
    drift_recent_hours: float = 168.0
    drift_history_hours: float = 720.0
    cache_name: str = "processed_cache_v16_shared_consume_embedding.pkl"

    @property
    def data_path(self) -> Path:
        return Path(self.data_root).resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.output_root).resolve()

    def make_dirs(self) -> None:
        for name in ("fig", "result", "logs", "cache"):
            (self.output_path / name).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["topk"] = list(self.topk)
        return data
