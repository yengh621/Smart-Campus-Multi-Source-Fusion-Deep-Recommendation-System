from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any


class CoreRecommendationAdapter:
    """对现有模型的薄适配；不复制、不改写任何推荐排序逻辑。"""

    def __init__(self, project_root: str | Path, data_root: str | Path,
                 checkpoint: str | Path, topk: int = 10):
        self.project_root = Path(project_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.topk = topk
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        core = self.project_root / "smartcampus_recommender"
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少核心推荐器依赖 torch；请先运行 "
                "pip install -r smartcampus_recommender/requirements.txt") from exc
        from config import ExperimentConfig
        from data.preprocessing import prepare_data
        from models.full_model import SmartCampusRecommender
        from utils.common import setup_logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        allowed = {field.name for field in fields(ExperimentConfig)}
        saved = {key: value for key, value in state.get("config", {}).items() if key in allowed}
        saved.update(data_root=str(self.data_root), output_root=str(self.project_root))
        if "topk" in saved:
            saved["topk"] = tuple(saved["topk"])
        self.config = ExperimentConfig(**saved)
        self.config.make_dirs()
        self.artifacts = prepare_data(self.config, setup_logger(self.project_root))
        self.model = SmartCampusRecommender(self.artifacts, self.config).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self._loaded = True

    def list_user_ids(self) -> list[int]:
        self.load()
        return sorted({int(row.user_id) for row in self.artifacts.samples if row.knowledge_valid})

    def get_user_sample(self, user_id: int) -> Any:
        self.load()
        candidates = [row for row in self.artifacts.samples
                      if int(row.user_id) == int(user_id) and row.knowledge_valid]
        if not candidates:
            raise ValueError(f"找不到可推理的用户 ID: {user_id}")
        # 主样本使用全量近期学习历史及无消费目标泄漏的行为窗口。
        return max(candidates, key=lambda row: (len(row.questions), len(row.consume_items),
                                                len(row.door_items)))

    def recommend(self, sample: Any) -> dict[str, Any]:
        self.load()
        from inference.recommender import recommend_user
        return recommend_user(self.model, sample, self.artifacts, self.device, self.topk)
