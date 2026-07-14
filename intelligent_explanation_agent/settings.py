from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_client import OpenAICompatibleClient
from .recommendation_adapter import CoreRecommendationAdapter


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("config.local.json")


def _load_config() -> dict[str, Any]:
    path = Path(os.getenv("INTELLIGENT_AGENT_CONFIG", _default_config_path())).resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件格式错误：{path}，{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 JSON 对象：{path}")
    return payload


def _env_or_config(env_name: str, config_value: Any, default: Any = None) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    if config_value is not None:
        return config_value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass
class Settings:
    project_root: Path
    data_root: Path
    checkpoint: Path
    topk: int = 10
    recent_limit: int = 10
    include_user_id_in_api: bool = False
    glm_api_key: str = ""
    glm_api_base: str = "https://api.deepseek.com"
    glm_model: str = "deepseek-v4-flash"
    glm_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        config = _load_config()
        glm = config.get("glm", {})
        if not isinstance(glm, dict):
            raise ValueError("配置项 glm 必须是 JSON 对象")

        root = Path(
            _env_or_config("PROJECT_ROOT", config.get("project_root"), Path(__file__).resolve().parents[1])
        ).resolve()
        return cls(
            project_root=root,
            data_root=_resolve_path(_env_or_config("DATA_ROOT", config.get("data_root"), "my_output"), root),
            checkpoint=_resolve_path(
                _env_or_config("CHECKPOINT_PATH", config.get("checkpoint_path"), "best_model.pth"),
                root,
            ),
            topk=int(_env_or_config("RECOMMEND_TOPK", config.get("recommend_topk"), 10)),
            recent_limit=int(
                _env_or_config("RECENT_EVENTS_PER_MODALITY", config.get("recent_events_per_modality"), 10)
            ),
            include_user_id_in_api=_as_bool(
                _env_or_config("INCLUDE_USER_ID_IN_API", config.get("include_user_id_in_api"), False)
            ),
            glm_api_key=str(_env_or_config(
                "DEEPSEEK_API_KEY",
                _env_or_config("GLM_API_KEY", glm.get("api_key"), ""),
            )).strip(),
            glm_api_base=str(
                _env_or_config(
                    "DEEPSEEK_API_BASE",
                    _env_or_config("GLM_API_BASE", glm.get("api_base"), "https://api.deepseek.com"),
                )
            ).strip(),
            glm_model=str(_env_or_config(
                "DEEPSEEK_MODEL",
                _env_or_config("GLM_MODEL", glm.get("model"), "deepseek-v4-flash"),
            )).strip(),
            glm_timeout_seconds=float(_env_or_config(
                "DEEPSEEK_TIMEOUT_SECONDS",
                _env_or_config("GLM_TIMEOUT_SECONDS", glm.get("timeout_seconds"), 60),
            )),
        )

    def make_recommender(self) -> CoreRecommendationAdapter:
        return CoreRecommendationAdapter(self.project_root, self.data_root, self.checkpoint, self.topk)

    def make_llm(self) -> OpenAICompatibleClient | None:
        key = self.glm_api_key or os.getenv("EXPLAIN_API_KEY", "").strip()
        if not key:
            return None
        return OpenAICompatibleClient(
            api_key=key,
            model=self.glm_model or os.getenv("EXPLAIN_MODEL", "deepseek-v4-flash"),
            base_url=self.glm_api_base or os.getenv("EXPLAIN_API_BASE", "https://api.deepseek.com"),
            timeout_seconds=self.glm_timeout_seconds,
        )
