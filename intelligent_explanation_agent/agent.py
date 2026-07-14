from __future__ import annotations

from typing import Any

from .llm_client import LLMError, OpenAICompatibleClient
from .snapshot import build_snapshot


class ExplainableRecommendationAgent:
    def __init__(
        self,
        recommender: Any,
        llm: OpenAICompatibleClient | None,
        recent_limit: int = 10,
        include_user_id_in_api: bool = False,
    ):
        self.recommender = recommender
        self.llm = llm
        self.recent_limit = recent_limit
        self.include_user_id_in_api = include_user_id_in_api

    def run(self, user_id: int, require_explanation: bool = False) -> dict[str, Any]:
        sample = self.recommender.get_user_sample(user_id)
        snapshot = build_snapshot(sample, self.recent_limit)
        recommendations = self.recommender.recommend(sample)
        result = {
            "user_id": int(user_id),
            "snapshot": snapshot.to_dict(include_user_id=True),
            "recommendations": recommendations,
            "explanation": None,
            "explanation_status": "not_configured",
        }
        if self.llm is None:
            if require_explanation:
                raise LLMError("未配置 DEEPSEEK_API_KEY，无法生成 API 解释")
            return result

        try:
            raw_explanation = self.llm.explain(
                snapshot.to_dict(include_user_id=self.include_user_id_in_api),
                recommendations,
            )
            result["explanation"] = self._clean_explanation(raw_explanation, recommendations)
            result["explanation_status"] = "ok"
        except LLMError as exc:
            result["explanation_status"] = "unavailable"
            result["explanation_error"] = str(exc)
            if require_explanation:
                raise
        return result

    @staticmethod
    def _clean_explanation(explanation: dict[str, Any], recommendations: dict[str, Any]) -> dict[str, Any]:
        """保留综合解释；若 API 仍返回逐条解释，则只校正顺序，不生成兜底废话。"""
        cleaned = dict(explanation)
        supplied = explanation.get("recommendation_explanations")
        if not isinstance(supplied, dict):
            return cleaned

        aligned: dict[str, list[dict[str, Any]]] = {}
        for task in ("knowledge", "course", "consume"):
            rows = supplied.get(task, [])
            by_item = {
                str(row.get("item")): row
                for row in rows
                if isinstance(row, dict) and "item" in row
            }
            aligned[task] = []
            for recommended in recommendations.get(task, []):
                item = str(recommended.get("item"))
                row = by_item.get(item)
                if not row:
                    continue
                aligned[task].append(
                    {
                        "item": item,
                        "reason": str(row.get("reason", "")),
                        "evidence": ExplainableRecommendationAgent._as_text_list(row.get("evidence", [])),
                        "uncertainty": str(row.get("uncertainty", "")),
                    }
                )
        cleaned["recommendation_explanations"] = aligned
        return cleaned

    @staticmethod
    def _as_text_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]
