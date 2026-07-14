"""智能推荐解释层：复用核心推荐器，并由兼容 API 生成用户可读解释。"""

from .agent import ExplainableRecommendationAgent

__all__ = ["ExplainableRecommendationAgent"]
