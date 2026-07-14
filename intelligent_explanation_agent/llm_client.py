from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleClient:
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 60.0
    temperature: float = 0.2

    def explain(self, snapshot: dict[str, Any], recommendations: dict[str, Any]) -> dict[str, Any]:
        system = (
            "你是校园推荐系统的用户端解释智能体。推荐排序已经由核心推荐网络给出，"
            "你不能新增、删除、调换或修改推荐项与分数。你的任务不是逐条解释每个推荐项，"
            "而是基于近期学习、消费、门禁三模态快照和推荐结果，生成一段综合性、用户容易接受的解释。\n"
            "解释要自然、有说服力，可以从兴趣较高、近期行为相似、知识掌握薄弱、需要巩固、"
            "课程方向匹配、生活消费习惯相近等角度说明，但不能把相关性说成因果，也不能推断敏感属性。\n"
            "不要输出“当前数据不足，暂无法形成可靠解释”这类兜底句。证据较弱时，请用温和表达，"
            "例如“从已有近期行为看，系统更倾向于认为……”“这些推荐更像是用于探索和巩固”。\n"
            "输出必须是单个 JSON 对象，字段为："
            "user_summary 字符串；"
            "overall_reason 字符串；"
            "learning_reason 字符串；"
            "course_reason 字符串；"
            "consumption_reason 字符串；"
            "modality_insights 对象，包含 learning/consumption/access 三个字符串；"
            "acceptance_note 字符串；"
            "caveats 字符串数组。"
        )
        user = json.dumps(
            {
                "recent_multimodal_snapshot": snapshot,
                "model_recommendations": recommendations,
                "instruction": (
                    "请只做综合解释，不要为每个 item 单独生成 explanation。"
                    "请结合推荐列表整体说明为什么这些内容可能适合该学生。"
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise LLMError(f"解释 API 调用或响应解析失败：{exc}") from exc
        if not isinstance(result, dict):
            raise LLMError("解释 API 没有返回 JSON 对象")
        return result
