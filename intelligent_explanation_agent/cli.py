from __future__ import annotations

import argparse
import json

from .agent import ExplainableRecommendationAgent
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="三模态可解释推荐智能体")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--require-explanation", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    agent = ExplainableRecommendationAgent(
        settings.make_recommender(), settings.make_llm(), settings.recent_limit,
        settings.include_user_id_in_api)
    print(json.dumps(agent.run(args.user_id, args.require_explanation),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
