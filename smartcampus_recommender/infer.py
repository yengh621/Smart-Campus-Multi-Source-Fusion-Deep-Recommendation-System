"""加载 best_model.pth，对任意 global_user_id 单独推理。"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields

import torch

from config import ExperimentConfig
from data.preprocessing import prepare_data
from inference.recommender import recommend_user
from models.full_model import SmartCampusRecommender
from utils.common import setup_logger
from visualization.plots import plot_recommendation_example, setup_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--data-root", default="../my_output")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = __import__("pathlib").Path(args.output_root).resolve() / args.checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    allowed = {field.name for field in fields(ExperimentConfig)}
    saved = {k: v for k, v in state.get("config", {}).items() if k in allowed}
    saved["data_root"] = args.data_root; saved["output_root"] = args.output_root
    if "topk" in saved: saved["topk"] = tuple(saved["topk"])
    config = ExperimentConfig(**saved)
    config.make_dirs(); logger = setup_logger(config.output_path)
    artifacts = prepare_data(config, logger)
    model = SmartCampusRecommender(artifacts, config).to(device)
    model.load_state_dict(state["model_state"])
    sample = next((x for x in artifacts.samples
                   if x.user_id == args.user_id and x.knowledge_valid), None)
    if sample is None: raise SystemExit(f"找不到有效用户 {args.user_id}")
    result = recommend_user(model, sample, artifacts, device, args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    setup_style(); plot_recommendation_example(result, config.output_path / "fig")


if __name__ == "__main__":
    main()
