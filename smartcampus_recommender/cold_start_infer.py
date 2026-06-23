"""零行为新用户冷启动推荐入口。"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import torch

os.chdir(Path(__file__).resolve().parent)

from config import ExperimentConfig
from data.preprocessing import prepare_data
from inference.recommender import recommend_cold_start
from models.full_model import SmartCampusRecommender
from utils.common import setup_logger


def main():
    parser = argparse.ArgumentParser(description="仅根据画像为零行为新用户推荐")
    parser.add_argument("--subject", default="其他")
    parser.add_argument("--gender", default="未知")
    parser.add_argument("--grade", default="未知")
    parser.add_argument("--data-root", default="../my_output")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(Path(args.output_root).resolve()/args.checkpoint,
                       map_location=device, weights_only=False)
    allowed = {field.name for field in fields(ExperimentConfig)}
    saved = {k: v for k, v in state.get("config", {}).items() if k in allowed}
    saved.update(data_root=args.data_root, output_root=args.output_root)
    if "topk" in saved: saved["topk"] = tuple(saved["topk"])
    config = ExperimentConfig(**saved); config.make_dirs(); logger = setup_logger(config.output_path)
    artifacts = prepare_data(config, logger)
    model = SmartCampusRecommender(artifacts, config).to(device)
    model.load_state_dict(state["model_state"])
    result = recommend_cold_start(model, artifacts, device, args.subject, args.gender, args.grade,
                                  args.topk, config.cold_start_popularity_mix)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (config.output_path/"result"/"cold_start_example.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
