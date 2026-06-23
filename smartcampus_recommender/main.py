from __future__ import annotations

import argparse
import copy
import json
import os
import random
import warnings
from collections import defaultdict
import torch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

from config import ExperimentConfig
from data.dataset import build_dataloaders
from data.preprocessing import prepare_data
from inference.recommender import export_test_recommendations, recommend_user
from models.full_model import SmartCampusRecommender
from reporting.excel_report import write_metric_workbook
from reporting.thesis_text import write_thesis_text
from training.trainer import Trainer, run_ablations
from utils.common import seed_everything, setup_logger
from visualization.plots import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description="智慧校园多源融合深度推荐系统")
    parser.add_argument("--data-root", default="../my_output")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--quick", action="store_true", help="小样本1轮端到端冒烟测试")
    parser.add_argument("--skip-ablations", action="store_true", help="仅调试时使用；正式论文实验不要跳过")
    return parser.parse_args()


def balanced_quick_artifacts(artifacts, per_subject=24):
    grouped = defaultdict(lambda: defaultdict(list))
    for sample in artifacts.samples: grouped[sample.subject][sample.split_group].append(sample)
    rng = random.Random(2026)
    quick = copy.copy(artifacts)
    quick.samples = []
    for subject_groups in grouped.values():
        keys = list(subject_groups); rng.shuffle(keys)
        for key in keys[:per_subject]:
            quick.samples.extend(subject_groups[key])
    quick.subject_counts = dict(__import__("collections").Counter(
        x.subject for x in quick.samples if x.knowledge_valid))
    return quick


def clean_metrics(metrics):
    return {k: float(v) for k, v in metrics.items() if k != "embeddings"}


def collect_formula_parameters(model):
    normalized = model.task_weights.detach().cpu()
    result = {
        "task_weight_strategy": "validation_difficulty_x_loss_progress_x_label_confidence_x_gradient_conflict",
        "task_normalized_weight": dict(zip(("knowledge", "course", "consume"), normalized.tolist())),
        "course_score_fusion_weight": dict(zip(
            ("retrieval", "ple", "logic"),
            torch.softmax(model.course_fusion_logits, 0).detach().cpu().tolist())),
        "course_score_temperature": dict(zip(
            ("retrieval", "ple", "logic"),
            (torch.nn.functional.softplus(model.course_log_temperatures) +
             model.course_score_temperature_floor).detach().cpu().tolist())),
        "time_decay_layers": [],
    }
    for index, layer in enumerate(model.akt.layers):
        result["time_decay_layers"].append({
            "layer": index + 1,
            "rates": torch.nn.functional.softplus(layer.decay_rates).detach().cpu().tolist(),
            "mixtures": torch.softmax(layer.mixture_logits, -1).detach().cpu().tolist(),
        })
    return result


def main():
    args = parse_args()
    config = ExperimentConfig(data_root=args.data_root, output_root=args.output_root)
    if args.epochs: config.epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.quick:
        config.epochs = 1; config.ablation_epochs = 1; config.early_stopping_patience = 1
        # 快速模式只减少用户与轮次，不改变模型结构，保证检查点可被正式推理加载。
        config.batch_size = 32
    config.make_dirs(); seed_everything(config.seed); logger = setup_logger(config.output_path)
    logger.info("设备与实验初始化完成")
    artifacts = prepare_data(config, logger, force_rebuild=args.force_preprocess)
    if args.quick: artifacts = balanced_quick_artifacts(artifacts)
    loaders, splits = build_dataloaders(artifacts, config, logger)

    model = SmartCampusRecommender(artifacts, config)
    trainer = Trainer(model, artifacts, config, logger)
    history = trainer.fit(loaders["train"], loaders["val"])
    # 测试集在完整模型选择结束后仅评估一次。
    test_with_embeddings = trainer.evaluate(loaders["test"], collect_embeddings=True)
    embeddings = test_with_embeddings.pop("embeddings")
    test_metrics = clean_metrics(test_with_embeddings)
    logger.info("完整模型测试指标：%s", json.dumps(test_metrics, ensure_ascii=False))

    if args.skip_ablations:
        ablation = {"完整模型": test_metrics}
    else:
        ablation = run_ablations(artifacts, loaders, config, logger, test_metrics)

    primary_test = [x for x in splits["test"] if x.knowledge_valid]
    example_sample = random.Random(config.seed).choice(primary_test)
    example = recommend_user(model, example_sample, artifacts, trainer.device, topk=10)
    diversity_metrics = export_test_recommendations(
        model, loaders["test"], artifacts, trainer.device,
        config.output_path / "result" / "test_topk_recommendations.csv", topk=10)
    test_metrics.update(diversity_metrics)
    formula_parameters = collect_formula_parameters(model)
    (config.output_path / "result" / "recommendation_example.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_path / "result" / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_path / "result" / "ablation_metrics.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_path / "result" / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_path / "result" / "learned_formula_parameters.json").write_text(
        json.dumps(formula_parameters, ensure_ascii=False, indent=2), encoding="utf-8")

    generate_all_figures(history, ablation, artifacts.subject_counts, embeddings,
                         artifacts.vocabs["subject"], example, diversity_metrics,
                         formula_parameters, config.output_path / "fig", config.seed)
    write_metric_workbook(config.output_path / "result" / "metric_result.xlsx",
                          test_metrics, ablation, config, history, artifacts.knowledge_mapping_stats,
                          formula_parameters)
    write_thesis_text(config.output_path / "result" / "thesis_ready_text.md",
                      config, artifacts, test_metrics, ablation)
    logger.info("全部完成：模型、图、指标、推荐结果和论文文字已自动导出")


if __name__ == "__main__":
    main()
