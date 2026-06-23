from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


TASK_LABELS = {"knowledge": "知识点", "course": "课程", "consume": "消费品类"}


def setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    })


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_loss(history, fig_dir):
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(df.epoch, df.train_loss, marker="o", ms=3, label="训练损失")
    ax.plot(df.epoch, df.val_loss, marker="s", ms=3, label="验证损失")
    ax.set(xlabel="Epoch", ylabel="联合损失", title="训练与验证损失变化")
    ax.legend()
    save(fig, fig_dir / "01_loss_curve.png")


def plot_metrics(history, fig_dir):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, metric, title in zip(axes, ("ndcg@5", "ndcg@10", "auc"), ("NDCG@5", "NDCG@10", "AUC")):
        for task, label in TASK_LABELS.items():
            ax.plot(df.epoch, df[f"val_{task}_{metric}"], marker="o", ms=4, label=label)
        ax.set(xlabel="Epoch", ylabel=title, title=f"验证集{title}变化")
        ax.set_ylim(0, 1.02)
    axes[0].legend()
    save(fig, fig_dir / "02_validation_metrics.png")


def plot_ablation(ablation, fig_dir):
    names, values = [], []
    for name, metrics in ablation.items():
        names.append(name)
        values.append(np.mean([metrics[f"{t}_ndcg@10"] for t in TASK_LABELS]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(names, values, color=["#94A3B8", "#F59E0B", "#2563EB"][:len(names)])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylabel="三任务平均 NDCG@10", title="个体跨域映射消融实验")
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    save(fig, fig_dir / "03_ablation_ndcg10.png")


def plot_subject_distribution(subject_counts, fig_dir):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    labels, values = list(subject_counts), list(subject_counts.values())
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           wedgeprops={"linewidth": 0.8, "edgecolor": "white"})
    ax.set_title("融合数据集学科样本分布")
    save(fig, fig_dir / "04_subject_distribution.png")


def plot_tsne(embeddings, subject_vocab, fig_dir, seed):
    z1, z2, subjects = embeddings["z_stu"], embeddings["z_behavior"], embeddings["subject"]
    combined = np.vstack([z1, z2])
    perplexity = min(30, max(5, len(z1) // 10))
    points = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto",
                  perplexity=perplexity, n_jobs=1).fit_transform(combined)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    palette = sns.color_palette("tab10", n_colors=max(len(subject_vocab)-2, 1))
    for ax, coords, title in ((axes[0], points[:len(z1)], "学业表征 $z_{stu}$"),
                              (axes[1], points[len(z1):], "映射行为表征 $z_{behavior}$")):
        for sid in sorted(set(subjects.tolist())):
            mask = subjects == sid
            label = subject_vocab.decode(int(sid))
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=.7,
                       label=label, color=palette[(int(sid)-2) % len(palette)])
        ax.set_title(title); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(fig, fig_dir / "05_tsne_embeddings.png")


def plot_recommendation_example(example, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for ax, task in zip(axes, TASK_LABELS):
        rows = example[task]
        names = [x["item"][-24:] for x in rows][::-1]
        scores = [x["score"] for x in rows][::-1]
        ax.barh(names, scores, color="#2563EB")
        ax.set(title=f"{TASK_LABELS[task]} Top-{len(rows)}", xlabel="推荐概率")
    fig.suptitle(f"测试用户 {example['user_id']} 的推荐示例", y=1.02)
    save(fig, fig_dir / "06_top_recommendation_example.png")


def plot_interest_drift(embeddings, subject_vocab, fig_dir):
    drift = embeddings["consume_drift_score"]
    door_drift = embeddings["door_drift_score"]
    gate = embeddings["short_gate"]
    subjects = embeddings["subject"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.histplot(drift, bins=20, kde=True, ax=axes[0], color="#7C3AED", label="消费")
    sns.histplot(door_drift, bins=20, kde=True, ax=axes[0], color="#0EA5E9", label="门禁", alpha=.45)
    axes[0].set(xlabel="兴趣/状态漂移分数", ylabel="用户数", title="消费与门禁漂移分布")
    axes[0].legend()
    rows = []
    for sid, value, alpha in zip(subjects, drift, gate):
        rows.append({"subject": subject_vocab.decode(int(sid)), "drift": value, "gate": alpha})
    frame = pd.DataFrame(rows)
    sns.scatterplot(data=frame, x="drift", y="gate", hue="subject", s=35, alpha=.8, ax=axes[1])
    axes[1].set(xlabel="兴趣漂移分数", ylabel="短期兴趣门控权重 α",
                title="漂移程度与短期兴趣权重")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    save(fig, fig_dir / "07_interest_drift.png")


def plot_diversity_metrics(metrics, fig_dir):
    names = ("coverage", "gini", "intra_list_diversity", "inter_list_diversity")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, name in zip(axes.flat, names):
        values = [metrics.get(f"{task}_{name}", 0) for task in TASK_LABELS]
        bars = ax.bar(list(TASK_LABELS.values()), values, color=["#2563EB", "#F59E0B", "#10B981"])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set(title=name.replace("_", " ").title(), ylim=(0, max(max(values)*1.2, 1e-3)))
    save(fig, fig_dir / "08_diversity_debias.png")


def plot_formula_parameters(parameters, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    task_weights = parameters.get("task_normalized_weight", {})
    bars = axes[0].bar(list(task_weights), list(task_weights.values()),
                       color=["#2563EB", "#F59E0B", "#10B981"])
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set(title="不确定性学习后的任务权重", ylabel="归一化权重", ylim=(0, 1))
    layers = parameters.get("time_decay_layers", [])
    if layers:
        first = layers[0]
        for head, (rates, mixtures) in enumerate(zip(first["rates"], first["mixtures"]), 1):
            axes[1].plot(rates, mixtures, marker="o", label=f"Head {head}")
    axes[1].set(title="第一层多尺度时间核", xlabel="衰减率 λ", ylabel="混合权重 α")
    axes[1].legend(fontsize=7)
    save(fig, fig_dir / "09_learned_formula_parameters.png")


def generate_all_figures(history, ablation, subject_counts, embeddings, subject_vocab,
                         recommendation_example, diversity_metrics, formula_parameters,
                         fig_dir: Path, seed: int):
    setup_style(); fig_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(history, fig_dir)
    plot_metrics(history, fig_dir)
    plot_ablation(ablation, fig_dir)
    plot_subject_distribution(subject_counts, fig_dir)
    plot_tsne(embeddings, subject_vocab, fig_dir, seed)
    plot_recommendation_example(recommendation_example, fig_dir)
    plot_interest_drift(embeddings, subject_vocab, fig_dir)
    plot_diversity_metrics(diversity_metrics, fig_dir)
    plot_formula_parameters(formula_parameters, fig_dir)
