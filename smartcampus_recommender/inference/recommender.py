from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import numpy as np

from data.dataset import SmartCampusDataset, collate_batch
from data.preprocessing import UserSample
from training.trainer import move_batch


TASK_VOCAB = {"knowledge": "concept", "course": "course", "consume": "consume"}


def decode_topk(logits, vocab, k=10):
    logits = logits.clone()
    logits[..., :2] = torch.finfo(logits.dtype).min
    probabilities = torch.softmax(logits, dim=-1)
    scores, indices = torch.topk(probabilities, min(k, probabilities.size(-1)), dim=-1)
    return [[{"item": vocab.decode(int(idx)), "score": float(score)}
             for idx, score in zip(row_i, row_s)] for row_i, row_s in zip(indices.cpu(), scores.cpu())]


def task_item_matrix(model, task):
    if task == "knowledge":
        graph = model.transe.entity(model.concept_to_kg)
    elif task == "course":
        graph = model.transe.entity(model.course_to_kg)
    else:
        graph = None
    return model.retriever.candidate_matrix(task, graph)


def diversified_topk(outputs, model, artifacts, row_index: int, task: str, k: int,
                     exclude_indices=None):
    """PLE相关性基础上执行MMR；消费任务额外限制单一兴趣向量的占位数量。"""
    logits = outputs[task][row_index].detach().clone()
    logits[:2] = torch.finfo(logits.dtype).min
    if exclude_indices:
        logits[list(exclude_indices)] = torch.finfo(logits.dtype).min
    pool_size = min(model.rerank_pool_size, logits.numel() - 2)
    pool = torch.topk(logits, max(pool_size, k)).indices
    ranking_embeddings = outputs.get("ranking_item_embeddings", {})
    if task in ranking_embeddings:
        # Use the exact user-conditioned representation consumed by the
        # ranker, so MMR diversity and relevance live in the same space.
        matrix = F.normalize(ranking_embeddings[task][row_index].detach(), dim=-1)
    else:
        matrix = F.normalize(task_item_matrix(model, task).detach(), dim=-1)
    relevance = torch.softmax(logits[pool], dim=0)
    selected, interest_counts = [], {}
    quota = model.interest_quota_per_vector
    interest_assignment = None
    if task == "consume" and "multi_interests" in outputs:
        interests = F.normalize(outputs["multi_interests"][row_index].detach(), dim=-1)
        if "multi_interest_mask" in outputs:
            interests = interests[outputs["multi_interest_mask"][row_index]]
        if interests.numel() > 0:
            interest_assignment = (interests @ matrix[pool].T).argmax(0)
    remaining = list(range(len(pool)))
    while remaining and len(selected) < k:
        best_local, best_value = None, -float("inf")
        for local in remaining:
            if interest_assignment is not None:
                group = int(interest_assignment[local])
                if interest_counts.get(group, 0) >= quota:
                    continue
            similarity = 0.0
            if selected:
                similarity = float((matrix[pool[local]] @ matrix[torch.tensor(selected, device=pool.device)].T).max())
            value = ((1.0 - model.mmr_diversity_weight) * float(relevance[local]) -
                     model.mmr_diversity_weight * similarity)
            if value > best_value:
                best_local, best_value = local, value
        if best_local is None:  # 配额过严时放宽，保证返回完整Top-K。
            quota += 1
            continue
        item_index = int(pool[best_local])
        selected.append(item_index); remaining.remove(best_local)
        if interest_assignment is not None:
            group = int(interest_assignment[best_local]); interest_counts[group] = interest_counts.get(group, 0) + 1
    probabilities = torch.softmax(logits, dim=0)
    vocab = artifacts.vocabs[TASK_VOCAB[task]]
    result = []
    for idx in selected:
        row = {"index": idx, "item": vocab.decode(idx), "score": float(probabilities[idx])}
        if task == "consume" and task in ranking_embeddings:
            row["_ranking_embedding"] = matrix[idx].cpu().tolist()
        if task == "course" and "course_logic_score" in outputs:
            row.update({
                "logic_score": float(outputs["course_logic_score"][row_index, idx]),
                "explicit_mapping_available": bool(outputs["course_explicit_mapping_available"][row_index, idx]),
                "explicit_knowledge_coverage": float(outputs["course_explicit_coverage_score"][row_index, idx]),
                "weakness_match": float(outputs["course_weakness_score"][row_index, idx]),
                "history_interest": float(outputs["course_interest_score"][row_index, idx]),
                "prerequisite_readiness": float(outputs["course_prerequisite_score"][row_index, idx]),
                "difficulty_fit": float(outputs["course_difficulty_score"][row_index, idx]),
            })
        result.append(row)
    return result


def diversity_metrics(all_lists, model, task):
    lists = [x for x in all_lists if x]
    if not lists: return {}
    catalog_size = task_item_matrix(model, task).size(0) - 2
    exposure = np.zeros(max(catalog_size + 2, 2), dtype=np.float64)
    matrix = F.normalize(task_item_matrix(model, task).detach().cpu(), dim=-1)
    ild_values, novelties, popularities = [], [], []
    pop = getattr(model, f"popularity_{task}").detach().cpu().numpy()
    pop_prob = pop / max(pop.sum(), 1.0)
    for items in lists:
        indices = [x["index"] for x in items]
        exposure[indices] += 1
        if items and all("_ranking_embedding" in item for item in items):
            vectors = F.normalize(torch.tensor(
                [item["_ranking_embedding"] for item in items], dtype=torch.float32), dim=-1)
        else:
            vectors = matrix[indices]
        if len(indices) > 1:
            similarity = vectors @ vectors.T
            mask = ~torch.eye(len(indices), dtype=torch.bool)
            ild_values.append(float((1.0 - similarity[mask]).mean()))
        popularities.extend(pop[indices].tolist())
        novelties.extend((-np.log2(np.maximum(pop_prob[indices], 1e-12))).tolist())
    used = exposure[2:]
    sorted_exp = np.sort(used); n = len(sorted_exp)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_exp) / max(n * sorted_exp.sum(), 1e-12)
            - (n + 1) / max(n, 1))
    pair_dist = []
    for i in range(min(len(lists), 200)):
        a, b = set(x["index"] for x in lists[i]), set(x["index"] for x in lists[(i+1) % len(lists)])
        pair_dist.append(1.0 - len(a & b) / max(len(a | b), 1))
    return {f"{task}_coverage": float((used > 0).sum() / max(catalog_size, 1)),
            f"{task}_gini": float(gini),
            f"{task}_novelty": float(np.mean(novelties)),
            f"{task}_average_popularity": float(np.mean(popularities)),
            f"{task}_intra_list_diversity": float(np.mean(ild_values) if ild_values else 0),
            f"{task}_inter_list_diversity": float(np.mean(pair_dist) if pair_dist else 0)}


@torch.no_grad()
def recommend_user(model, sample, artifacts, device, topk=10):
    model.eval()
    row = SmartCampusDataset([sample], artifacts)[0]
    batch = move_batch(collate_batch([row]), device)
    outputs = model(batch)
    result = {"user_id": sample.user_id}
    for task in TASK_VOCAB:
        # 知识/课程避免重复推荐；消费品类允许复购，因此不能把历史品类全部屏蔽。
        history = {"knowledge": sample.concepts, "course": sample.course_history,
                   "consume": []}[task]
        excluded = {artifacts.vocabs[TASK_VOCAB[task]].encode(x) for x in history}
        result[task] = [{k: v for k, v in x.items() if k != "index" and not k.startswith("_")}
                        for x in diversified_topk(outputs, model, artifacts, 0, task, topk, excluded)]
    return result


def make_cold_start_sample(subject="其他", gender="未知", grade="未知"):
    """零行为新用户：仅保留基础画像，所有时序由mask关闭。"""
    return UserSample(
        user_id=-1, split_group="COLD_START", subject=subject, gender=gender, grade=grade,
        questions=[], concepts=[], correct=[], intervals=[], night_study=[], wrong_streak=[],
        relative_position=[], course_questions=[], course_concepts=[], course_correct=[],
        course_intervals=[], course_night_study=[], course_wrong_streak=[],
        course_relative_position=[], consume_items=[], consume_meal=[], consume_weekly=[], consume_hour=[],
        consume_weekday=[], consume_holiday=[], door_items=[], door_late=[], door_weekly=[],
        door_hour=[], door_weekday=[], door_holiday=[], consume_age_hours=[], door_age_hours=[],
        target_concept="<UNK>", target_knowledge_source="cold_start",
        target_course="<UNK>", course_history=[], target_consume="<UNK>", consume_valid=False,
        knowledge_valid=False, course_valid=False, alignment_valid=False,
        context_hour=0, context_weekday=0, context_holiday=0, sample_weight=1.0)


@torch.no_grad()
def recommend_cold_start(model, artifacts, device, subject="其他", gender="未知", grade="未知",
                         topk=10, popularity_mix=0.35):
    model.eval(); sample = make_cold_start_sample(subject, gender, grade)
    row = SmartCampusDataset([sample], artifacts)[0]
    batch = move_batch(collate_batch([row]), device)
    outputs = model(batch)
    # 零历史时适度混入训练集热门先验，但仍经过热门惩罚、MMR与多样性重排。
    for task in TASK_VOCAB:
        pop = outputs["popularity"][task]
        prior = torch.log1p(pop); prior = prior / prior.max().clamp_min(1.0)
        outputs[task] = (1.0 - popularity_mix) * outputs[task] + popularity_mix * prior.unsqueeze(0)
    result = {"user_id": "COLD_START", "subject": subject, "gender": gender, "grade": grade}
    for task in TASK_VOCAB:
        result[task] = [{k: v for k, v in x.items() if k != "index" and not k.startswith("_")}
                        for x in diversified_topk(outputs, model, artifacts, 0, task, topk)]
    return result


@torch.no_grad()
def export_test_recommendations(model, loader, artifacts, device, path: Path, topk=10):
    model.eval(); rows = []; recommendation_lists = {task: [] for task in TASK_VOCAB}
    primary_by_uid = {x.user_id: x for x in loader.dataset.samples if x.knowledge_valid}
    course_sample_by_key = {
        (x.user_id, artifacts.vocabs["course"].encode(x.target_course)): x
        for x in loader.dataset.samples if x.course_valid}
    final_hits = {task: {5: [], 10: []} for task in TASK_VOCAB}
    final_weights = {task: [] for task in TASK_VOCAB}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch)
        for i, uid in enumerate(batch["user_id"].cpu().tolist()):
            for task in TASK_VOCAB:
                valid_key = {"knowledge": "knowledge_valid", "course": "course_valid",
                             "consume": "consume_valid"}[task]
                if not bool(batch[valid_key][i]):
                    continue
                target_key = int(batch["target_course"][i]) if task == "course" else None
                sample = (course_sample_by_key[(uid, target_key)] if task == "course"
                          else primary_by_uid.get(uid, loader.dataset.samples[0]))
                history = {"knowledge": sample.concepts, "course": sample.course_history,
                           "consume": []}[task]
                vocab = artifacts.vocabs[TASK_VOCAB[task]]
                excluded = {vocab.encode(x) for x in history}
                items = diversified_topk(outputs, model, artifacts, i, task, topk, excluded)
                recommendation_lists[task].append(items)
                target = int(batch[f"target_{'concept' if task == 'knowledge' else task}"][i])
                ranked = [x["index"] for x in items]
                final_weights[task].append(float(batch["sample_weight"][i]))
                for cutoff in (5, 10):
                    rank = ranked[:cutoff].index(target) + 1 if target in ranked[:cutoff] else 0
                    final_hits[task][cutoff].append(rank)
                for rank, item in enumerate(items, 1):
                    rows.append({"global_user_id": uid, "task": task, "rank": rank,
                                 "recommended_item": item["item"], "score": item["score"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["global_user_id", "task", "rank", "recommended_item", "score"])
        writer.writeheader(); writer.writerows(rows)
    metrics = {}
    for task, lists in recommendation_lists.items():
        metrics.update(diversity_metrics(lists, model, task))
        for cutoff in (5, 10):
            ranks = final_hits[task][cutoff]
            weights = np.asarray(final_weights[task], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-12)
            metrics[f"{task}_final_recall@{cutoff}"] = float(
                np.sum(np.asarray([r > 0 for r in ranks])*weights)) if ranks else 0.0
            metrics[f"{task}_final_ndcg@{cutoff}"] = float(
                np.sum(np.asarray([1/np.log2(r+1) if r else 0 for r in ranks])*weights)) if ranks else 0.0
    return metrics
