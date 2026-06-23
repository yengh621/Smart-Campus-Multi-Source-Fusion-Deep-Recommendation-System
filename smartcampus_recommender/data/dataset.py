from __future__ import annotations

from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig
from data.preprocessing import DataArtifacts, UserSample


class SmartCampusDataset(Dataset):
    def __init__(self, samples: list[UserSample], artifacts: DataArtifacts):
        self.samples = samples
        self.v = artifacts.vocabs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        encode = lambda name, seq: torch.tensor([self.v[name].encode(t) for t in seq], dtype=torch.long)
        return {
            "user_id": x.user_id,
            "split_group": self.v["split_group"].encode(x.split_group),
            "subject": self.v["subject"].encode(x.subject),
            "gender": self.v["gender"].encode(x.gender),
            "grade": self.v["grade"].encode(x.grade),
            "questions": encode("question", x.questions),
            "concepts": encode("concept", x.concepts),
            "correct": torch.tensor(x.correct, dtype=torch.float32),
            "intervals": torch.tensor(x.intervals, dtype=torch.float32),
            "night": torch.tensor(x.night_study, dtype=torch.float32),
            "wrong_streak": torch.tensor(x.wrong_streak, dtype=torch.float32),
            "relative_position": torch.tensor(x.relative_position, dtype=torch.long),
            "course_questions": encode("question", x.course_questions),
            "course_concepts": encode("concept", x.course_concepts),
            "course_correct": torch.tensor(x.course_correct, dtype=torch.float32),
            "course_intervals": torch.tensor(x.course_intervals, dtype=torch.float32),
            "course_night": torch.tensor(x.course_night_study, dtype=torch.float32),
            "course_wrong_streak": torch.tensor(x.course_wrong_streak, dtype=torch.float32),
            "course_relative_position": torch.tensor(x.course_relative_position, dtype=torch.long),
            "course_history": encode("course", x.course_history),
            "consume_behavior": torch.tensor([
                self.v["consume"].encode(
                    item[len("CONSUME::"):] if item.startswith("CONSUME::") else item)
                for item in x.consume_items
            ], dtype=torch.long),
            "consume_meal": torch.tensor(x.consume_meal, dtype=torch.long),
            "consume_weekly": torch.tensor(x.consume_weekly, dtype=torch.float32),
            "consume_hour": torch.tensor(x.consume_hour, dtype=torch.long),
            "consume_weekday": torch.tensor(x.consume_weekday, dtype=torch.long),
            "consume_holiday": torch.tensor(x.consume_holiday, dtype=torch.long),
            "consume_age_hours": torch.tensor(x.consume_age_hours, dtype=torch.float32),
            "door_behavior": encode("door_behavior", x.door_items),
            "door_late": torch.tensor(x.door_late, dtype=torch.float32),
            "door_weekly": torch.tensor(x.door_weekly, dtype=torch.float32),
            "door_hour": torch.tensor(x.door_hour, dtype=torch.long),
            "door_weekday": torch.tensor(x.door_weekday, dtype=torch.long),
            "door_holiday": torch.tensor(x.door_holiday, dtype=torch.long),
            "door_age_hours": torch.tensor(x.door_age_hours, dtype=torch.float32),
            "target_concept": self.v["concept"].encode(x.target_concept),
            "target_course": self.v["course"].encode(x.target_course),
            "target_consume": self.v["consume"].encode(x.target_consume),
            "consume_valid": x.consume_valid,
            "knowledge_valid": x.knowledge_valid,
            "course_valid": x.course_valid,
            "alignment_valid": x.alignment_valid,
            "context_hour": x.context_hour,
            "context_weekday": x.context_weekday,
            "context_holiday": x.context_holiday,
            "sample_weight": x.sample_weight,
            "target_direct_concept": x.target_knowledge_source == "direct_concept_problem",
            "target_resolved_concept": x.target_knowledge_source in {
                "direct_concept_problem", "fallback_problem_exercise_course_concept"},
        }


def collate_batch(rows):
    def pad(key, value=0):
        result = pad_sequence([x[key] for x in rows], batch_first=True, padding_value=value)
        if result.size(1) == 0:
            result = torch.zeros((len(rows), 1), dtype=rows[0][key].dtype)
        return result
    learning_mask = pad("questions").ne(0)
    course_learning_mask = pad("course_questions").ne(0)
    course_history_mask = pad("course_history").ne(0)
    consume_mask = pad("consume_behavior").ne(0)
    door_mask = pad("door_behavior").ne(0)
    return {
        "user_id": torch.tensor([x["user_id"] for x in rows]),
        "split_group": torch.tensor([x["split_group"] for x in rows]),
        "subject": torch.tensor([x["subject"] for x in rows]),
        "gender": torch.tensor([x["gender"] for x in rows]),
        "grade": torch.tensor([x["grade"] for x in rows]),
        "questions": pad("questions"), "concepts": pad("concepts"),
        "correct": pad("correct", 0.0), "intervals": pad("intervals", 0.0),
        "night": pad("night", 0.0), "wrong_streak": pad("wrong_streak", 0.0),
        "relative_position": pad("relative_position"),
        "learning_mask": learning_mask,
        "course_questions": pad("course_questions"), "course_concepts": pad("course_concepts"),
        "course_correct": pad("course_correct", 0.0),
        "course_intervals": pad("course_intervals", 0.0),
        "course_night": pad("course_night", 0.0),
        "course_wrong_streak": pad("course_wrong_streak", 0.0),
        "course_relative_position": pad("course_relative_position"),
        "course_learning_mask": course_learning_mask,
        "course_history": pad("course_history"), "course_history_mask": course_history_mask,
        "consume_behavior": pad("consume_behavior"), "consume_meal": pad("consume_meal"),
        "consume_weekly": pad("consume_weekly", 0.0), "consume_hour": pad("consume_hour"),
        "consume_weekday": pad("consume_weekday"), "consume_holiday": pad("consume_holiday"),
        "consume_age_hours": pad("consume_age_hours", 0.0),
        "consume_mask": consume_mask,
        "door_behavior": pad("door_behavior"), "door_late": pad("door_late", 0.0),
        "door_weekly": pad("door_weekly", 0.0), "door_hour": pad("door_hour"),
        "door_weekday": pad("door_weekday"), "door_holiday": pad("door_holiday"),
        "door_age_hours": pad("door_age_hours", 0.0),
        "door_mask": door_mask,
        "target_concept": torch.tensor([x["target_concept"] for x in rows]),
        "target_course": torch.tensor([x["target_course"] for x in rows]),
        "target_consume": torch.tensor([x["target_consume"] for x in rows]),
        "consume_valid": torch.tensor([x["consume_valid"] for x in rows], dtype=torch.bool),
        "knowledge_valid": torch.tensor([x["knowledge_valid"] for x in rows], dtype=torch.bool),
        "course_valid": torch.tensor([x["course_valid"] for x in rows], dtype=torch.bool),
        "alignment_valid": torch.tensor([x["alignment_valid"] for x in rows], dtype=torch.bool),
        "context_hour": torch.tensor([x["context_hour"] for x in rows]),
        "context_weekday": torch.tensor([x["context_weekday"] for x in rows]),
        "context_holiday": torch.tensor([x["context_holiday"] for x in rows]),
        "sample_weight": torch.tensor([x["sample_weight"] for x in rows], dtype=torch.float32),
        "target_direct_concept": torch.tensor([x["target_direct_concept"] for x in rows], dtype=torch.bool),
        "target_resolved_concept": torch.tensor([x["target_resolved_concept"] for x in rows], dtype=torch.bool),
    }


def stratified_split(samples: list[UserSample], config: ExperimentConfig):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.split_group, []).append(sample)
    groups = list(grouped)
    labels = [grouped[g][0].subject for g in groups]
    train_groups, remain_groups = train_test_split(
        groups, test_size=1-config.train_ratio, random_state=config.seed, stratify=labels)
    remain_labels = [grouped[g][0].subject for g in remain_groups]
    relative_test = config.test_ratio / (config.val_ratio + config.test_ratio)
    val_groups, test_groups = train_test_split(
        remain_groups, test_size=relative_test, random_state=config.seed, stratify=remain_labels)
    flatten = lambda keys: [sample for key in keys for sample in grouped[key]]
    train, val, test = flatten(train_groups), flatten(val_groups), flatten(test_groups)
    return train, val, test


def build_dataloaders(artifacts: DataArtifacts, config: ExperimentConfig, logger):
    train, val, test = stratified_split(artifacts.samples, config)
    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  collate_fn=collate_batch, pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(SmartCampusDataset(train, artifacts), shuffle=True, **kwargs),
        "val": DataLoader(SmartCampusDataset(val, artifacts), shuffle=False, **kwargs),
        "test": DataLoader(SmartCampusDataset(test, artifacts), shuffle=False, **kwargs),
    }
    logger.info("按原始卡分组划分：train=%d, val=%d, test=%d 条事件样本；分组数=%d/%d/%d",
                len(train), len(val), len(test),
                len({x.split_group for x in train}), len({x.split_group for x in val}),
                len({x.split_group for x in test}))
    logger.info("训练集主用户学科分布：%s",
                dict(Counter(x.subject for x in train if x.knowledge_valid)))
    return loaders, {"train": train, "val": val, "test": test}
