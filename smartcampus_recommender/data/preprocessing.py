"""融合数据加载、时序特征构造、知识图谱整理与缓存。"""
from __future__ import annotations

import csv
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from data.vocab import Vocabulary
from data.knowledge_mapping import build_enhanced_problem_mapping
from utils.common import first_value, iter_jsonl, resolve_file


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("/", "-")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    return None


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


@dataclass
class UserSample:
    user_id: int
    split_group: str
    subject: str
    gender: str
    grade: str
    questions: list[str]
    concepts: list[str]
    correct: list[float]
    intervals: list[float]
    night_study: list[float]
    wrong_streak: list[float]
    relative_position: list[int]
    course_questions: list[str]
    course_concepts: list[str]
    course_correct: list[float]
    course_intervals: list[float]
    course_night_study: list[float]
    course_wrong_streak: list[float]
    course_relative_position: list[int]
    consume_items: list[str]
    consume_meal: list[int]
    consume_weekly: list[float]
    consume_hour: list[int]
    consume_weekday: list[int]
    consume_holiday: list[int]
    door_items: list[str]
    door_late: list[float]
    door_weekly: list[float]
    door_hour: list[int]
    door_weekday: list[int]
    door_holiday: list[int]
    consume_age_hours: list[float]
    door_age_hours: list[float]
    target_concept: str
    target_knowledge_source: str
    target_course: str
    course_history: list[str]
    target_consume: str
    consume_valid: bool
    knowledge_valid: bool
    course_valid: bool
    alignment_valid: bool
    context_hour: int
    context_weekday: int
    context_holiday: int
    sample_weight: float


@dataclass
class DataArtifacts:
    samples: list[UserSample]
    vocabs: dict[str, Vocabulary]
    kg_triples: list[tuple[int, int, int]]
    subject_counts: dict[str, int]
    file_paths: dict[str, str]
    knowledge_mapping_stats: dict[str, int]
    course_difficulty: list[float]
    knowledge_course_edges: list[tuple[int, int, float]]
    hierarchy_mapping_stats: dict[str, int]


def relation_pairs(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                yield parts[0].strip(), parts[1].strip()


def normalize_course(value: Any) -> str:
    text = str(value)
    return text if text.startswith("C_") else f"C_{text}"


HOLIDAYS_2019 = {
    "2019-01-01", "2019-04-05", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04",
    "2019-06-07", "2019-09-13", "2019-10-01", "2019-10-02", "2019-10-03",
    "2019-10-04", "2019-10-05", "2019-10-06", "2019-10-07",
}


def holiday_flag(time: datetime) -> int:
    return int(time.weekday() >= 5 or time.strftime("%Y-%m-%d") in HOLIDAYS_2019)


def build_learning_events(path: Path, problem_concept: dict[str, str], mapping_source: dict[str, str]):
    events: dict[int, list[tuple[datetime, str, str, float, str]]] = defaultdict(list)
    orphan_problem_units = set()
    for row in iter_jsonl(path):
        uid = int(first_value(row, "global_user_id"))
        problem = str(first_value(row, "problem_id"))
        correct = float(first_value(row, "correct", "is_correct", default=0) or 0)
        time = parse_time(first_value(row, "time", "submit_time"))
        if time:
            concept = problem_concept.get(problem)
            if concept is None:
                concept = f"PROBLEM_UNIT::{problem}"
                orphan_problem_units.add(problem)
                source = "orphan_problem_unit"
            else:
                source = mapping_source.get(problem, "unknown")
            events[uid].append((time, problem, concept, correct, source))
    return events, len(orphan_problem_units)


def build_behavior_events(consume_path: Path, door_path: Path):
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(consume_path):
        uid = int(first_value(row, "global_user_id"))
        time = parse_time(first_value(row, "time", "Date"))
        category = str(first_value(row, "consume_type", "Dept", "Type", default="消费"))
        if time:
            events[uid].append({"time": time, "item": f"CONSUME::{category}",
                                "consume_target": category, "is_door": False})
    for row in read_csv(door_path):
        uid = int(first_value(row, "global_user_id"))
        time = parse_time(first_value(row, "time", "Date"))
        address = str(first_value(row, "Address", "address", default="门禁"))
        if time:
            events[uid].append({"time": time, "item": f"DOOR::{address}",
                                "consume_target": None, "is_door": True})
    return events


def normalize_gender(value: Any) -> str:
    text = str(value).strip().lower()
    return {"0": "女", "f": "女", "female": "女", "女": "女",
            "1": "男", "m": "男", "male": "男", "男": "男"}.get(text, "未知")


def normalize_grade(value: Any, major: Any = "") -> str:
    match = re.search(r"(?:19|20)?(\d{2})", str(value or "") or str(major or ""))
    if not match:
        return "未知"
    year = int(match.group(1))
    year = year if year >= 1900 else (2000 + year if year < 50 else 1900 + year)
    return f"{year}级" if 2000 <= year <= 2035 else "未知"


def build_profiles(user_path: Path, student_path: Path, mapping_path: Path | None = None) -> dict[int, dict[str, Any]]:
    profiles = {}
    for row in iter_jsonl(user_path):
        uid = int(first_value(row, "global_user_id"))
        profiles[uid] = {
            "subject": str(first_value(row, "subject", "major", default="其他")),
            "gender": normalize_gender(first_value(row, "gender", "Sex", default="")),
            "grade": normalize_grade(first_value(row, "grade", default="")),
            "courses": [normalize_course(x) for x in first_value(row, "course_order", default=[])],
            "course_enroll_times": [parse_time(x) for x in first_value(row, "enroll_time", default=[])],
            "split_group": str(uid),
            "sample_weight": 1.0,
        }
    for row in read_csv(student_path):
        uid = int(first_value(row, "global_user_id"))
        if uid not in profiles:
            continue
        major = first_value(row, "major", "Major", default="")
        profiles[uid]["gender"] = normalize_gender(
            first_value(row, "gender", "Sex", default=profiles[uid]["gender"]))
        if profiles[uid]["grade"] == "未知":
            profiles[uid]["grade"] = normalize_grade("", major)
    if mapping_path:
        mapping_rows = list(read_csv(mapping_path))
        reuse = Counter(str(first_value(row, "teddy_origin_card", default="")) for row in mapping_rows)
        for row in mapping_rows:
            uid = int(first_value(row, "global_user_id"))
            if uid in profiles:
                card = str(first_value(row, "teddy_origin_card", default=uid))
                profiles[uid]["split_group"] = card
                profiles[uid]["sample_weight"] = 1.0 / max(reuse[card], 1)
    return profiles


def make_sample(uid: int, profile: dict[str, Any], learning, behavior,
                max_learning: int, max_behavior: int, min_learning: int,
                consume_target_index: int | None = None, primary: bool = True,
                course_target_index: int | None = None, consume_enabled: bool = True,
                course_only: bool = False) -> UserSample | None:
    learning = sorted(learning, key=lambda x: x[0])
    behavior = sorted(behavior, key=lambda x: x["time"])
    consume_events = [x for x in behavior if x["consume_target"]]
    courses = profile["courses"]
    if len(learning) < min_learning or not courses:
        return None

    target_learning = learning[-1]
    # 最后一次消费是监督目标；目标自身及其后的门禁事件不能进入输入。
    target_consume_event = (consume_events[consume_target_index]
                            if consume_events and consume_target_index is not None
                            else consume_events[-1] if consume_events and consume_enabled else None)
    cutoff = target_consume_event["time"] if target_consume_event else None
    behavior_history = [x for x in behavior if cutoff is None or x["time"] < cutoff]
    # Consumption windows must not see learning events that happened after the
    # target transaction. Knowledge/course samples have their own independent
    # observation window and therefore retain the normal next-event history.
    history_source = learning[:-1] if cutoff is None else learning
    history = [x for x in history_source if cutoff is None or x[0] < cutoff][-max_learning:]
    context_time = behavior_history[-1]["time"] if behavior_history else cutoff
    questions, concepts, correct, intervals, night, streaks = [], [], [], [], [], []
    previous = None
    streak = 0
    for time, question, concept, is_correct, _source in history:
        questions.append(question)
        concepts.append(concept)
        correct.append(is_correct)
        delta_hours = 0.0 if previous is None else max(0.0, (time - previous).total_seconds() / 3600)
        intervals.append(min(delta_hours, 8760.0))
        night.append(float(time.hour >= 23 or time.hour < 6))
        streak = 0 if is_correct else streak + 1
        streaks.append(min(streak / 10.0, 1.0))
        previous = time

    # 课程推荐使用独立时间切片：只能读取目标课程选课时刻之前的答题记录。
    enroll_times = profile.get("course_enroll_times", [])
    course_index = course_target_index if course_target_index is not None else len(courses)-1
    course_cutoff = (enroll_times[course_index]
                     if len(enroll_times) == len(courses) and enroll_times else None)
    course_events = ([x for x in learning if course_cutoff is not None and x[0] < course_cutoff]
                     [-max_learning:])
    cq, cc, cr, ci, cn, cw = [], [], [], [], [], []
    previous = None
    streak = 0
    for time, question, concept, is_correct, _source in course_events:
        cq.append(question); cc.append(concept); cr.append(is_correct)
        delta_hours = 0.0 if previous is None else max(0.0, (time-previous).total_seconds()/3600)
        ci.append(min(delta_hours, 8760.0))
        cn.append(float(time.hour >= 23 or time.hour < 6))
        streak = 0 if is_correct else streak + 1
        cw.append(min(streak/10.0, 1.0)); previous = time

    def encode_behavior_stream(stream, is_door):
        stream = stream[-max_behavior:]
        reference_time = cutoff or (stream[-1]["time"] if stream else None)
        week_counts = Counter((x["time"].isocalendar()[0], x["time"].isocalendar()[1]) for x in stream)
        result = {key: [] for key in ("items", "meal", "late", "weekly", "hour", "weekday", "holiday", "age")}
        for event in stream:
            time = event["time"]
            result["items"].append(event["item"])
            result["meal"].append(1 if 6 <= time.hour < 10 else 2 if 10 <= time.hour < 14 else 3 if 17 <= time.hour < 21 else 0)
            result["late"].append(float(is_door and (time.hour >= 23 or time.hour < 5)))
            count = week_counts[(time.isocalendar()[0], time.isocalendar()[1])]
            result["weekly"].append(min(math.log1p(count) / 5.0, 1.0))
            result["hour"].append(time.hour); result["weekday"].append(time.weekday())
            result["holiday"].append(holiday_flag(time))
            result["age"].append(max(0.0, (reference_time-time).total_seconds()/3600) if reference_time else 0.0)
        return result

    consume_stream = encode_behavior_stream([x for x in behavior_history if not x["is_door"]], False)
    door_stream = encode_behavior_stream([x for x in behavior_history if x["is_door"]], True)

    return UserSample(
        user_id=uid, split_group=profile.get("split_group", str(uid)),
        subject=profile["subject"], gender=profile["gender"], grade=profile["grade"],
        questions=questions, concepts=concepts, correct=correct, intervals=intervals,
        night_study=night, wrong_streak=streaks, relative_position=list(range(len(history))),
        course_questions=cq, course_concepts=cc, course_correct=cr, course_intervals=ci,
        course_night_study=cn, course_wrong_streak=cw,
        course_relative_position=list(range(len(course_events))),
        consume_items=consume_stream["items"], consume_meal=consume_stream["meal"],
        consume_weekly=consume_stream["weekly"], consume_hour=consume_stream["hour"],
        consume_weekday=consume_stream["weekday"], consume_holiday=consume_stream["holiday"],
        door_items=door_stream["items"], door_late=door_stream["late"],
        door_weekly=door_stream["weekly"], door_hour=door_stream["hour"],
        door_weekday=door_stream["weekday"], door_holiday=door_stream["holiday"],
        consume_age_hours=consume_stream["age"], door_age_hours=door_stream["age"],
        target_concept=target_learning[2],
        target_knowledge_source=target_learning[4],
        target_course=courses[course_index],
        course_history=courses[:course_index],
        target_consume=target_consume_event["consume_target"] if target_consume_event else "<UNK>",
        consume_valid=target_consume_event is not None,
        knowledge_valid=primary and not course_only,
        course_valid=primary or course_only,
        alignment_valid=not course_only,
        context_hour=context_time.hour if context_time else 0,
        context_weekday=context_time.weekday() if context_time else 0,
        context_holiday=holiday_flag(context_time) if context_time else 0,
        sample_weight=float(profile.get("sample_weight", 1.0)),
    )


def build_kg(root: Path, concept_vocab: Vocabulary, course_vocab: Vocabulary):
    entity_vocab = Vocabulary()
    relation_vocab = Vocabulary()
    triples_raw: list[tuple[str, str, str]] = []
    cc = resolve_file(root, "concept-course.txt", "sampled_concept_course.txt", required=False)
    if cc:
        for concept, course in relation_pairs(cc):
            triples_raw.append((normalize_course(course), "has_concept", concept))

    concept_names: dict[str, str] = {}
    concept_file = resolve_file(root, "concept.json", "sampled_concept.json", required=False)
    if concept_file:
        for row in iter_jsonl(concept_file):
            concept_names.setdefault(str(row.get("name", "")), str(row.get("id", "")))
    prereq_dir = root / "sampled_prerequisites"
    if not prereq_dir.exists():
        prereq_dir = root / "prerequisites"
    for path in prereq_dir.glob("*.json") if prereq_dir.exists() else []:
        for row in iter_jsonl(path):
            head, tail = concept_names.get(str(row.get("c1"))), concept_names.get(str(row.get("c2")))
            if head and tail:
                triples_raw.append((head, "prerequisite_of", tail))

    for h, r, t in triples_raw:
        entity_vocab.add(h); entity_vocab.add(t); relation_vocab.add(r)
    triples = [(entity_vocab.encode(h), relation_vocab.encode(r), entity_vocab.encode(t)) for h, r, t in triples_raw]
    return entity_vocab, relation_vocab, triples


def build_vocabs(samples: list[UserSample], root: Path):
    vocabs = {name: Vocabulary() for name in (
        "subject", "gender", "grade", "question", "concept", "course",
        "consume_behavior", "door_behavior", "consume", "split_group"
    )}
    for sample in samples:
        vocabs["subject"].add(sample.subject); vocabs["gender"].add(sample.gender); vocabs["grade"].add(sample.grade)
        vocabs["split_group"].add(sample.split_group)
        vocabs["question"].build(sample.questions); vocabs["concept"].build(sample.concepts)
        vocabs["question"].build(sample.course_questions); vocabs["concept"].build(sample.course_concepts)
        # 预测目标必须显式进入词表，不能因只出现在序列末端而退化为UNK。
        vocabs["concept"].add(sample.target_concept)
        vocabs["course"].add(sample.target_course)
        vocabs["course"].build(sample.course_history)
        vocabs["consume_behavior"].build(sample.consume_items)
        vocabs["consume"].build(
            item[len("CONSUME::"):] if item.startswith("CONSUME::") else item
            for item in sample.consume_items)
        vocabs["door_behavior"].build(sample.door_items)
        if sample.consume_valid:
            vocabs["consume"].add(sample.target_consume)
    entity_vocab, relation_vocab, triples = build_kg(root, vocabs["concept"], vocabs["course"])
    vocabs["kg_entity"] = entity_vocab; vocabs["kg_relation"] = relation_vocab
    return vocabs, triples


def build_course_difficulty(root: Path, course_vocab: Vocabulary) -> list[float]:
    """用课程覆盖的唯一知识点数量构造无标签难度先验，并归一化到[0,1]。"""
    concepts_by_course: dict[str, set[str]] = defaultdict(set)
    path = resolve_file(root, "concept-course.txt", "sampled_concept_course.txt", required=False)
    if path:
        for concept, course in relation_pairs(path):
            token = normalize_course(course)
            if token in course_vocab.token_to_idx:
                concepts_by_course[token].add(concept)
    raw = [0.0] * len(course_vocab)
    for token, concepts in concepts_by_course.items():
        raw[course_vocab.encode(token)] = math.log1p(len(concepts))
    observed = [x for x in raw[2:] if x > 0]
    low, high = (min(observed), max(observed)) if observed else (0.0, 1.0)
    midpoint = 0.5
    return [0.0 if i < 2 else ((x-low)/(high-low) if x > 0 and high > low else midpoint)
            for i, x in enumerate(raw)]


def build_hierarchical_knowledge_course_mapping(
        root: Path, knowledge_vocab: Vocabulary, course_vocab: Vocabulary
        ) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    """构造确定性分级映射；同一边优先保留可信度更高的来源。"""
    edges: dict[tuple[int, int], tuple[float, str]] = {}

    def add(knowledge: str, course: str, confidence: float, source: str):
        if knowledge not in knowledge_vocab.token_to_idx or course not in course_vocab.token_to_idx:
            return
        key = (knowledge_vocab.encode(knowledge), course_vocab.encode(course))
        if key not in edges or confidence > edges[key][0]:
            edges[key] = (confidence, source)

    # 第一级：数据集明确给出的 Concept -> Course。
    cc = resolve_file(root, "concept-course.txt", "sampled_concept_course.txt", required=False)
    if cc:
        for concept, course in relation_pairs(cc):
            add(concept, normalize_course(course), 1.0, "direct_concept_course")

    # 第二级：Course.resource 中明确出现的 Exercise -> Course。
    exercise_course: dict[str, set[str]] = defaultdict(set)
    course_file = resolve_file(root, "course.json", "sampled_course.json", required=False)
    if course_file:
        for row in iter_jsonl(course_file):
            course = normalize_course(first_value(row, "id", "course_id"))
            for resource in row.get("resource") or []:
                rid = str(resource.get("resource_id", "")) if isinstance(resource, dict) else ""
                if rid.startswith("Ex_"):
                    exercise_course[rid].add(course)
                    add(f"EXERCISE_UNIT::{rid}", course, 0.90, "exercise_course")

    # 第三级：Problem.exercise_id -> Exercise -> Course。
    problem_file = resolve_file(root, "problem.json", "sampled_problem.json", required=False)
    if problem_file:
        for row in iter_jsonl(problem_file):
            raw = first_value(row, "problem_id", "id", default="")
            problem = str(raw) if str(raw).startswith("Pm_") else f"Pm_{raw}"
            exercise = str(row.get("exercise_id", ""))
            for course in exercise_course.get(exercise, ()):
                add(f"PROBLEM_UNIT::{problem}", course, 0.80, "problem_exercise_course")

    stats = Counter(source for _, source in edges.values())
    stats["total_edges"] = len(edges)
    stats["mapped_knowledge_units"] = len({k for k, _ in edges})
    stats["mapped_courses"] = len({c for _, c in edges})
    return [(k, c, confidence) for (k, c), (confidence, _) in edges.items()], dict(stats)


def prepare_data(config: ExperimentConfig, logger, force_rebuild: bool = False) -> DataArtifacts:
    config.make_dirs()
    cache_path = config.output_path / "cache" / config.cache_name
    if cache_path.exists() and not force_rebuild:
        logger.info("加载预处理缓存：%s", cache_path)
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    root = config.data_path
    paths = {
        "user": resolve_file(root, "mooc_user_mapped.jsonl", "sampled_mooc_user.json"),
        "problem": resolve_file(root, "user_problem_mapped.jsonl", "sampled_user_problem.json"),
        "student": resolve_file(root, "teddy_student_mapped.csv", "mapped_teddy_student.csv"),
        "consume": resolve_file(root, "teddy_consume_mapped.csv", "mapped_teddy_consume.csv"),
        "door": resolve_file(root, "teddy_door_mapped.csv", "mapped_teddy_door.csv"),
        "concept_problem": resolve_file(root, "concept-problem.txt", "sampled_concept_problem.txt", required=False),
        "mapping": resolve_file(root, "all_global_id_mapping.csv", required=False),
    }
    logger.info("读取用户画像与关系映射")
    profiles = build_profiles(paths["user"], paths["student"], paths["mapping"])
    problem_concept, mapping_source, mapping_stats = build_enhanced_problem_mapping(root, paths["concept_problem"])
    logger.info("知识点映射：直接=%d，课程链路回退=%d，习题组回退=%d，仍未映射=%d",
                mapping_stats.get("direct", 0), mapping_stats.get("fallback", 0),
                mapping_stats.get("exercise_unit_fallback", 0),
                mapping_stats.get("unmapped", 0))
    logger.info("读取答题时序")
    learning, orphan_count = build_learning_events(paths["problem"], problem_concept, mapping_source)
    mapping_stats["orphan_problem_unit"] = orphan_count
    logger.info("答题表中缺少problem实体的孤立题目单元：%d", orphan_count)
    logger.info("读取消费与门禁时序")
    behavior = build_behavior_events(paths["consume"], paths["door"])
    samples = []
    for uid, profile in profiles.items():
        user_behavior = behavior.get(uid, [])
        consume_count = sum(bool(x.get("consume_target")) for x in user_behavior)
        eligible = list(range(config.min_consume_history, consume_count,
                              max(config.consume_window_stride, 1)))
        if consume_count and (not eligible or eligible[-1] != consume_count-1):
            eligible.append(consume_count-1)
        eligible = eligible[-config.max_consume_windows_per_user:]
        if not eligible:
            eligible = [None]
        # Keep learning/course supervision in a dedicated sample. Mixing it with
        # a consumption cutoff gives the three tasks different observation times
        # inside one row and can leak future learning events into consumption.
        sample = make_sample(
            uid, profile, learning.get(uid, []), user_behavior,
            config.max_learning_len, config.max_behavior_len, config.min_learning_events,
            consume_target_index=None, primary=True, consume_enabled=False)
        if sample:
            samples.append(sample)
        for target_index in eligible:
            if target_index is None:
                continue
            sample = make_sample(
                uid, profile, learning.get(uid, []), user_behavior,
                config.max_learning_len, config.max_behavior_len, config.min_learning_events,
                consume_target_index=target_index, primary=False)
            if sample:
                samples.append(sample)
        # 每次后续选课都可形成一个课程窗口；最后一门课程已由主样本承担。
        earlier_course_indices = list(range(1, max(len(profile["courses"])-1, 1)))
        for course_index in earlier_course_indices[-config.max_course_windows_per_user:]:
            sample = make_sample(
                uid, profile, learning.get(uid, []), user_behavior,
                config.max_learning_len, config.max_behavior_len, config.min_learning_events,
                primary=False, course_target_index=course_index,
                consume_enabled=False, course_only=True)
            if sample:
                samples.append(sample)
    if len(samples) < 100:
        raise RuntimeError(f"有效用户仅 {len(samples)}，请检查输入字段与数据完整性")
    vocabs, triples = build_vocabs(samples, root)
    course_difficulty = build_course_difficulty(root, vocabs["course"])
    hierarchy_edges, hierarchy_stats = build_hierarchical_knowledge_course_mapping(
        root, vocabs["concept"], vocabs["course"])
    artifacts = DataArtifacts(samples=samples, vocabs=vocabs, kg_triples=triples,
                              subject_counts=dict(Counter(x.subject for x in samples if x.knowledge_valid)),
                              file_paths={k: str(v) for k, v in paths.items() if v},
                              knowledge_mapping_stats=mapping_stats,
                              course_difficulty=course_difficulty,
                              knowledge_course_edges=hierarchy_edges,
                              hierarchy_mapping_stats=hierarchy_stats)
    (config.output_path / "result" / "knowledge_mapping_stats.json").write_text(
        __import__("json").dumps(mapping_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_path / "result" / "hierarchy_mapping_stats.json").write_text(
        __import__("json").dumps(hierarchy_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with cache_path.open("wb") as handle:
        pickle.dump(artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("预处理完成：事件级样本 %d，唯一用户 %d，KG三元组 %d",
                len(samples), len({x.user_id for x in samples}), len(triples))
    return artifacts
