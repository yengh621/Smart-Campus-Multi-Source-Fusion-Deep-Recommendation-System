"""构建高精度直接映射与课程级回退映射：problem→exercise→course→concept。"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from utils.common import iter_jsonl, resolve_file


def relation_pairs(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                yield parts[0].strip(), parts[1].strip()


def normalize_problem(value) -> str:
    text = str(value)
    return text if text.startswith("Pm_") else f"Pm_{text}"


def normalize_course(value) -> str:
    text = str(value)
    return text if text.startswith("C_") else f"C_{text}"


def concept_name(concept_id: str) -> str:
    if concept_id.startswith("K_"):
        return concept_id[2:].rsplit("_", 1)[0]
    return concept_id


def char_bigrams(text: str) -> set[str]:
    cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i+2] for i in range(len(cleaned)-1)}


def relevance_score(problem_text: str, concept_id: str) -> tuple[float, str]:
    """优先知识点名称直接命中，否则用字符二元组Jaccard相似度。"""
    name = concept_name(concept_id)
    if name and name in problem_text:
        return 2.0 + len(name) / 1000.0, concept_id
    left, right = char_bigrams(problem_text), char_bigrams(name)
    score = len(left & right) / max(len(left | right), 1)
    return score, concept_id


def build_enhanced_problem_mapping(root: Path, direct_path: Path | None):
    direct: dict[str, str] = {}
    if direct_path:
        for concept, problem in relation_pairs(direct_path):
            direct.setdefault(normalize_problem(problem), concept)

    problem_file = resolve_file(root, "problem.json", "sampled_problem.json")
    course_file = resolve_file(root, "course.json", "sampled_course.json")
    concept_course_file = resolve_file(root, "concept-course.txt", "sampled_concept_course.txt")

    # exercise_id → course_id：课程resource中Ex_开头的资源即习题组。
    exercise_course: dict[str, str] = {}
    for course in iter_jsonl(course_file):
        cid = normalize_course(course.get("id") or course.get("course_id"))
        for resource in course.get("resource") or []:
            rid = str(resource.get("resource_id", "")) if isinstance(resource, dict) else ""
            if rid.startswith("Ex_"):
                exercise_course.setdefault(rid, cid)

    course_concepts: dict[str, list[str]] = defaultdict(list)
    for concept, course in relation_pairs(concept_course_file):
        course_concepts[normalize_course(course)].append(concept)
    for course, concepts in course_concepts.items():
        course_concepts[course] = sorted(set(concepts))

    mapping = dict(direct)
    source = {problem: "direct_concept_problem" for problem in direct}
    stats = Counter(direct=len(direct))
    for problem in iter_jsonl(problem_file):
        pid = normalize_problem(problem.get("id") or problem.get("problem_id"))
        if pid in mapping:
            continue
        exercise = str(problem.get("exercise_id", ""))
        course = exercise_course.get(exercise)
        candidates = course_concepts.get(course, []) if course else []
        if not candidates:
            # problem仍有exercise归属时，保留明确的习题组知识单元，不退化为无语义哈希桶。
            if exercise:
                mapping[pid] = f"EXERCISE_UNIT::{exercise}"
                source[pid] = "fallback_exercise_unit"
                stats["exercise_unit_fallback"] += 1
            else:
                stats["unmapped"] += 1
            continue
        text = " ".join(str(problem.get(key, "")) for key in ("title", "content", "typetext"))
        # 分数并列时按concept_id稳定决胜，保证实验可复现。
        best = max(candidates, key=lambda concept: relevance_score(text, concept))
        mapping[pid] = best
        source[pid] = "fallback_problem_exercise_course_concept"
        stats["fallback"] += 1
    stats["mapped_total"] = len(mapping)
    stats["course_with_concepts"] = len(course_concepts)
    stats["exercise_with_course"] = len(exercise_course)
    return mapping, source, dict(stats)
