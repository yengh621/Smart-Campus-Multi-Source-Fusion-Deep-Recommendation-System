#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOOCCubeX 与泰迪杯一卡通数据的学科分层匹配、全局用户 ID 统一脚本。

设计事实（论文中也应如此表述）：
1. MOOCCubeX 用户表没有专业字段。本脚本只根据 course_order 反查课程 field，
   以出现频次最高的原始 field 推导用户学科，不读取或伪造用户专业。
2. 两套数据不存在真实身份对应关系。global_user_id 是按学科分层建立的模拟配对键，
   不能描述成现实中的同一个人。
3. 输入大文件全部逐行读取；内存中仅保留课程索引、计数/ID 集合和最终样本。

默认路径适配当前仓库：
  MOOCCubeX/MOOCCubeX-main/{entities,relations,prerequisites}
  dd0f7-main/data1.csv  (student)
  dd0f7-main/data2.csv  (consume)
  dd0f7-main/data3.csv  (door)

运行：
  python build_cross_dataset.py

自定义路径示例：
  python build_cross_dataset.py --mooc-root D:/MOOCCubeX --teddy-root D:/teddy \
      --student student.csv --consume consume.csv --door door.csv --output output
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SUBJECTS: Tuple[str, ...] = (
    "计算机类", "数学理学", "经管类", "机械工科",
    "土建工科", "艺术传媒", "外语", "其他",
)

# MOOCCubeX course.field 是研究生学科/专业领域列表。映射顺序不参与判定，
# 用户主 field 先按原始 field 频次选出，再使用本表归并为八大类。
MOOC_FIELD_TO_SUBJECT: Dict[str, str] = {
    # 计算机与电子信息
    "计算机科学与技术": "计算机类", "软件工程": "计算机类",
    "网络空间安全": "计算机类", "电子科学与技术": "计算机类",
    "信息与通信工程": "计算机类", "控制科学与工程": "计算机类",
    "系统科学": "计算机类", "计算机技术": "计算机类",
    "电子与通信工程": "计算机类", "光学工程": "计算机类",
    # 数学与自然科学
    "数学": "数学理学", "统计学": "数学理学", "物理学": "数学理学",
    "化学": "数学理学", "生物学": "数学理学", "地理学": "数学理学",
    "大气科学": "数学理学", "海洋科学": "数学理学",
    "地球物理学": "数学理学", "地质学": "数学理学",
    "天文学": "数学理学", "生态学": "数学理学", "科学技术史": "数学理学",
    # 经济与管理
    "理论经济学": "经管类", "应用经济学": "经管类", "工商管理": "经管类",
    "管理科学与工程": "经管类", "公共管理": "经管类",
    "农林经济管理": "经管类", "情报与档案管理": "经管类",
    "图书情报与档案管理": "经管类",
    # 机械及广义制造工科
    "机械工程": "机械工科", "仪器科学与技术": "机械工科",
    "材料科学与工程": "机械工科", "冶金工程": "机械工科",
    "动力工程及工程热物理": "机械工科", "电气工程": "机械工科",
    "化学工程与技术": "机械工科", "石油与天然气工程": "机械工科",
    "矿业工程": "机械工科", "纺织科学与工程": "机械工科",
    "轻工技术与工程": "机械工科", "交通运输工程": "机械工科",
    "船舶与海洋工程": "机械工科", "航空宇航科学与技术": "机械工科",
    "核科学与技术": "机械工科", "兵器科学与技术": "机械工科",
    "农业工程": "机械工科", "食品科学与工程": "机械工科",
    # 土建与空间工程
    "土木工程": "土建工科", "建筑学": "土建工科",
    "城乡规划学": "土建工科", "风景园林学": "土建工科",
    "水利工程": "土建工科", "测绘科学与技术": "土建工科",
    "地质资源与地质工程": "土建工科", "环境科学与工程": "土建工科",
    # 艺术和传媒
    "艺术学": "艺术传媒", "美术学": "艺术传媒", "设计学": "艺术传媒",
    "音乐与舞蹈学": "艺术传媒", "戏剧与影视学": "艺术传媒",
    "新闻传播学": "艺术传媒",
    # 外语必须在中国语言文学等人文门类之外单列
    "外国语言文学": "外语",
}

# 泰迪杯细分专业关键词。按顺序匹配；更具体的规则应放在前面。
# “宝玉石鉴定”按矿物/材料/地学方向归入数学理学，使本数据的八组均可配对。
TEDDY_MAJOR_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("计算机类", ("计算机", "软件", "网络", "嵌入式", "大数据", "人工智能", "信息安全")),
    ("数学理学", ("数学", "统计", "物理", "化学", "生物", "科学", "宝玉石鉴定", "地质")),
    ("土建工科", ("建筑", "土木", "工程造价", "市政", "道路", "桥梁", "测绘", "园林")),
    ("艺术传媒", ("艺术", "视觉", "动漫", "动画", "传媒", "广告", "首饰", "工业设计", "产品设计")),
    ("外语", ("英语", "日语", "俄语", "德语", "法语", "西班牙语", "外语", "翻译")),
    ("经管类", ("金融", "会计", "审计", "投资", "理财", "商务", "国贸", "贸易", "工商",
               "物流", "连锁", "营销", "电商", "电子商务", "旅游", "酒店", "管理")),
    ("机械工科", ("机械", "机电", "自动化", "电气", "机器人", "模具", "汽车", "制造",
                 "工业工程", "石油", "化工", "材料", "能源")),
)


def log(message: str) -> None:
    print(message, flush=True)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """逐行读取 JSONL；坏行静默跳过，并容忍原始大文件中的少量坏字节。"""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            yield line_no, value


def detect_csv_encoding(path: Path) -> str:
    """泰迪杯原文件通常为 GBK/GB18030；也兼容用户转存后的 UTF-8。"""
    # 只读取文件头，不能用 Path.read_bytes()，后者会把几十 MB 的消费表整体读入。
    with path.open("rb") as handle:
        sample = handle.read(65536)
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        # 部分竞赛原始消费表混有极少量坏字节；GB18030 容错读取比整表失败更合理。
        return "gb18030"


def csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    encoding = detect_csv_encoding(path)
    errors = "strict" if encoding == "utf-8-sig" else "replace"
    with path.open("r", encoding=encoding, errors=errors, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV 无表头：{path}")
        for row in reader:
            yield {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def normalized_id(value: Any) -> str:
    """统一比较用 ID；兼容 CSV 软件把整数卡号写成 180001.0。"""
    text = "" if value is None else str(value).strip()
    return text[:-2] if re.fullmatch(r"[-+]?\d+\.0", text) else text


def first_present(row: Mapping[str, Any], names: Sequence[str], *, required: bool = True) -> str:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return normalized_id(row[name])
    if required:
        raise KeyError(f"字段缺失，候选字段为 {list(names)}；实际字段为 {list(row.keys())}")
    return ""


def load_override(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or any(v not in SUBJECTS for v in data.values()):
        raise ValueError("专业覆盖 JSON 必须是 {细分专业: 八大学科之一} 的对象")
    return {str(k).strip(): str(v) for k, v in data.items()}


def classify_teddy_major(major: str, override: Mapping[str, str]) -> str:
    clean = re.sub(r"^\s*\d{2,4}", "", major).strip()
    if major in override:
        return override[major]
    if clean in override:
        return override[clean]
    for subject, keywords in TEDDY_MAJOR_RULES:
        if any(keyword in clean for keyword in keywords):
            return subject
    return "其他"


def normalize_course_id(value: Any) -> str:
    text = normalized_id(value)
    return text if text.startswith("C_") else f"C_{text}"


def normalize_problem_id(value: Any) -> str:
    """兼容 user-problem 的 Pm_123 与 problem.json 的数字 problem_id=123。"""
    text = normalized_id(value)
    return text if text.startswith("Pm_") else f"Pm_{text}"


def load_course_fields(mooc_root: Path) -> Tuple[Dict[str, Tuple[str, ...]], Counter]:
    """读取课程及 field。course.json 是事实主源，course-field.json 只补缺失 field。"""
    course_path = mooc_root / "entities" / "course.json"
    course_fields: Dict[str, Tuple[str, ...]] = {}
    field_stat: Counter = Counter()
    for _, row in iter_jsonl(course_path):
        cid = normalize_course_id(row.get("id"))
        raw = row.get("field") or []
        fields = [str(raw)] if isinstance(raw, str) else [str(x) for x in raw if x]
        course_fields[cid] = tuple(fields)

    supplemental = next((p for p in (
        mooc_root / "relations" / "course-field.json",
        mooc_root / "course-field.json",
    ) if p.exists()), None)
    if supplemental:
        for _, row in iter_jsonl(supplemental):
            cid = normalize_course_id(row.get("course_id") or row.get("id"))
            raw = row.get("field") or []
            fields = [str(raw)] if isinstance(raw, str) else [str(x) for x in raw if x]
            if cid not in course_fields or not course_fields[cid]:
                course_fields[cid] = tuple(fields)

    for fields in course_fields.values():
        field_stat.update(fields)
    log(f"[1] 课程总量：{len(course_fields):,}；原始 field 种类：{len(field_stat):,}")
    return course_fields, field_stat


def map_mooc_field(field: str) -> str:
    """未显式覆盖的医学、人文、农林、军事等统一进入“其他”。"""
    return MOOC_FIELD_TO_SUBJECT.get(field, "其他")


def count_active_users(user_problem_path: Path, threshold: int) -> Tuple[Set[str], int, int]:
    """
    流式计数。用户达到阈值后移入 active 集合并删除其小计，计数不再增长，
    因而不会为高活跃用户保存无意义的大整数。
    """
    below: MutableMapping[str, int] = defaultdict(int)
    active: Set[str] = set()
    records = 0
    for _, row in iter_jsonl(user_problem_path):
        records += 1
        uid = normalized_id(row.get("user_id"))
        if not uid or uid in active:
            continue
        below[uid] += 1
        if below[uid] >= threshold:
            active.add(uid)
            del below[uid]
        if records % 5_000_000 == 0:
            log(f"    已扫描答题 {records:,} 条，达到阈值用户 {len(active):,}")
    users_with_problem = len(active) + len(below)
    log(f"[2] 答题记录：{records:,}；有答题用户：{users_with_problem:,}；答题≥{threshold}：{len(active):,}")
    return active, records, users_with_problem


def infer_user_subject(course_order: Sequence[Any], course_fields: Mapping[str, Tuple[str, ...]]) -> Tuple[str, str]:
    """
    严格按需求：对用户所有选课的原始 field 计数，最高频 field 为推导 field，
    再将该 field 归并为八大学科。并列时按 course_order 中首次出现的 field 决胜，
    使结果稳定且可复现。
    """
    counts: Counter = Counter()
    first_position: Dict[str, int] = {}
    position = 0
    for raw_cid in course_order:
        fields = course_fields.get(normalize_course_id(raw_cid), ())
        for field in fields:
            counts[field] += 1
            first_position.setdefault(field, position)
            position += 1
    if not counts:
        return "", ""
    dominant = min(counts, key=lambda f: (-counts[f], first_position[f], f))
    return dominant, map_mooc_field(dominant)


def reservoir_add(bucket: List[Dict[str, Any]], item: Dict[str, Any], seen: int,
                  capacity: int, rng: random.Random) -> None:
    if len(bucket) < capacity:
        bucket.append(item)
    else:
        index = rng.randrange(seen)
        if index < capacity:
            bucket[index] = item


def collect_mooc_candidates(user_path: Path, active_users: Set[str],
                            course_fields: Mapping[str, Tuple[str, ...]], min_courses: int,
                            target: int, seed: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter, Dict[str, int]]:
    """第二次只流式读 user.json；为每组维护至多 target 个均匀蓄水池样本。"""
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SUBJECTS}
    available: Counter = Counter()
    stat = Counter()
    for _, user in iter_jsonl(user_path):
        stat["raw_users"] += 1
        uid = normalized_id(user.get("id"))
        order = user.get("course_order") or []
        if not isinstance(order, list) or len(order) < min_courses:
            continue
        stat["course_qualified"] += 1
        if uid not in active_users:
            continue
        stat["active_and_course"] += 1
        dominant_field, subject = infer_user_subject(order, course_fields)
        if not subject:
            stat["no_field"] += 1
            continue
        stat["fully_qualified"] += 1
        available[subject] += 1
        item = {"origin_user_id": uid, "subject": subject,
                "dominant_field": dominant_field, "user": user}
        reservoir_add(reservoirs[subject], item, available[subject], target, rng)
    log(f"[3] MOOCCubeX 原始用户：{stat['raw_users']:,}；选课≥{min_courses}：{stat['course_qualified']:,}；"
        f"同时答题达标：{stat['active_and_course']:,}；可推导学科：{stat['fully_qualified']:,}")
    return reservoirs, available, dict(stat)


def allocate_quotas(available: Mapping[str, int], target: int) -> Dict[str, int]:
    """先均分；不足组取全部，缺额按固定学科顺序轮转给仍有容量的组。"""
    base, extra = divmod(target, len(SUBJECTS))
    quotas = {s: min(available.get(s, 0), base + (1 if i < extra else 0)) for i, s in enumerate(SUBJECTS)}
    remaining = target - sum(quotas.values())
    while remaining:
        progressed = False
        for subject in SUBJECTS:
            if quotas[subject] < available.get(subject, 0):
                quotas[subject] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError(f"MOOCCubeX 合格用户总量不足 {target}，最多只能抽取 {sum(available.values())}")
    return quotas


def sample_mooc(reservoirs: Mapping[str, List[Dict[str, Any]]], available: Mapping[str, int],
                target: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    quotas = allocate_quotas(available, target)
    rng = random.Random(seed + 1)
    sampled: List[Dict[str, Any]] = []
    for subject in SUBJECTS:
        bucket = list(reservoirs[subject])
        rng.shuffle(bucket)
        sampled.extend(bucket[:quotas[subject]])
        log(f"    MOOC {subject}：合格 {available.get(subject, 0):,}，抽样 {quotas[subject]:,}")
    if len(sampled) != target:
        raise AssertionError(f"抽样数量异常：{len(sampled)} != {target}")
    return sampled, quotas


def collect_activity_ids(consume_path: Path, door_path: Path) -> Tuple[Set[str], Set[str], int, int]:
    consume_cards: Set[str] = set()
    door_cards: Set[str] = set()
    consume_rows = 0
    door_rows = 0
    for row in csv_rows(consume_path):
        consume_rows += 1
        consume_cards.add(first_present(row, ("CardNo", "卡号", "校园卡号")))
    for row in csv_rows(door_path):
        door_rows += 1
        door_cards.add(first_present(row, ("AccessCardNo", "门禁卡号", "CardNo", "卡号")))
    log(f"[4] 消费记录 {consume_rows:,} 条/{len(consume_cards):,} 张卡；"
        f"门禁记录 {door_rows:,} 条/{len(door_cards):,} 张门禁卡")
    return consume_cards, door_cards, consume_rows, door_rows


def collect_teddy_students(student_path: Path, consume_cards: Set[str], door_cards: Set[str],
                           override: Mapping[str, str], activity_policy: str
                           ) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter, Counter, Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SUBJECTS}
    raw_subject: Counter = Counter()
    valid_subject: Counter = Counter()
    major_subject: Dict[str, str] = {}
    total = 0
    for row in csv_rows(student_path):
        total += 1
        card = first_present(row, ("CardNo", "校园卡号", "卡号"))
        access = first_present(row, ("AccessCardNo", "门禁卡号"))
        major = first_present(row, ("Major", "专业", "专业名称"))
        subject = classify_teddy_major(major, override)
        major_subject[major] = subject
        raw_subject[subject] += 1
        has_consume = card in consume_cards
        has_door = access in door_cards
        valid = (has_consume and has_door) if activity_policy == "both" else (has_consume or has_door)
        if not valid:
            continue
        item = {"card": card, "access_card": access, "major": major,
                "subject": subject, "student": row}
        grouped[subject].append(item)
        valid_subject[subject] += 1
    log(f"[5] 泰迪杯原始学生：{total:,}；有效学生（activity={activity_policy}）：{sum(valid_subject.values()):,}")
    for subject in SUBJECTS:
        log(f"    泰迪 {subject}：原始 {raw_subject[subject]:,}，有效 {valid_subject[subject]:,}")
    return grouped, raw_subject, valid_subject, major_subject


def create_global_mapping(sampled: Sequence[Dict[str, Any]], teddy_grouped: Mapping[str, List[Dict[str, Any]]]
                          ) -> List[Dict[str, Any]]:
    mooc_grouped: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SUBJECTS}
    for user in sampled:
        mooc_grouped[user["subject"]].append(user)
    mappings: List[Dict[str, Any]] = []
    gid = 1
    for subject in SUBJECTS:
        mooc_users = sorted(mooc_grouped[subject], key=lambda x: x["origin_user_id"])
        teddy_users = sorted(teddy_grouped[subject], key=lambda x: (x["card"], x["access_card"]))
        if mooc_users and not teddy_users:
            raise RuntimeError(
                f"泰迪杯 {subject} 有效学生为 0，无法在坚持‘仅同学科配对’时匹配 "
                f"{len(mooc_users)} 名 MOOC 用户。请修订专业覆盖映射或 activity 策略。"
            )
        for index, mooc_user in enumerate(mooc_users):
            teddy_user = teddy_users[index % len(teddy_users)]  # 人数不足时按要求循环复用
            mappings.append({
                "global_user_id": gid,
                "mooc_origin_userid": mooc_user["origin_user_id"],
                "teddy_origin_card": teddy_user["card"],
                "teddy_origin_access_card": teddy_user["access_card"],  # 内部改写门禁表使用
                "subject": subject,
                "mooc": mooc_user,
                "teddy": teddy_user,
            })
            gid += 1
        log(f"    匹配 {subject}：MOOC {len(mooc_users):,}，泰迪可用 {len(teddy_users):,}，映射 {len(mooc_users):,}")
    return mappings


def write_mapping(mapping: Sequence[Dict[str, Any]], output: Path) -> None:
    path = output / "all_global_id_mapping.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["global_user_id", "mooc_origin_userid", "teddy_origin_card", "subject"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in mapping:
            writer.writerow({k: item[k] for k in fields})


def write_sampled_mooc_users(mapping: Sequence[Dict[str, Any]], output: Path) -> Tuple[Set[str], Set[str]]:
    uid_map: Dict[str, int] = {}
    selected_courses: Set[str] = set()
    with (output / "sampled_mooc_user.json").open("w", encoding="utf-8") as handle:
        for item in mapping:
            origin = item["mooc_origin_userid"]
            uid_map[origin] = item["global_user_id"]
            user = dict(item["mooc"]["user"])
            user.pop("id", None)
            user.pop("user_id", None)
            user = {"global_user_id": item["global_user_id"], "subject": item["subject"],
                    "derived_dominant_field": item["mooc"]["dominant_field"], **user}
            for cid in user.get("course_order") or []:
                selected_courses.add(normalize_course_id(cid))
            handle.write(json.dumps(user, ensure_ascii=False) + "\n")
    return set(uid_map), selected_courses


def rewrite_user_problem(path: Path, mapping: Sequence[Dict[str, Any]], output: Path) -> Set[str]:
    uid_map = {m["mooc_origin_userid"]: m["global_user_id"] for m in mapping}
    selected_problems: Set[str] = set()
    written = 0
    with (output / "sampled_user_problem.json").open("w", encoding="utf-8") as handle:
        for _, row in iter_jsonl(path):
            origin = normalized_id(row.get("user_id"))
            gid = uid_map.get(origin)
            if gid is None:
                continue
            row = dict(row)
            row.pop("user_id", None)
            row = {"global_user_id": gid, **row}
            problem = normalize_problem_id(row.get("problem_id"))
            if problem:
                selected_problems.add(problem)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    log(f"[6] sampled_user_problem.json：{written:,} 条；涉及习题 {len(selected_problems):,}")
    return selected_problems


def split_relation(line: str) -> Optional[Tuple[str, str]]:
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 2:
        parts = line.rstrip("\r\n").split(",", 1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) >= 2 else None


def crop_text_relation(source: Path, destination: Path, keep_pair, collect_left: Optional[Set[str]] = None) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig") as inp, destination.open("w", encoding="utf-8") as out:
        for line in inp:
            pair = split_relation(line)
            if not pair or not keep_pair(*pair):
                continue
            out.write(f"{pair[0]}\t{pair[1]}\n")
            if collect_left is not None:
                collect_left.add(pair[0])
            count += 1
    return count


def crop_json_entities(source: Path, destination: Path, selected: Set[str],
                       id_fields: Sequence[str] = ("id",), id_normalizer=normalized_id
                       ) -> Tuple[int, Dict[str, str]]:
    count = 0
    id_to_name: Dict[str, str] = {}
    with destination.open("w", encoding="utf-8") as out:
        for _, row in iter_jsonl(source):
            raw_id = next((row.get(field) for field in id_fields if row.get(field) is not None), None)
            entity_id = id_normalizer(raw_id)
            if entity_id not in selected:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            id_to_name[entity_id] = str(row.get("name") or "")
            count += 1
    return count, id_to_name


def crop_graph(mooc_root: Path, output: Path, selected_courses: Set[str], selected_problems: Set[str]) -> None:
    """裁剪课程、习题、知识点关系和三个先修关系文件；缺失的可选源静默跳过。"""
    relations = mooc_root / "relations"
    entities = mooc_root / "entities"
    relevant_concepts: Set[str] = set()

    cp = relations / "concept-problem.txt"
    if cp.exists():
        n = crop_text_relation(cp, output / "sampled_concept_problem.txt",
                               lambda concept, problem: problem in selected_problems,
                               relevant_concepts)
        log(f"    sampled_concept_problem.txt：{n:,} 条")

    cc = relations / "concept-course.txt"
    if cc.exists():
        n = crop_text_relation(cc, output / "sampled_concept_course.txt",
                               lambda concept, course: normalize_course_id(course) in selected_courses,
                               relevant_concepts)
        log(f"    sampled_concept_course.txt：{n:,} 条")

    for source_name, out_name, selected, id_fields, normalizer in (
        ("course.json", "sampled_course.json", selected_courses, ("id", "course_id"), normalize_course_id),
        # 当前 MOOCCubeX 原文件使用数字 problem_id；部分发布说明写作 id，两者都兼容。
        ("problem.json", "sampled_problem.json", selected_problems, ("id", "problem_id"), normalize_problem_id),
    ):
        source = entities / source_name
        if source.exists():
            n, _ = crop_json_entities(source, output / out_name, selected, id_fields, normalizer)
            log(f"    {out_name}：{n:,} 条")

    concept_names: Set[str] = set()
    concept_source = entities / "concept.json"
    if concept_source.exists():
        n, id_to_name = crop_json_entities(concept_source, output / "sampled_concept.json", relevant_concepts)
        concept_names.update(name for name in id_to_name.values() if name)
        # 原始先修文件 c1/c2 保存概念名称；概念 ID 通常形如 K_名称_领域，故同时解析 ID。
        for cid in relevant_concepts:
            if cid.startswith("K_"):
                concept_names.add(cid[2:].rsplit("_", 1)[0])
        log(f"    sampled_concept.json：{n:,} 条")

    prereq_out = output / "sampled_prerequisites"
    prereq_out.mkdir(exist_ok=True)
    for name in ("cs.json", "math.json", "psy.json"):
        source = mooc_root / "prerequisites" / name
        destination = prereq_out / name
        if not source.exists():
            continue
        kept = 0
        with destination.open("w", encoding="utf-8") as out:
            for _, row in iter_jsonl(source):
                if str(row.get("c1", "")) in concept_names and str(row.get("c2", "")) in concept_names:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        log(f"    sampled_prerequisites/{name}：{kept:,} 条")

    # course-field 文件位置在不同发布包中不同，存在时按课程裁剪。
    course_field = next((p for p in (relations / "course-field.json", mooc_root / "course-field.json") if p.exists()), None)
    if course_field:
        kept = 0
        with (output / "sampled_course_field.json").open("w", encoding="utf-8") as out:
            for _, row in iter_jsonl(course_field):
                cid = normalize_course_id(row.get("course_id") or row.get("id"))
                if cid in selected_courses:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        log(f"    sampled_course_field.json：{kept:,} 条")


def rewrite_csv_with_global(source: Path, destination: Path, lookup_fields: Sequence[str],
                            identity_fields_to_remove: Set[str], id_to_globals: Mapping[str, List[int]]) -> int:
    """循环复用卡号时，一条原记录会为绑定该卡的每个 global_user_id 各输出一份。"""
    encoding = detect_csv_encoding(source)
    written = 0
    errors = "strict" if encoding == "utf-8-sig" else "replace"
    with source.open("r", encoding=encoding, errors=errors, newline="") as inp, destination.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(inp)
        if not reader.fieldnames:
            raise ValueError(f"CSV 无表头：{source}")
        lookup = next((f for f in lookup_fields if f in reader.fieldnames), None)
        if not lookup:
            raise KeyError(f"{source} 找不到关联字段 {list(lookup_fields)}")
        retained = [f for f in reader.fieldnames if f not in identity_fields_to_remove]
        writer = csv.DictWriter(out, fieldnames=["global_user_id", *retained], extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            key = normalized_id(row.get(lookup))
            for gid in id_to_globals.get(key, ()):
                cleaned = {k: v for k, v in row.items() if k not in identity_fields_to_remove}
                writer.writerow({"global_user_id": gid, **cleaned})
                written += 1
    return written


def rewrite_teddy(mapping: Sequence[Dict[str, Any]], student: Path, consume: Path, door: Path, output: Path) -> None:
    card_to_globals: Dict[str, List[int]] = defaultdict(list)
    access_to_globals: Dict[str, List[int]] = defaultdict(list)
    for item in mapping:
        card_to_globals[item["teddy_origin_card"]].append(item["global_user_id"])
        access_to_globals[item["teddy_origin_access_card"]].append(item["global_user_id"])
    student_n = rewrite_csv_with_global(
        student, output / "mapped_teddy_student.csv", ("CardNo", "校园卡号", "卡号"),
        {"CardNo", "校园卡号", "卡号", "AccessCardNo", "门禁卡号"}, card_to_globals)
    consume_n = rewrite_csv_with_global(
        consume, output / "mapped_teddy_consume.csv", ("CardNo", "卡号", "校园卡号"),
        {"CardNo", "卡号", "校园卡号"}, card_to_globals)
    door_n = rewrite_csv_with_global(
        door, output / "mapped_teddy_door.csv", ("AccessCardNo", "门禁卡号", "CardNo", "卡号"),
        {"AccessCardNo", "门禁卡号", "CardNo", "卡号"}, access_to_globals)
    log(f"[7] 泰迪改写：student {student_n:,}，consume {consume_n:,}，door {door_n:,} 条")


def write_stats(output: Path, mooc_available: Mapping[str, int], mooc_sampled: Mapping[str, int],
                teddy_raw: Mapping[str, int], teddy_valid: Mapping[str, int], mapping: Sequence[Dict[str, Any]]) -> None:
    matched = Counter(item["subject"] for item in mapping)
    fields = ["subject", "mooc_qualified_count", "mooc_sampled_count",
              "teddy_raw_count", "teddy_valid_count", "matched_count"]
    with (output / "subject_dist_stat.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject in SUBJECTS:
            writer.writerow({
                "subject": subject,
                "mooc_qualified_count": mooc_available.get(subject, 0),
                "mooc_sampled_count": mooc_sampled.get(subject, 0),
                "teddy_raw_count": teddy_raw.get(subject, 0),
                "teddy_valid_count": teddy_valid.get(subject, 0),
                "matched_count": matched.get(subject, 0),
            })


def write_major_audit(output: Path, major_subject: Mapping[str, str]) -> None:
    with (output / "teddy_major_subject_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["teddy_major", "subject"])
        for major in sorted(major_subject):
            writer.writerow([major, major_subject[major]])


def write_run_metadata(output: Path, mooc_available: Mapping[str, int], mooc_sampled: Mapping[str, int],
                       teddy_raw: Mapping[str, int], teddy_valid: Mapping[str, int]) -> None:
    """保存统计检查点，使断点续跑不必再次扫描 MOOC 全量文件。"""
    data = {
        "mooc_available": {s: int(mooc_available.get(s, 0)) for s in SUBJECTS},
        "mooc_sampled": {s: int(mooc_sampled.get(s, 0)) for s in SUBJECTS},
        "teddy_raw": {s: int(teddy_raw.get(s, 0)) for s in SUBJECTS},
        "teddy_valid": {s: int(teddy_valid.get(s, 0)) for s in SUBJECTS},
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_resume_state(output: Path, student_path: Path
                      ) -> Tuple[List[Dict[str, Any]], Set[str], Set[str], Dict[str, Any]]:
    """从已完成的核心输出恢复，不再读取 5GB 原始答题表和 333 万用户表。"""
    required = [output / "all_global_id_mapping.csv", output / "sampled_mooc_user.json",
                output / "sampled_user_problem.json"]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("无法断点续跑，缺少有效检查点文件：\n  " + "\n  ".join(missing))

    access_by_card: Dict[str, str] = {}
    for row in csv_rows(student_path):
        card = first_present(row, ("CardNo", "校园卡号", "卡号"))
        access_by_card[card] = first_present(row, ("AccessCardNo", "门禁卡号"))

    mapping: List[Dict[str, Any]] = []
    with (output / "all_global_id_mapping.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            card = normalized_id(row.get("teddy_origin_card"))
            access = access_by_card.get(card)
            if not access:
                raise RuntimeError(f"检查点中的泰迪卡号 {card} 无法在 student.csv 中找到门禁卡号")
            mapping.append({
                "global_user_id": int(row["global_user_id"]),
                "mooc_origin_userid": normalized_id(row["mooc_origin_userid"]),
                "teddy_origin_card": card,
                "teddy_origin_access_card": access,
                "subject": row["subject"],
            })

    selected_courses: Set[str] = set()
    for _, row in iter_jsonl(output / "sampled_mooc_user.json"):
        for cid in row.get("course_order") or []:
            selected_courses.add(normalize_course_id(cid))
    selected_problems: Set[str] = set()
    for _, row in iter_jsonl(output / "sampled_user_problem.json"):
        if row.get("problem_id") is not None:
            selected_problems.add(normalize_problem_id(row["problem_id"]))

    metadata: Dict[str, Any] = {}
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8-sig") as handle:
            metadata = json.load(handle)
    log(f"[断点] 已恢复映射 {len(mapping):,} 条、课程 {len(selected_courses):,}、习题 {len(selected_problems):,}")
    return mapping, selected_courses, selected_problems, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按八大学科对齐 MOOCCubeX 与泰迪杯一卡通数据")
    parser.add_argument("--mooc-root", type=Path, default=Path("MOOCCubeX/MOOCCubeX-main"))
    parser.add_argument("--teddy-root", type=Path, default=Path("dd0f7-main"))
    parser.add_argument("--student", default="data1.csv", help="相对 teddy-root 的学生 CSV")
    parser.add_argument("--consume", default="data2.csv", help="相对 teddy-root 的消费 CSV")
    parser.add_argument("--door", default="data3.csv", help="相对 teddy-root 的门禁 CSV")
    parser.add_argument("--output", type=Path, default=Path("aligned_output"))
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--min-problems", type=int, default=20)
    parser.add_argument("--min-courses", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--teddy-activity", choices=("any", "both"), default="any",
                        help="any=消费/门禁至少一种，剔除两者皆无的僵尸学生（默认）；both=两者都必须有")
    parser.add_argument("--major-map-override", type=Path,
                        help="可选 JSON：{\"细分专业\": \"八大学科\"}")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录")
    parser.add_argument("--resume", action="store_true",
                        help="从输出目录中的映射、抽样用户和答题文件继续，跳过全量 MOOC 扫描")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    mooc = args.mooc_root.resolve()
    teddy = args.teddy_root.resolve()
    student, consume, door = (teddy / args.student, teddy / args.consume, teddy / args.door)
    required = [mooc / "entities" / "user.json", mooc / "entities" / "course.json",
                mooc / "relations" / "user-problem.json", student, consume, door]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n  " + "\n  ".join(missing))
    if args.target <= 0 or args.min_problems <= 0 or args.min_courses <= 0:
        raise ValueError("target/min-problems/min-courses 必须为正整数")
    return mooc, student, consume, door


def prepare_output(path: Path, overwrite: bool, resume: bool) -> Path:
    path = path.resolve()
    if overwrite and resume:
        raise ValueError("--overwrite 与 --resume 不能同时使用")
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"断点输出目录不存在：{path}")
        return path
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空：{path}；如需覆盖请添加 --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    mooc, student, consume, door = validate_paths(args)
    output = prepare_output(args.output, args.overwrite, args.resume)
    override = load_override(args.major_map_override)
    log("=" * 72)
    log("MOOCCubeX × 泰迪杯：八大学科分层模拟匹配")
    log(f"输出目录：{output}")
    log("=" * 72)

    if args.resume:
        mapping, selected_courses, selected_problems, metadata = load_resume_state(output, student)
        if len(mapping) != args.target:
            raise RuntimeError(f"检查点映射数量 {len(mapping)} 与目标 {args.target} 不一致")
        crop_graph(mooc, output, selected_courses, selected_problems)
        consume_cards, door_cards, _, _ = collect_activity_ids(consume, door)
        _, teddy_raw, teddy_valid, major_subject = collect_teddy_students(
            student, consume_cards, door_cards, override, args.teddy_activity)
        rewrite_teddy(mapping, student, consume, door, output)
        sampled_count = Counter(item["subject"] for item in mapping)
        mooc_available = metadata.get("mooc_available", sampled_count)
        mooc_sampled = metadata.get("mooc_sampled", sampled_count)
        write_stats(output, mooc_available, mooc_sampled, teddy_raw, teddy_valid, mapping)
        write_major_audit(output, major_subject)
        write_run_metadata(output, mooc_available, mooc_sampled, teddy_raw, teddy_valid)
        log("=" * 72)
        log(f"断点续跑完成：最终全局映射 {len(mapping):,} 条，global_user_id 范围 1..{len(mapping)}")
        return 0

    course_fields, _ = load_course_fields(mooc)
    active_users, _, _ = count_active_users(mooc / "relations" / "user-problem.json", args.min_problems)
    reservoirs, mooc_available, _ = collect_mooc_candidates(
        mooc / "entities" / "user.json", active_users, course_fields,
        args.min_courses, args.target, args.seed)
    # 活跃集合此后不再使用，及时释放大集合。
    del active_users
    sampled, mooc_quotas = sample_mooc(reservoirs, mooc_available, args.target, args.seed)

    consume_cards, door_cards, _, _ = collect_activity_ids(consume, door)
    teddy_grouped, teddy_raw, teddy_valid, major_subject = collect_teddy_students(
        student, consume_cards, door_cards, override, args.teddy_activity)
    mapping = create_global_mapping(sampled, teddy_grouped)
    if len(mapping) != args.target:
        raise AssertionError(f"最终映射不是目标数量：{len(mapping)} != {args.target}")

    write_mapping(mapping, output)
    # 在进入耗时图谱裁剪前写入统计检查点；之后失败可直接 --resume。
    write_run_metadata(output, mooc_available, mooc_quotas, teddy_raw, teddy_valid)
    _, selected_courses = write_sampled_mooc_users(mapping, output)
    selected_problems = rewrite_user_problem(mooc / "relations" / "user-problem.json", mapping, output)
    crop_graph(mooc, output, selected_courses, selected_problems)
    rewrite_teddy(mapping, student, consume, door, output)
    write_stats(output, mooc_available, mooc_quotas, teddy_raw, teddy_valid, mapping)
    write_major_audit(output, major_subject)

    log("=" * 72)
    log(f"完成：最终全局映射 {len(mapping):,} 条，global_user_id 范围 1..{len(mapping)}")
    log("注意：该 ID 是学科约束下的模拟配对键，不是真实身份关联。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, RuntimeError) as exc:
        print(f"\n[失败] {exc}", file=sys.stderr)
        raise SystemExit(2)
