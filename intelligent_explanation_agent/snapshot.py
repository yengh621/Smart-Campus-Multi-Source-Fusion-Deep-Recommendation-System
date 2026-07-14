from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass
class UserSnapshot:
    user_id: int
    profile: dict[str, str]
    learning: list[dict[str, Any]]
    consumption: list[dict[str, Any]]
    access: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self, include_user_id: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_user_id:
            payload["user_id"] = "REDACTED"
        return payload


def _tail_rows(columns: dict[str, Iterable[Any]], limit: int) -> list[dict[str, Any]]:
    values = {name: list(items) for name, items in columns.items()}
    size = min((len(items) for items in values.values()), default=0)
    start = max(0, size - limit)
    return [{name: items[index] for name, items in values.items()}
            for index in range(start, size)]


def build_snapshot(sample: Any, recent_limit: int = 10) -> UserSnapshot:
    """从推荐器实际使用的 UserSample 构造三个模态的近期、可审计快照。"""
    learning = _tail_rows({
        "question": sample.questions,
        "concept": sample.concepts,
        "correct": sample.correct,
        "interval_hours": sample.intervals,
        "night_study": sample.night_study,
        "wrong_streak": sample.wrong_streak,
    }, recent_limit)
    consumption = _tail_rows({
        "category": sample.consume_items,
        "meal_period": sample.consume_meal,
        "hour": sample.consume_hour,
        "weekday": sample.consume_weekday,
        "holiday": sample.consume_holiday,
        "age_hours": sample.consume_age_hours,
    }, recent_limit)
    access = _tail_rows({
        "location": sample.door_items,
        "hour": sample.door_hour,
        "weekday": sample.door_weekday,
        "holiday": sample.door_holiday,
        "late_night": sample.door_late,
        "age_hours": sample.door_age_hours,
    }, recent_limit)
    accuracy = (sum(float(row["correct"]) for row in learning) / len(learning)
                if learning else None)
    return UserSnapshot(
        user_id=int(sample.user_id),
        profile={"subject": sample.subject, "gender": sample.gender, "grade": sample.grade},
        learning=learning,
        consumption=consumption,
        access=access,
        summary={
            "recent_limit_per_modality": recent_limit,
            "learning_count": len(learning),
            "recent_learning_accuracy": accuracy,
            "consumption_count": len(consumption),
            "access_count": len(access),
            "late_access_count": sum(bool(row["late_night"]) for row in access),
        },
    )
