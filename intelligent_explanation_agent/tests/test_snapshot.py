from types import SimpleNamespace
import unittest

from intelligent_explanation_agent.snapshot import build_snapshot


class SnapshotTest(unittest.TestCase):
    def test_recent_rows_stay_aligned(self):
        sample = SimpleNamespace(
            user_id=7, subject="cs", gender="unknown", grade="2026",
            questions=["q1", "q2", "q3"], concepts=["c1", "c2", "c3"],
            correct=[0, 1, 1], intervals=[0, 2, 3], night_study=[0, 0, 1],
            wrong_streak=[1, 0, 0], consume_items=["a", "b"], consume_meal=[1, 2],
            consume_hour=[8, 12], consume_weekday=[1, 2], consume_holiday=[0, 0],
            consume_age_hours=[4, 0], door_items=["d1"], door_hour=[23],
            door_weekday=[2], door_holiday=[0], door_late=[1], door_age_hours=[0])
        snapshot = build_snapshot(sample, recent_limit=2)
        self.assertEqual([row["question"] for row in snapshot.learning], ["q2", "q3"])
        self.assertEqual(snapshot.summary["recent_learning_accuracy"], 1.0)
        self.assertEqual(snapshot.summary["late_access_count"], 1)


if __name__ == "__main__":
    unittest.main()
