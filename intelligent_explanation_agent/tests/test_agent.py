from types import SimpleNamespace
import unittest

from intelligent_explanation_agent.agent import ExplainableRecommendationAgent


def sample():
    return SimpleNamespace(
        user_id=7, subject="cs", gender="unknown", grade="2026",
        questions=["q1"], concepts=["c1"], correct=[1], intervals=[0],
        night_study=[0], wrong_streak=[0], consume_items=["food"], consume_meal=[2],
        consume_hour=[12], consume_weekday=[1], consume_holiday=[0], consume_age_hours=[0],
        door_items=["library"], door_hour=[18], door_weekday=[1], door_holiday=[0],
        door_late=[0], door_age_hours=[0])


class FakeRecommender:
    def get_user_sample(self, user_id):
        return sample()

    def recommend(self, user_sample):
        return {"user_id": 7, "knowledge": [{"item": "k1", "score": .8}],
                "course": [{"item": "c1", "score": .7}],
                "consume": [{"item": "x1", "score": .6}]}


class FakeLLM:
    def explain(self, snapshot, recommendations):
        return {"user_summary": "summary", "recommendation_explanations": {
            "knowledge": [{"item": "invented", "reason": "bad"},
                          {"item": "k1", "reason": "supported", "evidence": ["c1"]}],
            "course": [], "consume": []}}


class AgentTest(unittest.TestCase):
    def test_api_cannot_change_recommendation_set_or_order(self):
        result = ExplainableRecommendationAgent(FakeRecommender(), FakeLLM()).run(7)
        explanations = result["explanation"]["recommendation_explanations"]
        self.assertEqual([row["item"] for row in explanations["knowledge"]], ["k1"])
        self.assertEqual(explanations["knowledge"][0]["reason"], "supported")
        self.assertEqual(explanations["course"], [])


if __name__ == "__main__":
    unittest.main()
