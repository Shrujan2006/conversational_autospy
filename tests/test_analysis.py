import unittest
import pandas as pd

from src.analysis_engine import parse_conversation_text
from src.conflict_detector import analyze_conflicts
from src.explainability import locate_breakdown


class ConversationAnalysisTests(unittest.TestCase):
    def test_parse_valid_conversation(self):
        self.assertEqual(
            parse_conversation_text("A: hello\nB: hi: there"),
            [("A", "hello"), ("B", "hi: there")]
        )

    def test_parse_rejects_malformed_line(self):
        with self.assertRaises(ValueError):
            parse_conversation_text("A: hello\nmissing separator")

    def test_disagreement_alone_is_not_conflict(self):
        frame = pd.DataFrame([{
            "Speaker": "A", "Emotion": "neutral",
            "Dialogue Act": "Disagreement", "Emotion Changed": False
        }])
        result = analyze_conflicts(frame)
        self.assertEqual(result.iloc[0]["Conversation State"], "Normal")

    def test_negative_emotion_alone_is_not_escalation(self):
        frame = pd.DataFrame([{
            "Speaker": "A", "Emotion": "anger",
            "Dialogue Act": "Statement", "Emotion Changed": False
        }])
        result = analyze_conflicts(frame)
        self.assertEqual(result.iloc[0]["Conversation State"], "Normal")

    def test_reciprocal_accusation_can_escalate(self):
        frame = pd.DataFrame([
            {"Speaker": "A", "Emotion": "anger", "Dialogue Act": "Accusation", "Emotion Changed": False},
            {"Speaker": "B", "Emotion": "anger", "Dialogue Act": "Accusation", "Emotion Changed": True}
        ])
        result = analyze_conflicts(frame)
        self.assertEqual(result.iloc[1]["Conversation State"], "Escalation")

    def test_breakdown_status_prefers_escalation(self):
        frame = pd.DataFrame([
            {"Turn": 1, "Conversation State": "Tension"},
            {"Turn": 2, "Conversation State": "Escalation"}
        ])
        result = locate_breakdown(frame)
        self.assertEqual(result["Breakdown Status"], "Escalation")
        self.assertEqual(result["First Tension Turn"], 1)
        self.assertEqual(result["First Escalation Turn"], 2)


if __name__ == "__main__":
    unittest.main()
