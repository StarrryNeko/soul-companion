"""Intent classifier tests."""

from src.intent.classifier import IntentClassifier


def test_emotion_log_intent() -> None:
    assert IntentClassifier().classify("帮我记录一下今天的心情：焦虑7分") == "tool_emotion_log"


def test_breathing_intent() -> None:
    assert IntentClassifier().classify("我现在很紧张，推荐一个呼吸练习") == "tool_breathing"


def test_general_intent() -> None:
    assert IntentClassifier().classify("帮我写一段 Python 排序代码") == "general"

