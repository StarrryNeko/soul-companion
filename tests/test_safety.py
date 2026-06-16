"""Safety detector tests."""

from src.safety.crisis_detector import CrisisDetector


def test_crisis_keyword_high_risk() -> None:
    detector = CrisisDetector()
    result = detector.check("我不想活了")
    assert result["is_crisis"] is True
    assert result["matched_keyword"] == "不想活了"


def test_non_crisis_input() -> None:
    detector = CrisisDetector()
    result = detector.check("最近考试压力有点大")
    assert result["is_crisis"] is False

