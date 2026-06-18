"""Safety detector tests."""

import numpy as np

from src.safety.crisis_detector import CrisisDetector


class FakeSemanticModel:
    """Small deterministic encoder that keeps tests offline."""

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            if any(marker in text for marker in ("结束自己的生命", "伤害自己", "不应该继续活", "永远离开")):
                rows.append([1.0, 0.0, 0.0])
            elif any(marker in text for marker in ("撑不下去", "希望", "消失", "生活的力气")):
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_crisis_keyword_high_risk() -> None:
    detector = CrisisDetector(enable_semantic=False)
    result = detector.check("我不想活了")
    assert result["is_crisis"] is True
    assert result["matched_keyword"] == "不想活了"


def test_non_crisis_input() -> None:
    detector = CrisisDetector(embedding_model=FakeSemanticModel())
    result = detector.check("最近考试压力有点大")
    assert result["is_crisis"] is False


def test_semantic_high_risk_expression() -> None:
    detector = CrisisDetector(embedding_model=FakeSemanticModel())

    result = detector.check("我已经决定永远离开，不希望明天再醒来")

    assert result["is_crisis"] is True
    assert result["trigger_type"] == "semantic"
    assert result["risk_level"] == "high"
    assert result["similarity_score"] >= 0.85


def test_semantic_model_failure_falls_back_to_keyword_detection() -> None:
    class FailingSemanticModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    detector = CrisisDetector(embedding_model=FailingSemanticModel())

    result = detector.check("最近考试压力有点大")

    assert result["is_crisis"] is False
    assert "model unavailable" in detector.last_semantic_error
