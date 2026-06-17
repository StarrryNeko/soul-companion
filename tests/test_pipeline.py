"""Pipeline tests."""

from src.pipeline import MentalHealthPipeline


def test_pipeline_auto_records_emotion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOUL_LOAD_LOCAL_MODEL", "0")
    pipeline = MentalHealthPipeline()
    pipeline.tools["tool_emotion_log"].log_path = tmp_path / "emotion_logs.json"
    result = pipeline.process("我今天有点焦虑，强度7分，快考试了")
    assert result["auto_emotion_record"]["emotion"] == "焦虑"
    assert pipeline.tools["tool_emotion_log"].get_history()[0]["intensity"] == 7


def test_pipeline_crisis_priority(monkeypatch) -> None:
    monkeypatch.setenv("SOUL_LOAD_LOCAL_MODEL", "0")
    result = MentalHealthPipeline().process("帮我记录心情：我想伤害自己，9分")
    assert result["is_crisis"] is True
    assert result["tool_used"] == "crisis_resource"


def test_pipeline_invalid_input(monkeypatch) -> None:
    monkeypatch.setenv("SOUL_LOAD_LOCAL_MODEL", "0")
    result = MentalHealthPipeline().process("")
    assert result["intent"] == "invalid"
