"""Pipeline tests."""

from src.pipeline import MentalHealthPipeline


def test_pipeline_crisis_priority() -> None:
    result = MentalHealthPipeline().process("帮我记录心情：我想伤害自己，9分")
    assert result["is_crisis"] is True
    assert result["tool_used"] == "crisis_resource"


def test_pipeline_invalid_input() -> None:
    result = MentalHealthPipeline().process("")
    assert result["intent"] == "invalid"

