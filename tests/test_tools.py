"""Tool tests."""

from src.tools.breathing_exercise import BreathingExerciseTool
from src.tools.emotion_logger import EmotionLoggerTool


def test_breathing_tool() -> None:
    result = BreathingExerciseTool().execute(user_input="我很紧张")
    assert result["success"] is True
    assert "方块呼吸" in result["message"]


def test_emotion_logger_tool(tmp_path) -> None:
    tool = EmotionLoggerTool(log_path=str(tmp_path / "emotion_logs.json"))
    parsed = tool.parse("帮我记录心情：焦虑，7分")
    result = tool.execute(**parsed)
    assert result["success"] is True
    assert tool.get_history()[0]["emotion"] == "焦虑"


def test_emotion_logger_auto_record_detection() -> None:
    tool = EmotionLoggerTool()
    assert tool.should_auto_record("我今天有点低落，强度6分", "mental_health") is True
    assert tool.parse("我今天有点低落，强度6分")["intensity"] == 6
    assert tool.should_auto_record("翻译这句话：今天心情不太好", "general") is False


def test_breathing_tool_has_more_relaxation_material() -> None:
    result = BreathingExerciseTool().execute(user_input="我最近压力很大，想放松一下")
    assert result["success"] is True
    assert "肩颈卸力练习" in result["message"]
