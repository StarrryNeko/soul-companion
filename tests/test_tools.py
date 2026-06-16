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

