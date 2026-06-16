"""Breathing and relaxation exercise tool."""

from src.tools.base import BaseTool


class BreathingExerciseTool(BaseTool):
    """Recommend a relaxation exercise."""

    EXERCISES = {
        "焦虑": {
            "name": "4-6 缓慢呼吸",
            "steps": ["吸气4秒", "呼气6秒", "重复5轮", "把注意力放在呼气变长的感觉上"],
        },
        "紧张": {
            "name": "方块呼吸",
            "steps": ["吸气4秒", "屏息4秒", "呼气4秒", "停顿4秒", "重复4轮"],
        },
        "失眠": {
            "name": "渐进式肌肉放松",
            "steps": ["从脚趾开始轻轻绷紧5秒", "慢慢放松10秒", "依次移动到小腿、大腿、肩膀和手臂"],
        },
        "愤怒": {
            "name": "暂停加长呼气",
            "steps": ["离开冲突现场一分钟", "吸气4秒", "呼气8秒", "重复5轮后再决定是否沟通"],
        },
        "default": {
            "name": "腹式呼吸",
            "steps": ["一只手放在腹部", "吸气时腹部轻轻鼓起", "呼气时自然回落", "练习5轮"],
        },
    }

    @property
    def name(self) -> str:
        return "breathing_exercise"

    @property
    def description(self) -> str:
        return "根据用户状态推荐呼吸或放松练习。"

    def execute(self, emotion_type: str = "default", user_input: str = "", **kwargs) -> dict:
        key = emotion_type
        for candidate in self.EXERCISES:
            if candidate != "default" and candidate in user_input:
                key = candidate
                break
        exercise = self.EXERCISES.get(key, self.EXERCISES["default"])
        steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(exercise["steps"], start=1))
        return {
            "success": True,
            "message": f"可以试试「{exercise['name']}」：\n{steps}\n\n如果练习中不舒服，请恢复自然呼吸。",
            "data": exercise,
        }

