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
        "低落": {
            "name": "五感唤醒练习",
            "steps": ["说出眼前3种颜色", "摸一摸身边物体的质地", "缓慢站起并伸展肩颈", "选一个两分钟内能完成的小行动"],
        },
        "压力": {
            "name": "肩颈卸力练习",
            "steps": ["双脚踩地坐稳", "吸气时轻轻耸肩到耳边", "呼气时让肩膀自然落下", "重复8轮并放松下颌"],
        },
        "烦躁": {
            "name": "数息降速练习",
            "steps": ["自然吸气", "呼气时在心里数1", "数到10后重新从1开始", "走神时温和地回到下一次呼气"],
        },
        "孤独": {
            "name": "自我安抚练习",
            "steps": ["把一只手放在胸口或手臂上", "用平稳语气对自己说一句支持的话", "慢慢呼气6秒", "想一个今天可以联系的安全对象"],
        },
        "惊恐": {
            "name": "5-4-3-2-1落地练习",
            "steps": ["说出看到的5样东西", "感受摸到的4种触感", "听见3种声音", "注意2种气味和1种味道"],
        },
        "考试": {
            "name": "考前落地练习",
            "steps": ["双脚踩实地面", "吸气4秒、呼气6秒，重复5轮", "在心里说出接下来最重要的三件小事"],
        },
        "default": {
            "name": "腹式呼吸",
            "steps": ["一只手放在腹部", "吸气时腹部轻轻鼓起", "呼气时自然回落", "练习5轮"],
        },
    }
    KEYWORD_TO_EXERCISE = {
        "担心": "焦虑",
        "害怕": "焦虑",
        "心慌": "惊恐",
        "胸闷": "惊恐",
        "低落": "低落",
        "沮丧": "低落",
        "难过": "低落",
        "压力": "压力",
        "疲惫": "压力",
        "烦躁": "烦躁",
        "孤独": "孤独",
        "孤单": "孤独",
        "考试": "考试",
        "答辩": "考试",
    }

    @property
    def name(self) -> str:
        return "breathing_exercise"

    @property
    def description(self) -> str:
        return "根据用户状态推荐呼吸或放松练习。"

    def execute(self, emotion_type: str = "default", user_input: str = "", **kwargs) -> dict:
        """根据指定情绪或输入关键词，返回匹配的呼吸/落地练习步骤。"""
        key = emotion_type
        for candidate in self.EXERCISES:
            if candidate != "default" and candidate in user_input:
                key = candidate
                break
        else:
            for keyword, exercise_key in self.KEYWORD_TO_EXERCISE.items():
                if keyword in user_input:
                    key = exercise_key
                    break
        exercise = self.EXERCISES.get(key, self.EXERCISES["default"])
        steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(exercise["steps"], start=1))
        return {
            "success": True,
            "message": f"可以试试「{exercise['name']}」：\n{steps}\n\n如果练习中不舒服，请恢复自然呼吸。",
            "data": exercise,
        }
