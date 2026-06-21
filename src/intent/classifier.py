"""Rule-based intent classifier."""


class IntentClassifier:
    """Classify input into mental-health, tools, or general routes."""

    emotion_log_keywords = ["记录心情", "记录情绪", "心情记录", "情绪记录", "帮我记录"]
    breathing_keywords = ["呼吸", "放松", "紧张", "冥想", "腹式呼吸", "方块呼吸"]
    mental_health_keywords = [
        "焦虑",
        "抑郁",
        "压力",
        "失眠",
        "睡不着",
        "情绪",
        "人际",
        "室友",
        "拖延",
        "自卑",
        "孤独",
        "心理",
        "咨询",
        "难过",
        "烦躁",
        "低落",
        "崩溃",
        "内耗",
        "害怕",
        "担心",
    ]

    def classify(self, user_input: str) -> str:
        """根据关键词将输入路由到情绪记录、呼吸练习、心理健康或通用对话。"""
        text = user_input.strip()
        if any(keyword in text for keyword in self.emotion_log_keywords):
            return "tool_emotion_log"
        if any(keyword in text for keyword in self.breathing_keywords):
            return "tool_breathing"
        if any(keyword in text for keyword in self.mental_health_keywords):
            return "mental_health"
        return "general"
