"""Emotion logging tool."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from config.settings import TOOLS_CONFIG
from src.tools.base import BaseTool


class EmotionLoggerTool(BaseTool):
    """Record simple emotion entries into a local JSON file."""

    EMOTION_ALIASES = {
        "焦虑": ["焦虑", "担心", "害怕", "不安", "慌", "心慌"],
        "紧张": ["紧张", "压力大", "有压力", "压抑", "绷紧"],
        "低落": ["低落", "沮丧", "没精神", "提不起劲", "郁闷"],
        "难过": ["难过", "伤心", "委屈", "想哭"],
        "开心": ["开心", "高兴", "愉快", "快乐", "满足"],
        "愤怒": ["愤怒", "生气", "气愤", "恼火", "火大"],
        "烦躁": ["烦躁", "烦", "烦闷", "急躁"],
        "平静": ["平静", "放松", "安稳", "踏实"],
        "孤独": ["孤独", "孤单", "没人理解"],
        "压力": ["压力", "撑不住", "累", "疲惫"],
    }
    AUTO_RECORD_CUES = [
        "我",
        "最近",
        "今天",
        "现在",
        "这几天",
        "这段时间",
        "一直",
        "有点",
        "很",
        "特别",
        "感觉",
        "觉得",
    ]
    NON_PERSONAL_CUES = ["翻译", "造句", "写一段", "解释这个词", "这句话", "代码", "Python", "python"]

    @property
    def name(self) -> str:
        return "emotion_logger"

    @property
    def description(self) -> str:
        return "记录用户当前情绪、强度和备注。"

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or TOOLS_CONFIG["emotion_log_path"])

    def parse(self, user_input: str) -> dict:
        emotion = self.detect_emotion(user_input) or "未标注"
        match = re.search(r"(10|[1-9])\s*(?:分|级|/10)", user_input)
        if match is None:
            match = re.search(r"(?:强度|程度|评分|打分)\D{0,3}(10|[1-9])", user_input)
        intensity = int(match.group(1)) if match else 5
        note = user_input.strip()
        return {"emotion": emotion, "intensity": intensity, "note": note}

    def detect_emotion(self, user_input: str) -> str | None:
        for emotion, keywords in self.EMOTION_ALIASES.items():
            if any(keyword in user_input for keyword in keywords):
                return emotion
        return None

    def should_auto_record(self, user_input: str, intent: str | None = None) -> bool:
        text = user_input.strip()
        if not text or any(cue in text for cue in self.NON_PERSONAL_CUES):
            return False
        if self.detect_emotion(text) is None:
            return False
        if intent in {"mental_health", "tool_breathing"}:
            return True
        return any(cue in text for cue in self.AUTO_RECORD_CUES)

    def execute(self, emotion: str, intensity: int = 5, note: str = "") -> dict:
        intensity = max(1, min(int(intensity), 10))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": []}
        if self.log_path.exists():
            payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "emotion": emotion,
            "intensity": intensity,
            "note": note,
        }
        payload.setdefault("records", []).append(record)
        self.log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "success": True,
            "message": f"已记录：情绪为{emotion}，强度为{intensity}分。",
            "data": record,
        }

    def get_history(self, days: int = 7) -> list[dict]:
        if not self.log_path.exists():
            return []
        payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        return payload.get("records", [])[-100:]
