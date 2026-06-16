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

    @property
    def name(self) -> str:
        return "emotion_logger"

    @property
    def description(self) -> str:
        return "记录用户当前情绪、强度和备注。"

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or TOOLS_CONFIG["emotion_log_path"])

    def parse(self, user_input: str) -> dict:
        emotions = ["焦虑", "紧张", "低落", "难过", "开心", "愤怒", "烦躁", "平静", "孤独", "压力"]
        emotion = next((item for item in emotions if item in user_input), "未标注")
        match = re.search(r"([1-9]|10)\s*分?", user_input)
        intensity = int(match.group(1)) if match else 5
        note = user_input.strip()
        return {"emotion": emotion, "intensity": intensity, "note": note}

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

