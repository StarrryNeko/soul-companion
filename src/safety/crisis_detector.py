"""Crisis signal detection."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import SAFETY_CONFIG


class CrisisDetector:
    """Keyword-first crisis detector for classroom demonstration."""

    def __init__(self, keywords_path: str | None = None, threshold: float | None = None) -> None:
        self.keywords_path = Path(keywords_path or SAFETY_CONFIG["keywords_path"])
        self.threshold = threshold or SAFETY_CONFIG["semantic_threshold"]
        payload = json.loads(self.keywords_path.read_text(encoding="utf-8"))
        self.high_risk = payload.get("high_risk", [])
        self.medium_risk = payload.get("medium_risk", [])

    def check(self, user_input: str) -> dict:
        text = user_input.strip().lower()
        for keyword in self.high_risk:
            if keyword.lower() in text:
                return self._hit("keyword", keyword, 1.0, "high")
        for keyword in self.medium_risk:
            if keyword.lower() in text:
                return self._hit("keyword", keyword, 0.9, "medium")
        return {
            "is_crisis": False,
            "trigger_type": None,
            "matched_keyword": None,
            "similarity_score": None,
            "risk_level": None,
        }

    @staticmethod
    def _hit(trigger_type: str, keyword: str, score: float, risk_level: str) -> dict:
        return {
            "is_crisis": True,
            "trigger_type": trigger_type,
            "matched_keyword": keyword,
            "similarity_score": score,
            "risk_level": risk_level,
        }

