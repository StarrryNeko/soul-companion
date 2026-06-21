"""Crisis resource tool."""

import json
from pathlib import Path

from config.settings import SAFETY_CONFIG
from src.tools.base import BaseTool


class CrisisResourceTool(BaseTool):
    """Return a fixed crisis-support response without model generation."""

    @property
    def name(self) -> str:
        return "crisis_resource"

    @property
    def description(self) -> str:
        return "检测到危机信号时，返回固定安全转介话术。"

    def execute(self, **kwargs) -> dict:
        """从安全配置读取固定危机回复，避免高风险场景依赖模型生成。"""
        path = Path(SAFETY_CONFIG["keywords_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "success": True,
            "message": payload["response_template"],
            "data": {
                "resource_type": "demo_placeholder",
                "note": "真实部署前请替换为学校或当地官方资源。",
            },
        }
