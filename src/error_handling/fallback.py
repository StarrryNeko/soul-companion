"""Fallback responses."""


class FallbackHandler:
    """Centralized friendly fallback messages."""

    def handle_model_failure(self, user_input: str) -> str:
        return "当前模型暂时不可用。你可以稍后重试；如果正在经历明显痛苦，请优先联系现实中的可信任支持者或学校心理咨询中心。"

    def handle_rag_empty(self) -> str:
        return "我没有在知识库中检索到足够可靠的资料，下面只提供一般性信息，不作为专业建议。"

    def handle_tool_failure(self, tool_name: str, error: Exception) -> str:
        return f"{tool_name} 工具暂时不可用，请稍后重试。"

    def handle_crisis_fallback(self) -> str:
        return (
            "我担心你现在可能处在高风险状态。请立即联系身边可信任的人、学校心理咨询中心或当地紧急救助电话。"
            "如果存在即时危险，请马上拨打当地紧急电话或前往安全地点。"
        )

    def handle_invalid_input(self, input_type: str) -> str:
        if input_type == "too_long":
            return "这段内容有点长，请尽量分成几段输入，我会更容易理解和回应。"
        return "你好像还没有输入内容。你可以告诉我最近的情绪、压力来源，或直接问一个心理健康相关问题。"

