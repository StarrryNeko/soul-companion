"""Core orchestration pipeline."""

from __future__ import annotations

from src.error_handling.fallback import FallbackHandler
from src.intent.classifier import IntentClassifier
from src.model.generator import ResponseGenerator
from src.rag.retriever import KnowledgeRetriever
from src.safety.crisis_detector import CrisisDetector
from src.tools.breathing_exercise import BreathingExerciseTool
from src.tools.crisis_resource import CrisisResourceTool
from src.tools.emotion_logger import EmotionLoggerTool


class MentalHealthPipeline:
    """Route user input through safety, intent, tools, RAG, and generation."""

    def __init__(self) -> None:
        self.fallback = FallbackHandler()
        self.crisis_detector = CrisisDetector()
        self.intent_classifier = IntentClassifier()
        self.retriever = KnowledgeRetriever()
        self.generator = ResponseGenerator()
        self.tools = {
            "crisis_resource": CrisisResourceTool(),
            "tool_emotion_log": EmotionLoggerTool(),
            "tool_breathing": BreathingExerciseTool(),
        }

    def process(self, user_input: str, chat_history: list | None = None) -> dict:
        text = (user_input or "").strip()
        if not text:
            return self._result(self.fallback.handle_invalid_input("empty"), "invalid")
        if len(text) > 2000:
            return self._result(self.fallback.handle_invalid_input("too_long"), "invalid")

        try:
            safety = self.crisis_detector.check(text)
        except Exception:
            return self._result(
                self.fallback.handle_crisis_fallback(),
                "crisis",
                is_crisis=True,
                tool_used="crisis_resource",
            )

        if safety["is_crisis"]:
            tool_result = self.tools["crisis_resource"].execute()
            return self._result(
                tool_result["message"],
                "crisis",
                is_crisis=True,
                tool_used="crisis_resource",
            )

        intent = self.intent_classifier.classify(text)

        if intent == "tool_emotion_log":
            try:
                parsed = self.tools[intent].parse(text)
                tool_result = self.tools[intent].execute(**parsed)
                return self._result(tool_result["message"], intent, tool_used="emotion_logger")
            except Exception as exc:
                return self._result(self.fallback.handle_tool_failure("emotion_logger", exc), intent)

        if intent == "tool_breathing":
            try:
                tool_result = self.tools[intent].execute(user_input=text)
                return self._result(tool_result["message"], intent, tool_used="breathing_exercise")
            except Exception as exc:
                return self._result(self.fallback.handle_tool_failure("breathing_exercise", exc), intent)

        sources = []
        if intent == "mental_health":
            sources = self.retriever.retrieve(text)

        response = self.generator.generate(
            text,
            retrieved_context=[item["content"] for item in sources],
            chat_history=chat_history,
        )
        return self._result(response, intent, rag_sources=sources)

    @staticmethod
    def _result(
        response: str,
        intent: str,
        is_crisis: bool = False,
        tool_used: str | None = None,
        rag_sources: list | None = None,
        error: str | None = None,
    ) -> dict:
        return {
            "response": response,
            "intent": intent,
            "is_crisis": is_crisis,
            "tool_used": tool_used,
            "rag_sources": rag_sources or [],
            "error": error,
        }

