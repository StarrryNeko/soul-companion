"""Core orchestration pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from config.settings import MODEL_CONFIG
from src.error_handling.fallback import FallbackHandler
from src.model.external_api import DeepSeekFallbackClient
from src.intent.classifier import IntentClassifier
from src.model.generator import ResponseGenerator
from src.model.loader import ModelLoader
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
        self.model_info = {
            "backend": "fallback",
            "model_path": "",
            "model_kind": "none",
            "load_quantization": "none",
            "lora_adapter_path": "",
        }
        model, tokenizer = self._load_local_model()
        self.generator = ResponseGenerator(model=model, tokenizer=tokenizer, fallback_client=DeepSeekFallbackClient())
        self.tools = {
            "crisis_resource": CrisisResourceTool(),
            "tool_emotion_log": EmotionLoggerTool(),
            "tool_breathing": BreathingExerciseTool(),
        }

    def _load_local_model(self):
        if os.getenv("SOUL_LOAD_LOCAL_MODEL", "1") != "1":
            return None, None

        merged_model_path = Path(MODEL_CONFIG["merged_model_path"])
        if not merged_model_path.exists():
            return None, None

        try:
            loader = ModelLoader(merged_model_path=str(merged_model_path), lora_adapter_path=None)
            model, tokenizer = loader.load()
            self.model_info = {"backend": "local_model", **loader.last_model_info}
            return model, tokenizer
        except Exception as exc:
            print(f"Local merged model load failed, using fallback backend: {exc}")
            return None, None

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
                self._auto_record_emotion(text, intent)
                tool_result = self.tools[intent].execute(user_input=text)
                return self._result(tool_result["message"], intent, tool_used="breathing_exercise")
            except Exception as exc:
                return self._result(self.fallback.handle_tool_failure("breathing_exercise", exc), intent)

        sources = self._retrieve_context(text)
        auto_emotion_record = self._auto_record_emotion(text, intent)

        response = self.generator.generate(
            text,
            retrieved_context=[item["content"] for item in sources],
            chat_history=chat_history,
        )
        return self._result(
            response,
            intent,
            rag_sources=sources,
            generation_backend=self.generator.last_backend,
            error=self.generator.last_error,
            auto_emotion_record=auto_emotion_record,
        )

    def _retrieve_context(self, text: str) -> list[dict]:
        try:
            return self.retriever.retrieve(text)
        except Exception:
            return []

    def _auto_record_emotion(self, text: str, intent: str) -> dict | None:
        tool = self.tools["tool_emotion_log"]
        try:
            if not tool.should_auto_record(text, intent):
                return None
            parsed = tool.parse(text)
            result = tool.execute(**parsed)
            return result["data"]
        except Exception:
            return None

    def get_model_status(self) -> str:
        info = self.model_info
        if info["backend"] != "local_model":
            return "模型：未加载本地模型，当前使用 fallback 后端"

        return (
            f"模型：{info['model_kind']} | "
            f"加载量化：{info['load_quantization']} | "
            f"路径：{info['model_path']}"
        )

    @staticmethod
    def _result(
        response: str,
        intent: str,
        is_crisis: bool = False,
        tool_used: str | None = None,
        rag_sources: list | None = None,
        error: str | None = None,
        generation_backend: str | None = None,
        auto_emotion_record: dict | None = None,
    ) -> dict:
        return {
            "response": response,
            "intent": intent,
            "is_crisis": is_crisis,
            "tool_used": tool_used,
            "rag_sources": rag_sources or [],
            "error": error,
            "generation_backend": generation_backend,
            "auto_emotion_record": auto_emotion_record,
        }
