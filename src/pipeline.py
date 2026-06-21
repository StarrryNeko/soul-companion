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

    CHAT_MODEL_OPTIONS = {
        "auto": {"label": "自动选择", "backend": "auto", "external_model": None},
        "deepseek-chat": {"label": "DeepSeek Chat", "backend": "deepseek_api", "external_model": "deepseek-chat"},
        "deepseek-reasoner": {
            "label": "DeepSeek Reasoner",
            "backend": "deepseek_api",
            "external_model": "deepseek-reasoner",
        },
        "template": {"label": "模板兜底回复", "backend": "template", "external_model": None},
    }

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
            "label": "",
            "load_error": "",
        }
        self.local_model_options = self._build_local_model_options()
        model, tokenizer = self._load_local_model(MODEL_CONFIG["merged_model_path"], "微调模型：基础步数")
        self.generator = ResponseGenerator(model=model, tokenizer=tokenizer, fallback_client=DeepSeekFallbackClient())
        self.tools = {
            "crisis_resource": CrisisResourceTool(),
            "tool_emotion_log": EmotionLoggerTool(),
            "tool_breathing": BreathingExerciseTool(),
        }

    def get_chat_model_choices(self) -> list[str]:
        local_labels = [option["label"] for option in self.local_model_options]
        remote_labels = [option["label"] for option in self.CHAT_MODEL_OPTIONS.values()]
        return [*local_labels, *remote_labels]

    def get_default_chat_model_choice(self) -> str:
        return self.CHAT_MODEL_OPTIONS["auto"]["label"]

    def switch_chat_model(self, selected_label: str) -> str:
        """按界面显示名切换本地或远程对话后端，并返回状态文本。"""
        local_option = next((option for option in self.local_model_options if option["label"] == selected_label), None)
        if local_option is not None:
            return self._switch_local_model(local_option)

        selected_option = next(
            (option for option in self.CHAT_MODEL_OPTIONS.values() if option["label"] == selected_label),
            None,
        )
        if selected_option is None:
            return f"未找到模型选项：{selected_label}"

        self.generator.set_chat_model(selected_option["backend"], selected_option["external_model"])
        return f"已切换对话模型：{selected_option['label']}\n{self.get_model_status()}"

    def _build_local_model_options(self) -> list[dict]:
        return [
            {
                "label": item["label"],
                "path": item["path"],
                "backend": "local_model",
            }
            for item in MODEL_CONFIG.get("local_chat_models", [])
        ]

    def _switch_local_model(self, option: dict) -> str:
        target_path = str(Path(option["path"]))
        if self.model_info.get("backend") == "local_model" and self.model_info.get("model_path") == target_path:
            self.generator.local_model_label = option["label"]
            self.generator.set_chat_model("local_model")
            return f"已切换对话模型：{option['label']}\n{self.get_model_status()}"

        self.generator.clear_local_model()
        model, tokenizer = self._load_local_model(target_path, option["label"])
        if model is None or tokenizer is None:
            self.generator.set_chat_model("template")
            self.model_info = {
                "backend": "fallback",
                "model_path": "",
                "model_kind": "none",
                "load_quantization": "none",
                "lora_adapter_path": "",
                "label": "",
                "load_error": self.model_info.get("load_error", ""),
            }
            return f"模型加载失败或目录不存在：{option['label']}\n路径：{target_path}\n已临时切换到模板兜底回复。"

        self.generator.set_local_model(model, tokenizer, option["label"])
        return f"已切换对话模型：{option['label']}\n{self.get_model_status()}"

    def _load_local_model(self, merged_model_path: str | None = None, label: str = ""):
        if os.getenv("SOUL_LOAD_LOCAL_MODEL", "1") != "1":
            self.model_info["load_error"] = "SOUL_LOAD_LOCAL_MODEL is not 1"
            return None, None

        model_path = Path(merged_model_path or MODEL_CONFIG["merged_model_path"])
        if not model_path.exists():
            self.model_info["load_error"] = f"model path does not exist: {model_path}"
            return None, None

        try:
            loader = ModelLoader(merged_model_path=str(model_path), lora_adapter_path=None)
            model, tokenizer = loader.load()
            self.model_info = {"backend": "local_model", "label": label, "load_error": "", **loader.last_model_info}
            return model, tokenizer
        except Exception as exc:
            self.model_info["load_error"] = f"{type(exc).__name__}: {exc}"
            print(f"Local merged model load failed, using fallback backend: {exc}")
            return None, None

    def process(self, user_input: str, chat_history: list | None = None) -> dict:
        """执行单轮对话主流程：输入校验、危机检测、意图路由、RAG 和回复生成。"""
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

        sources = self._retrieve_context(text) if intent == "mental_health" else []
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
        """生成供 UI 展示的当前后端、本地模型和加载错误摘要。"""
        info = self.model_info
        selected = self.generator.get_selected_model_label()
        if info["backend"] != "local_model":
            error = f" | 加载错误：{info['load_error']}" if info.get("load_error") else ""
            return f"当前选择：{selected} | 本地模型：未加载{error}"

        model_label = info.get("label") or selected
        model_dir = Path(info["model_path"]).name
        return (
            f"当前选择：{selected} | "
            f"本地模型：{model_label} | "
            f"模型目录：{model_dir} | "
            f"量化：{info['load_quantization']} | "
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
