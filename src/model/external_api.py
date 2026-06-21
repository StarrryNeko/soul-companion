"""External OpenAI-compatible fallback API client."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from config.settings import FALLBACK_API_CONFIG, MODEL_CONFIG
from src.model.generator import ResponseGenerator


class ExternalAPIError(RuntimeError):
    """Raised when the external fallback API cannot return usable text."""


class DeepSeekFallbackClient:
    """Call DeepSeek through its OpenAI-compatible chat completions API."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or FALLBACK_API_CONFIG
        self.api_key = os.getenv(self.config["api_key_env"], "").strip()
        self.base_url = self.config["base_url"].rstrip("/")
        self.model = self.config["model"]
        self.timeout = int(self.config.get("timeout", 30))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        user_input: str,
        retrieved_context: list[str] | None = None,
        chat_history: list | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """调用 DeepSeek 兼容的 chat completions 接口，并在请求失败时抛出统一异常。"""
        if not self.available:
            raise ExternalAPIError(f"Missing API key environment variable: {self.config['api_key_env']}")

        payload = {
            "model": self.model,
            "messages": self._build_messages(user_input, retrieved_context or [], chat_history or []),
            "temperature": MODEL_CONFIG["temperature"],
            "top_p": MODEL_CONFIG["top_p"],
            "max_tokens": max_new_tokens or MODEL_CONFIG["max_new_tokens"],
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ExternalAPIError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise ExternalAPIError(f"DeepSeek API request failed: {exc}") from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalAPIError(f"Unexpected DeepSeek API response: {data}") from exc

        text = (text or "").strip()
        if not text:
            raise ExternalAPIError("DeepSeek API returned an empty response")
        return text

    def _build_messages(self, user_input: str, context: list[str], chat_history: list) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt(context)}]
        for item in chat_history[-6:]:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                user_message, assistant_message = item[0], item[1]
                if user_message:
                    messages.append({"role": "user", "content": str(user_message)})
                if assistant_message:
                    messages.append({"role": "assistant", "content": str(assistant_message)})
        messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _system_prompt(context: list[str]) -> str:
        context_text = "\n\n".join(context[:3]).strip()
        if not context_text:
            return ResponseGenerator.SYSTEM_PROMPT
        return f"{ResponseGenerator.SYSTEM_PROMPT}\n\n参考资料：\n{context_text}"
