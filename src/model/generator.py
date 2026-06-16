"""Response generation."""

from __future__ import annotations


class ResponseGenerator:
    """Generate responses with model when available, otherwise rule fallback."""

    SYSTEM_PROMPT = (
        "你是一位温和、专业的心理健康科普助手。你的职责是提供心理健康科普、一般性情绪支持和自助建议。"
        "你不能进行诊断、治疗、开药或替代专业心理咨询。"
    )

    def __init__(self, model=None, tokenizer=None, fallback_client=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.fallback_client = fallback_client
        self.last_backend = "template"
        self.last_error: str | None = None

    def generate(
        self,
        user_input: str,
        retrieved_context: list[str] | None = None,
        chat_history: list | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        context = retrieved_context or []
        self.last_error = None
        if self.model is not None and self.tokenizer is not None:
            try:
                self.last_backend = "local_model"
                return self._model_generate(user_input, context, max_new_tokens)
            except Exception as exc:
                self.last_error = f"local_model: {exc}"

        if self.fallback_client is not None:
            try:
                self.last_backend = "deepseek_api"
                return self.fallback_client.generate(user_input, context, chat_history, max_new_tokens)
            except Exception as exc:
                self.last_error = f"{self.last_error}; deepseek_api: {exc}" if self.last_error else f"deepseek_api: {exc}"

        self.last_backend = "template"
        return self._template_response(user_input, context)

    def _model_generate(self, user_input: str, context: list[str], max_new_tokens: int) -> str:
        import torch

        context_text = "\n\n".join(context)
        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}\n"
            f"参考资料：\n{context_text}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return text.strip()

    def _template_response(self, user_input: str, context: list[str]) -> str:
        prefix = "我理解你提到的困扰。"
        if context:
            sources = "\n\n".join(context[:2])
            return (
                f"{prefix}根据知识库资料，可以先从一个很小、可执行的步骤开始处理。\n\n"
                f"参考信息：{sources[:500]}\n\n"
                "如果这种状态持续影响睡眠、学习或日常生活，建议联系学校心理咨询中心或专业人员。"
            )
        return (
            f"{prefix}你可以先把问题具体化，区分哪些部分可以行动、哪些部分暂时只能接纳。"
            "如果困扰持续存在，建议寻求学校心理咨询中心或专业人员支持。"
        )
