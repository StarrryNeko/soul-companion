"""Load models produced by LLaMA-Factory."""

from __future__ import annotations

from pathlib import Path

from config.settings import MODEL_CONFIG


class ModelLoader:
    """Load either a base model plus LoRA adapter or an exported merged model."""

    def __init__(
        self,
        base_model_path: str | None = None,
        lora_adapter_path: str | None = None,
        merged_model_path: str | None = None,
        quantization: str | None = None,
    ) -> None:
        self.base_model_path = base_model_path or MODEL_CONFIG["base_model"]
        self.lora_adapter_path = lora_adapter_path or MODEL_CONFIG["lora_adapter_path"]
        self.merged_model_path = merged_model_path
        self.quantization = quantization or MODEL_CONFIG["quantization"]

    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = self.merged_model_path or self.base_model_path
        quant_config = None
        if self.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
            model = PeftModel.from_pretrained(model, self.lora_adapter_path)
        return model, tokenizer
