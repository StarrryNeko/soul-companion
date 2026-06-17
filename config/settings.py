"""Global settings for SoulCompanion."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = {
    "base_model": os.getenv("SOUL_BASE_MODEL", str(PROJECT_ROOT.parent / "models" / "Qwen2.5-1.5B-Instruct")),
    "use_modelscope": os.getenv("SOUL_USE_MODELSCOPE", "1") == "1",
    "model_cache_dir": os.getenv("SOUL_MODEL_CACHE_DIR", str(PROJECT_ROOT / "models")),
    "lora_adapter_path": str(PROJECT_ROOT / "output" / "lora_adapter"),
    "merged_model_path": str(PROJECT_ROOT / "output" / "merged_model"),
    "local_chat_models": [
        {
            "key": "merged_model",
            "label": "微调模型：基础步数",
            "path": str(PROJECT_ROOT / "output" / "merged_model"),
        },
        {
            "key": "merged_model_medium_steps",
            "label": "微调模型：中等步数",
            "path": str(PROJECT_ROOT / "output" / "merged_model_medium_steps"),
        },
        {
            "key": "merged_model_long_steps",
            "label": "微调模型：长步数",
            "path": str(PROJECT_ROOT / "output" / "merged_model_long_steps"),
        },
    ],
    "quantization": "int4",
    "max_new_tokens": int(os.getenv("SOUL_MAX_NEW_TOKENS", "320")),
    "temperature": 0.4,
    "top_p": 0.85,
}

RAG_CONFIG = {
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "persist_dir": str(PROJECT_ROOT / "output" / "chroma_db"),
    "docs_dir": str(PROJECT_ROOT / "data" / "knowledge_base" / "docs"),
    "chunk_size": 400,
    "chunk_overlap": 50,
    "top_k": 3,
}

SAFETY_CONFIG = {
    "keywords_path": str(PROJECT_ROOT / "config" / "crisis_keywords.json"),
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "semantic_threshold": 0.85,
}

TOOLS_CONFIG = {
    "emotion_log_path": str(PROJECT_ROOT / "output" / "emotion_logs.json"),
}

FALLBACK_API_CONFIG = {
    "provider": "deepseek",
    "api_key_env": "DEEPSEEK_API_KEY",
    "base_url": "https://api.deepseek.com",
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "timeout": 30,
}

UI_CONFIG = {
    "title": "心晴 - 心理健康支持助手",
    "disclaimer": "本系统仅提供心理健康科普和一般性情绪支持，不能替代专业心理咨询、医学诊断或治疗。",
    "server_name": os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
}
