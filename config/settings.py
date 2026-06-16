"""Global settings for SoulCompanion."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_adapter_path": str(PROJECT_ROOT / "output" / "lora_adapter"),
    "merged_model_path": str(PROJECT_ROOT / "output" / "merged_model"),
    "quantization": "int4",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
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
    "provider": "qwen-plus",
    "api_key_env": "FALLBACK_API_KEY",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "timeout": 30,
}

UI_CONFIG = {
    "title": "心晴 - 心理健康支持助手",
    "disclaimer": "本系统仅提供心理健康科普和一般性情绪支持，不能替代专业心理咨询、医学诊断或治疗。",
    "server_name": "0.0.0.0",
    "server_port": 7860,
}
