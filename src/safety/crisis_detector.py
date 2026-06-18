"""Crisis signal detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import SAFETY_CONFIG


class CrisisDetector:
    """Detect explicit keywords first, then semantically similar crisis signals."""

    def __init__(
        self,
        keywords_path: str | None = None,
        threshold: float | None = None,
        enable_semantic: bool | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self.keywords_path = Path(keywords_path or SAFETY_CONFIG["keywords_path"])
        self.threshold = threshold if threshold is not None else SAFETY_CONFIG["semantic_threshold"]
        self.enable_semantic = (
            enable_semantic if enable_semantic is not None else SAFETY_CONFIG.get("semantic_enabled", True)
        )
        self.embedding_model_name = SAFETY_CONFIG["embedding_model"]
        self._embedding_model = embedding_model
        self._semantic_embeddings: np.ndarray | None = None
        self._semantic_labels: list[tuple[str, str]] = []
        self.last_semantic_error: str | None = None
        payload = json.loads(self.keywords_path.read_text(encoding="utf-8"))
        self.high_risk = payload.get("high_risk", [])
        self.medium_risk = payload.get("medium_risk", [])
        self.high_risk_examples = payload.get("high_risk_examples", [])
        self.medium_risk_examples = payload.get("medium_risk_examples", [])

    def check(self, user_input: str) -> dict:
        text = user_input.strip().lower()
        for keyword in self.high_risk:
            if keyword.lower() in text:
                return self._hit("keyword", keyword, 1.0, "high")
        for keyword in self.medium_risk:
            if keyword.lower() in text:
                return self._hit("keyword", keyword, 0.9, "medium")

        semantic_hit = self._check_semantic(text)
        if semantic_hit is not None:
            return semantic_hit

        return {
            "is_crisis": False,
            "trigger_type": None,
            "matched_keyword": None,
            "similarity_score": None,
            "risk_level": None,
        }

    def _check_semantic(self, text: str) -> dict | None:
        if not self.enable_semantic or not text:
            return None
        if not self.high_risk_examples and not self.medium_risk_examples:
            return None

        try:
            model = self._get_embedding_model()
            reference_embeddings = self._get_semantic_embeddings(model)
            query_embedding = np.asarray(
                model.encode([text], normalize_embeddings=True), dtype=np.float32
            )[0]
            similarities = reference_embeddings @ query_embedding
            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])
            if best_score < self.threshold:
                return None

            risk_level, example = self._semantic_labels[best_index]
            return self._hit("semantic", example, round(best_score, 4), risk_level)
        except Exception as exc:
            # Semantic detection is an enhancement. If the embedding model is
            # unavailable, retain the deterministic keyword detector instead of
            # classifying every normal request as a crisis.
            self.last_semantic_error = f"{type(exc).__name__}: {exc}"
            return None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _get_semantic_embeddings(self, model) -> np.ndarray:
        if self._semantic_embeddings is None:
            self._semantic_labels = [
                *(("high", example) for example in self.high_risk_examples),
                *(("medium", example) for example in self.medium_risk_examples),
            ]
            examples = [example for _, example in self._semantic_labels]
            self._semantic_embeddings = np.asarray(
                model.encode(examples, normalize_embeddings=True), dtype=np.float32
            )
        return self._semantic_embeddings

    @staticmethod
    def _hit(trigger_type: str, keyword: str, score: float, risk_level: str) -> dict:
        return {
            "is_crisis": True,
            "trigger_type": trigger_type,
            "matched_keyword": keyword,
            "similarity_score": score,
            "risk_level": risk_level,
        }
