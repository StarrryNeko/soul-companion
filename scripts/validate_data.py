"""Validate project data files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
QUESTION_DAMAGE_RE = re.compile(r"\?{3,}")
MOJIBAKE_HINTS = ("鎴", "鍚", "涓€", "绱", "铏", "€?")


def validate_text_quality(index: int, key: str, text: str) -> None:
    if "\ufffd" in text:
        raise ValueError(f"row {index} {key} contains replacement characters")
    if QUESTION_DAMAGE_RE.search(text):
        raise ValueError(f"row {index} {key} contains repeated question marks")
    if any(hint in text for hint in MOJIBAKE_HINTS):
        raise ValueError(f"row {index} {key} may contain mojibake text")


def validate_training() -> int:
    path = ROOT / "data" / "training" / "mental_health_qa.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("training data must be a list")
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        for key in ("instruction", "input", "output"):
            if key not in row:
                raise ValueError(f"row {index} missing {key}")
            if not isinstance(row[key], str):
                raise ValueError(f"row {index} {key} must be a string")
            validate_text_quality(index, key, row[key])
        if not isinstance(row["output"], str) or not row["output"].strip():
            raise ValueError(f"row {index} has empty output")
        row_key = (row["instruction"], row["input"], row["output"])
        if row_key in seen:
            raise ValueError(f"row {index} is duplicated")
        seen.add(row_key)
    return len(rows)


def validate_tests() -> int:
    path = ROOT / "tests" / "test_cases.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for index, row in enumerate(rows):
        for key in ("id", "type", "input", "expected_behavior"):
            if key not in row:
                raise ValueError(f"test case {index} missing {key}")
    return len(rows)


def validate_docs() -> int:
    docs = sorted((ROOT / "data" / "knowledge_base" / "docs").glob("*.md"))
    if not docs:
        raise ValueError("knowledge base docs are empty")
    return len(docs)


def main() -> None:
    training_count = validate_training()
    test_count = validate_tests()
    doc_count = validate_docs()
    print(f"training_count={training_count}")
    print(f"test_count={test_count}")
    print(f"knowledge_docs={doc_count}")
    print("validation=ok")


if __name__ == "__main__":
    main()
