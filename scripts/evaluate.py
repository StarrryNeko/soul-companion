"""Run lightweight rule-level evaluation on standard test cases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import MentalHealthPipeline


def main() -> None:
    cases = json.loads((ROOT / "tests" / "test_cases.json").read_text(encoding="utf-8"))
    pipeline = MentalHealthPipeline()
    for case in cases:
        result = pipeline.process(case["input"])
        print(f"[{case['id']}] {case['type']} -> intent={result['intent']} crisis={result['is_crisis']}")
        print(result["response"][:160].replace("\n", " "))
    print(f"evaluated={len(cases)}")


if __name__ == "__main__":
    main()

