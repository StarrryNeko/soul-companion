"""Run lightweight rule-level evaluation on standard test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import MentalHealthPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "pipeline_case_results.json"),
        help="Path used to save all 16 pipeline responses and routing metadata.",
    )
    args = parser.parse_args()

    cases = json.loads((ROOT / "tests" / "test_cases.json").read_text(encoding="utf-8"))
    pipeline = MentalHealthPipeline()
    rows = []
    for case in cases:
        result = pipeline.process(case["input"])
        print(f"[{case['id']}] {case['type']} -> intent={result['intent']} crisis={result['is_crisis']}")
        print(result["response"][:160].replace("\n", " "))
        rows.append(
            {
                "id": case["id"],
                "type": case["type"],
                "input": case["input"],
                "expected_behavior": case["expected_behavior"],
                "intent": result["intent"],
                "is_crisis": result["is_crisis"],
                "tool_used": result.get("tool_used"),
                "generation_backend": result.get("generation_backend"),
                "rag_sources": result.get("rag_sources", []),
                "response": result["response"],
                "error": result.get("error"),
            }
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evaluated={len(cases)}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
