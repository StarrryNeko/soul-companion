"""Compare base, short-step LoRA, and long-step LoRA models.

The script can generate model responses and score them with a transparent
keyword-coverage rubric. It writes CSV/JSON tables plus charts that are useful
for course reports.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


DEFAULT_MODELS = [
    "base=base:../models/Qwen2.5-1.5B-Instruct",
    "short_steps=adapter:output/lora_adapter",
    "long_steps=adapter:output/lora_adapter_long_steps",
]


def parse_model_spec(spec: str) -> dict[str, str]:
    name, raw = spec.split("=", 1)
    kind, path = raw.split(":", 1)
    if kind not in {"base", "adapter", "merged"}:
        raise ValueError(f"Unsupported model kind: {kind}")
    return {"name": name, "kind": kind, "path": path}


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_for_model(model_spec: dict[str, str], cases: list[dict], args: argparse.Namespace) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_spec["name"]
    kind = model_spec["kind"]
    path = resolve_path(model_spec["path"])
    base_model_path = resolve_path(args.base_model)
    model_path = path if kind == "merged" else base_model_path

    if not model_path.exists():
        raise FileNotFoundError(f"{model_name}: model path not found: {model_path}")
    if kind == "adapter" and not path.exists():
        raise FileNotFoundError(f"{model_name}: adapter path not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if kind == "adapter":
        model = PeftModel.from_pretrained(model, str(path))
    model.eval()

    rows = []
    for case in cases:
        messages = [{"role": "user", "content": case["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        started = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = round(time.time() - started, 3)
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        rows.append(
            {
                "model": model_name,
                "case_id": case["id"],
                "category": case["category"],
                "category_label": case["category_label"],
                "input": case["input"],
                "expected_points": case["expected_points"],
                "answer": answer,
                "elapsed_sec": elapsed,
            }
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def score_answer(answer: str, expected_points: list[str]) -> tuple[float, int]:
    normalized = answer.lower()
    hits = 0
    for point in expected_points:
        if point.lower() in normalized:
            hits += 1
            continue
        # Chinese phrase matching is strict; give partial credit for compact key fragments.
        fragments = [frag for frag in point.replace("，", " ").replace("、", " ").split() if len(frag) >= 2]
        if fragments and any(fragment.lower() in normalized for fragment in fragments):
            hits += 1
    score = round((hits / max(len(expected_points), 1)) * 5, 2)
    return score, hits


def score_rows(rows: list[dict]) -> list[dict]:
    scored = []
    for row in rows:
        score, hits = score_answer(row["answer"], row["expected_points"])
        scored.append(
            {
                "model": row["model"],
                "case_id": row["case_id"],
                "category": row["category"],
                "category_label": row["category_label"],
                "input": row["input"],
                "answer": row["answer"],
                "matched_points": hits,
                "total_points": len(row["expected_points"]),
                "score_5": score,
                "elapsed_sec": row.get("elapsed_sec", ""),
            }
        )
    return scored


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(scored_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_model: dict[str, list[dict]] = {}
    by_pair: dict[tuple[str, str, str], list[dict]] = {}
    for row in scored_rows:
        by_model.setdefault(row["model"], []).append(row)
        key = (row["model"], row["category"], row["category_label"])
        by_pair.setdefault(key, []).append(row)

    model_summary = []
    for model, rows in by_model.items():
        scores = [float(row["score_5"]) for row in rows]
        times = [float(row["elapsed_sec"]) for row in rows if row["elapsed_sec"] != ""]
        model_summary.append(
            {
                "model": model,
                "case_count": len(rows),
                "avg_score_5": round(sum(scores) / len(scores), 2),
                "avg_elapsed_sec": round(sum(times) / len(times), 3) if times else "",
            }
        )

    category_summary = []
    for (model, category, label), rows in by_pair.items():
        scores = [float(row["score_5"]) for row in rows]
        category_summary.append(
            {
                "model": model,
                "category": category,
                "category_label": label,
                "case_count": len(rows),
                "avg_score_5": round(sum(scores) / len(scores), 2),
            }
        )
    return model_summary, category_summary


def make_charts(output_dir: Path, model_summary: list[dict], category_summary: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    model_df = pd.DataFrame(model_summary)
    ax = model_df.plot(kind="bar", x="model", y="avg_score_5", legend=False, color="#2f6f73", figsize=(7, 4))
    ax.set_title("模型整体评分对比")
    ax.set_xlabel("模型")
    ax.set_ylabel("平均得分（5分制）")
    ax.set_ylim(0, 5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_score_comparison.png", dpi=180)
    plt.close()

    cat_df = pd.DataFrame(category_summary)
    pivot = cat_df.pivot(index="category_label", columns="model", values="avg_score_5")
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_title("不同能力维度评分对比")
    ax.set_xlabel("评估维度")
    ax.set_ylabel("平均得分（5分制）")
    ax.set_ylim(0, 5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "category_score_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "data" / "evaluation" / "model_eval_cases.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "evaluation_comparison"))
    parser.add_argument("--base-model", default="../models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model", action="append", default=None, help="name=base|adapter|merged:path")
    parser.add_argument("--responses-json", default="", help="Score an existing responses JSON instead of generating.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    model_specs = args.model or DEFAULT_MODELS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(Path(args.cases))

    if args.responses_json:
        rows = json.loads(Path(args.responses_json).read_text(encoding="utf-8"))
    else:
        rows = []
        for spec in model_specs:
            rows.extend(generate_for_model(parse_model_spec(spec), cases, args))
        (output_dir / "responses.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    scored_rows = score_rows(rows)
    model_summary, category_summary = summarize(scored_rows)

    write_csv(output_dir / "case_scores.csv", scored_rows)
    write_csv(output_dir / "summary_by_model.csv", model_summary)
    write_csv(output_dir / "summary_by_category.csv", category_summary)
    make_charts(output_dir, model_summary, category_summary)

    print(f"wrote={output_dir}")
    for row in model_summary:
        print(f"{row['model']}: avg_score_5={row['avg_score_5']} cases={row['case_count']}")


if __name__ == "__main__":
    main()
