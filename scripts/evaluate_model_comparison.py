"""Compare base, short-step, medium-step, and long-step LoRA models.

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
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


DEFAULT_MODELS = [
    "base=base:../models/Qwen2.5-1.5B-Instruct",
    "short_steps=adapter:output/lora_adapter",
    "medium_steps=adapter:output/lora_adapter_medium_steps",
    "long_steps=adapter:output/lora_adapter_long_steps",
]

CATEGORY_LABELS_EN = {
    "域内-训练主题": "In-domain Training Topics",
    "域内泛化": "In-domain Generalization",
    "指令遵循": "Instruction Following",
    "安全边界": "Safety Boundaries",
    "域外能力": "Out-of-domain Ability",
}


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


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_model_specs(model_specs: list[str], args: argparse.Namespace) -> list[dict[str, str]]:
    parsed_specs = [parse_model_spec(spec) for spec in model_specs]
    base_model_path = resolve_path(args.base_model)
    missing: list[str] = []

    for model_spec in parsed_specs:
        model_name = model_spec["name"]
        kind = model_spec["kind"]
        path = resolve_path(model_spec["path"])
        model_path = path if kind == "merged" else base_model_path

        log(f"[preflight:{model_name}] kind={kind} model_path={model_path}")
        if not model_path.exists():
            missing.append(f"{model_name}: model path not found: {model_path}")
        if kind == "adapter":
            log(f"[preflight:{model_name}] adapter_path={path}")
            if not path.exists():
                missing.append(f"{model_name}: adapter path not found: {path}")

    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing model files before evaluation:\n{details}")

    return parsed_specs


def generate_for_model(model_spec: dict[str, str], cases: list[dict], args: argparse.Namespace) -> list[dict]:
    model_name = model_spec["name"]
    kind = model_spec["kind"]
    path = resolve_path(model_spec["path"])
    base_model_path = resolve_path(args.base_model)
    model_path = path if kind == "merged" else base_model_path

    log(f"[model:{model_name}] kind={kind}")
    log(f"[model:{model_name}] model_path={model_path}")
    if kind == "adapter":
        log(f"[model:{model_name}] adapter_path={path}")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name}: model path not found: {model_path}")
    if kind == "adapter" and not path.exists():
        raise FileNotFoundError(f"{model_name}: adapter path not found: {path}")
    log(f"[model:{model_name}] importing torch/transformers/peft ...")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(
        f"[model:{model_name}] torch cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )

    log(f"[model:{model_name}] loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    log(f"[model:{model_name}] loading base/merged model ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if kind == "adapter":
        log(f"[model:{model_name}] loading LoRA adapter ...")
        model = PeftModel.from_pretrained(model, str(path))
    model.eval()

    rows = []
    for index, case in enumerate(cases, start=1):
        log(f"[model:{model_name}] generating case {index}/{len(cases)} id={case['id']} ...")
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
        log(f"[model:{model_name}] finished case {case['id']} elapsed={elapsed}s")
        rows.append(
            {
                "model": model_name,
                "case_id": case["id"],
                "category": case["category"],
                "category_label": case["category_label"],
                "input": case["input"],
                "expected_points": case.get("expected_points", []),
                "criteria": case.get("criteria", []),
                "answer": answer,
                "elapsed_sec": elapsed,
            }
        )

    log(f"[model:{model_name}] releasing model memory ...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"[model:{model_name}] done rows={len(rows)}")
    return rows


def extract_list_items(answer: str) -> list[str]:
    """Extract markdown/bulleted/numbered items for instruction checks."""
    pattern = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.、)])\s*(.+?)(?=\n\s*(?:[-*•]|\d+[.、)])\s*|$)", re.S)
    return [re.sub(r"\s+", "", item).strip() for item in pattern.findall(answer) if item.strip()]


def count_sentences(answer: str) -> int:
    parts = [part.strip() for part in re.split(r"[。！？!?]+", answer) if part.strip()]
    return len(parts)


def criterion_matches(answer: str, criterion: dict) -> bool:
    normalized = answer.lower()
    compact = re.sub(r"\s+", "", answer)
    list_items = extract_list_items(answer)

    if criterion.get("any_of") and not any(term.lower() in normalized for term in criterion["any_of"]):
        return False
    if criterion.get("all_of") and not all(term.lower() in normalized for term in criterion["all_of"]):
        return False
    if criterion.get("none_of") and any(term.lower() in normalized for term in criterion["none_of"]):
        return False
    if criterion.get("regex_any") and not any(re.search(pattern, answer, re.I | re.M) for pattern in criterion["regex_any"]):
        return False
    if "min_list_items" in criterion and len(list_items) < int(criterion["min_list_items"]):
        return False
    if "exact_list_items" in criterion and len(list_items) != int(criterion["exact_list_items"]):
        return False
    if "max_list_item_chars" in criterion:
        if not list_items or any(len(item) > int(criterion["max_list_item_chars"]) for item in list_items):
            return False
    sentence_count = count_sentences(answer)
    if "min_sentences" in criterion and sentence_count < int(criterion["min_sentences"]):
        return False
    if "max_sentences" in criterion and sentence_count > int(criterion["max_sentences"]):
        return False
    if "min_chars" in criterion and len(compact) < int(criterion["min_chars"]):
        return False
    if "max_chars" in criterion and len(compact) > int(criterion["max_chars"]):
        return False
    return True


def score_answer(answer: str, criteria: list[dict], expected_points: list[str]) -> tuple[float, int, list[dict]]:
    """Score transparent rubric criteria; retain the old phrase matcher as a fallback."""
    if criteria:
        results = [
            {"label": item.get("label", f"criterion_{index}"), "passed": criterion_matches(answer, item)}
            for index, item in enumerate(criteria, start=1)
        ]
        hits = sum(1 for item in results if item["passed"])
        score = round((hits / len(results)) * 5, 2)
        return score, hits, results

    normalized = answer.lower()
    hits = 0
    results = []
    for point in expected_points:
        matched = point.lower() in normalized
        if not matched:
            fragments = [frag for frag in re.split(r"[，、；;\s]+", point) if len(frag) >= 2]
            matched = bool(fragments and any(fragment.lower() in normalized for fragment in fragments))
        hits += int(matched)
        results.append({"label": point, "passed": matched})
    score = round((hits / max(len(expected_points), 1)) * 5, 2)
    return score, hits, results


def score_rows(rows: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    scored = []
    for row in rows:
        case = cases_by_id.get(row["case_id"], {})
        criteria = case.get("criteria", row.get("criteria", []))
        expected_points = case.get("expected_points", row.get("expected_points", []))
        score, hits, criterion_results = score_answer(row["answer"], criteria, expected_points)
        total_points = len(criteria) if criteria else len(expected_points)
        scored.append(
            {
                "model": row["model"],
                "case_id": row["case_id"],
                "category": row["category"],
                "category_label": row["category_label"],
                "input": row["input"],
                "answer": row["answer"],
                "matched_points": hits,
                "total_points": total_points,
                "score_5": score,
                "criterion_results": criterion_results,
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


def validate_score_distribution(model_summary: list[dict]) -> None:
    scores = [float(row["avg_score_5"]) for row in model_summary]
    if not scores:
        raise ValueError("No model scores were produced.")
    if max(scores) <= 0.25:
        raise ValueError(
            "All model scores are near zero. Check that the evaluation cases match the responses "
            "and use the v2 criteria rubric before interpreting charts."
        )


def make_charts(output_dir: Path, model_summary: list[dict], category_summary: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    model_df = pd.DataFrame(model_summary).sort_values("avg_score_5", ascending=True)
    prompt_counts = sorted({int(value) for value in model_df["case_count"]})
    prompt_scope = str(prompt_counts[0]) if len(prompt_counts) == 1 else f"{prompt_counts[0]}–{prompt_counts[-1]}"
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="#FCFCFD")
    sns.barplot(
        data=model_df,
        x="avg_score_5",
        y="model",
        color="#A3BEFA",
        edgecolor="#2E4780",
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title("")
    ax.set_xlabel("Average score (0–5)")
    ax.set_ylabel("")
    ax.set_xlim(0, 5)
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for patch, value in zip(ax.patches, model_df["avg_score_5"]):
        ax.text(min(float(value) + 0.06, 4.86), patch.get_y() + patch.get_height() / 2, f"{value:.2f}", va="center", fontsize=9, color="#1F2430")
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    header_left = ax.get_position().x0
    fig.text(header_left, 0.965, "Overall model rubric score", ha="left", va="top", fontsize=14, fontweight="semibold", color="#1F2430")
    fig.text(header_left, 0.915, f"{prompt_scope} prompts per model; 4 transparent criteria per prompt; 0–5 scale", ha="left", va="top", fontsize=9, color="#6F768A")
    fig.savefig(output_dir / "overall_score_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log(f"[charts] wrote {output_dir / 'overall_score_comparison.png'}")

    cat_df = pd.DataFrame(category_summary)
    cat_df["category_label_en"] = cat_df["category_label"].map(CATEGORY_LABELS_EN).fillna(cat_df["category"])
    pivot = cat_df.pivot(index="category_label_en", columns="model", values="avg_score_5")
    category_counts = sorted({int(value) for value in cat_df["case_count"]})
    category_scope = str(category_counts[0]) if len(category_counts) == 1 else f"{category_counts[0]}–{category_counts[-1]}"
    category_order = [
        "In-domain Training Topics",
        "In-domain Generalization",
        "Instruction Following",
        "Safety Boundaries",
        "Out-of-domain Ability",
    ]
    pivot = pivot.reindex([item for item in category_order if item in pivot.index])
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="#FCFCFD")
    cmap = sns.blend_palette(["#FFFFFF", "#EAF1FE", "#CEDFFE", "#A3BEFA", "#5477C4"], as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=5,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Average score (0–5)"},
    )
    ax.set_title("")
    ax.set_xlabel("Model")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    header_left = ax.get_position().x0
    fig.text(header_left, 0.965, "Model score by evaluation dimension", ha="left", va="top", fontsize=14, fontweight="semibold", color="#1F2430")
    fig.text(header_left, 0.915, f"Each cell averages {category_scope} prompt(s); inspect safety as a hard gate, not only as part of the overall mean", ha="left", va="top", fontsize=9, color="#6F768A")
    fig.savefig(output_dir / "category_score_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log(f"[charts] wrote {output_dir / 'category_score_comparison.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "data" / "evaluation" / "model_eval_cases_v2.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "evaluation_comparison"))
    parser.add_argument("--base-model", default="../models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model", action="append", default=None, help="name=base|adapter|merged:path")
    parser.add_argument("--responses-json", default="", help="Score an existing responses JSON instead of generating.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Only run the first N cases for a quick smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    model_specs = args.model or DEFAULT_MODELS

    log("[start] evaluating model comparison")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = resolve_path(args.cases)
    log(f"[start] cases={cases_path}")
    log(f"[start] output_dir={output_dir}")
    cases = load_cases(cases_path)
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]
        log(f"[start] limit_cases={len(cases)}")
    log(f"[start] model_count={len(model_specs)} case_count={len(cases)}")
    parsed_model_specs = validate_model_specs(model_specs, args) if not args.responses_json else []

    if args.responses_json:
        responses_path = resolve_path(args.responses_json)
        log(f"[responses] loading existing responses={responses_path}")
        rows = json.loads(responses_path.read_text(encoding="utf-8"))
    else:
        rows = []
        for index, model_spec in enumerate(parsed_model_specs, start=1):
            log(f"[model] {index}/{len(parsed_model_specs)} name={model_spec['name']}")
            model_rows = generate_for_model(model_spec, cases, args)
            rows.extend(model_rows)
            write_json(output_dir / "responses.json", rows)
            log(f"[responses] checkpoint wrote rows={len(rows)} path={output_dir / 'responses.json'}")

    log(f"[score] scoring rows={len(rows)}")
    cases_by_id = {case["id"]: case for case in cases}
    scored_rows = score_rows(rows, cases_by_id)
    model_summary, category_summary = summarize(scored_rows)

    write_csv(output_dir / "case_scores.csv", scored_rows)
    write_csv(output_dir / "summary_by_model.csv", model_summary)
    write_csv(output_dir / "summary_by_category.csv", category_summary)
    write_json(output_dir / "case_scores.json", scored_rows)
    write_json(output_dir / "summary_by_model.json", model_summary)
    write_json(output_dir / "summary_by_category.json", category_summary)
    log("[score] wrote CSV/JSON score files")
    validate_score_distribution(model_summary)
    make_charts(output_dir, model_summary, category_summary)

    log(f"wrote={output_dir}")
    for row in model_summary:
        log(f"{row['model']}: avg_score_5={row['avg_score_5']} cases={row['case_count']}")


if __name__ == "__main__":
    main()
