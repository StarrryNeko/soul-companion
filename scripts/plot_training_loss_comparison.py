"""Plot loss curves for the short, medium, and long-step LoRA training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def load_loss_rows(run_name: str, output_dir: Path) -> list[dict]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json: {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = []
    for item in state.get("log_history", []):
        if "loss" not in item or "step" not in item:
            continue
        rows.append({"run": run_name, "step": int(item["step"]), "loss": float(item["loss"])})
    if not rows:
        raise ValueError(f"No loss records found in {state_path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "step", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, group in df.groupby("run"):
        group = group.sort_values("step")
        ax.plot(group["step"], group["loss"], marker="o", markersize=3, linewidth=1.8, label=run)

    ax.set_title("Training Loss Comparison Across Step Settings")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-name", default="short_steps")
    parser.add_argument("--short-dir", default="output/lora_adapter")
    parser.add_argument("--medium-name", default="medium_steps")
    parser.add_argument("--medium-dir", default="output/lora_adapter_medium_steps")
    parser.add_argument("--long-name", default="long_steps")
    parser.add_argument("--long-dir", default="output/lora_adapter_long_steps")
    parser.add_argument("--output-dir", default="output/evaluation_comparison")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_loss_rows(args.short_name, resolve_path(args.short_dir)))
    rows.extend(load_loss_rows(args.medium_name, resolve_path(args.medium_dir)))
    rows.extend(load_loss_rows(args.long_name, resolve_path(args.long_dir)))

    write_csv(output_dir / "training_loss_comparison.csv", rows)
    plot_loss(output_dir / "training_loss_comparison.png", rows)
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
