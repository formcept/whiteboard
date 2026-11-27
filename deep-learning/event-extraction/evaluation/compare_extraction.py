"""Compare extraction quality, schema validity, and per-class breakdown across models."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from evaluation.metrics_utils import evaluate_prediction_file, load_schema_validator

SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
PLOT_FILE = BASE / "reports" / "comparison_metrics.png"
CSV_FILE = BASE / "reports" / "comparison_metrics.csv"
PER_CLASS_FILE = BASE / "reports" / "per_class_metrics.json"

DEFAULT_MODELS = [
    ("Baseline (Qwen Instruct)", BASE / "reports" / "baseline_structured_outputs.jsonl"),
    ("Qwen2.5-1.5B (LoRA)", BASE / "reports" / "structured_outputs.jsonl"),
    ("SmolLM2-1.7B (LoRA)", BASE / "reports" / "SmolLM2-1.7B-Instruct_pipeline.jsonl"),
    ("TinyLlama-1.1B (LoRA)", BASE / "reports" / "TinyLlama-1.1B-Chat-v1.0_pipeline.jsonl"),
]


def _plot_comparison(summary: pd.DataFrame, plot_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    models = summary.index.tolist()
    x = np.arange(len(models))
    width_3 = 0.25

    rects1 = axes[0].bar(x - width_3, summary["precision"], width_3, label="Precision", color="#2a9d8f")
    rects2 = axes[0].bar(x, summary["recall"], width_3, label="Recall", color="#e9c46a")
    rects3 = axes[0].bar(x + width_3, summary["f1"], width_3, label="F1", color="#457b9d")
    axes[0].set_title("Overall Extraction Quality", fontsize=11, fontweight="bold", pad=10)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9, rotation=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    width_2 = 0.3
    rects_tl = axes[1].bar(x - width_2 / 2, summary["top_level_f1"], width_2, label="Top-Level F1", color="#1d3557")
    rects_of = axes[1].bar(x + width_2 / 2, summary["other_fields_f1"], width_2, label="Other Fields F1", color="#e63946")
    axes[1].set_title("F1 Performance Breakdown", fontsize=11, fontweight="bold", pad=10)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9, rotation=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for rects in (rects_tl, rects_of):
        for rect in rects:
            h = rect.get_height()
            axes[1].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    rects_valid = axes[2].bar(x, summary["schema_validity"], width_2 * 1.5, color="#8d99ae")
    axes[2].set_title("Schema Validity Rate", fontsize=11, fontweight="bold", pad=10)
    axes[2].set_ylabel("Rate", fontsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=9, rotation=10)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects_valid:
        h = rect.get_height()
        axes[2].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Baseline vs Pipeline Performance Comparison", fontsize=14, fontweight="bold")
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction JSONL predictions.")
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    parser.add_argument("--csv-out", type=Path, default=CSV_FILE)
    parser.add_argument("--plot-out", type=Path, default=PLOT_FILE)
    parser.add_argument("--per-class-out", type=Path, default=PER_CLASS_FILE)
    args = parser.parse_args()

    validator = load_schema_validator(SCHEMA_FILE)
    test_rows = [json.loads(line) for line in open(args.test_file, "r", encoding="utf-8") if line.strip()]

    summary_rows = []
    per_class_report = {}
    for label, path in DEFAULT_MODELS:
        if not path.exists():
            continue
        pred_rows = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
        eval_res = evaluate_prediction_file(test_rows, pred_rows, validator)
        overall = eval_res["overall"]
        summary_rows.append({
            "model": label,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "top_level_f1": overall["top_level_f1"],
            "other_fields_f1": overall["other_fields_f1"],
            "schema_validity": overall["schema_validity"],
        })
        per_class_report[label] = {
            "overall": overall,
            "per_class": eval_res["per_class"],
            "confusion_matrix": eval_res["confusion_matrix"],
        }

    summary = pd.DataFrame(summary_rows).set_index("model")
    cols = ["precision", "recall", "f1", "top_level_f1", "other_fields_f1", "schema_validity"]
    summary[cols].to_csv(args.csv_out)
    _plot_comparison(summary, args.plot_out)
    args.per_class_out.write_text(json.dumps(per_class_report, indent=2), encoding="utf-8")

    print(summary[cols])
    print("Saved:", args.csv_out)
    print("Saved:", args.plot_out)
    print("Saved:", args.per_class_out)


if __name__ == "__main__":
    main()
