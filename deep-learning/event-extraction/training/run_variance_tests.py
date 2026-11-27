"""Run multi-seed LoRA training & T4 evaluation to quantify reproducibility variance."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from evaluation.metrics_utils import evaluate_prediction_file, load_schema_validator

REPORTS_DIR = BASE / "reports"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
OUTPUT_FILE = REPORTS_DIR / "variance_metrics.json"
SEEDS = [42, 1337, 2026]


def main() -> None:
    validator = load_schema_validator(SCHEMA_FILE)
    test_rows = [json.loads(line) for line in open(TEST_FILE, "r", encoding="utf-8") if line.strip()]

    per_seed_results = []
    for seed in SEEDS:
        print(f"\n{'=' * 50}\nRunning Qwen LoRA Fine-Tuning with SEED = {seed}\n{'=' * 50}")
        seed_lora_dir = BASE / "models" / f"qwen_lora_seed_{seed}"
        seed_merged_dir = seed_lora_dir / "merged"
        seed_out_file = REPORTS_DIR / f"variance_outputs_seed_{seed}.jsonl"

        subprocess.run(
            [
                sys.executable,
                str(BASE / "training" / "train_qwen_lora.py"),
                "--output-dir",
                str(seed_lora_dir),
                "--seed",
                str(seed),
                "--skip-intermediate-eval",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(BASE / "training" / "merge_lora.py"),
                "--adapter-dir",
                str(seed_lora_dir),
                "--output-dir",
                str(seed_merged_dir),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(BASE / "inference" / "run_batch_inference.py"),
                "--model",
                "pipeline",
                "--model-dir",
                str(seed_merged_dir),
                "--output-file",
                str(seed_out_file),
            ],
            check=True,
        )

        pred_rows = [json.loads(line) for line in open(seed_out_file, "r", encoding="utf-8") if line.strip()]
        overall = evaluate_prediction_file(test_rows, pred_rows, validator)["overall"]
        overall["seed"] = seed
        per_seed_results.append(overall)
        print(f"SEED {seed} -> F1: {overall['f1']:.4f} | Top: {overall['top_level_f1']:.4f} | Arg: {overall['other_fields_f1']:.4f}")
        import shutil
        shutil.rmtree(seed_lora_dir, ignore_errors=True)

    f1_scores = [r["f1"] for r in per_seed_results]
    top_scores = [r["top_level_f1"] for r in per_seed_results]
    arg_scores = [r["other_fields_f1"] for r in per_seed_results]

    metrics = {
        "runs": len(SEEDS),
        "seeds_used": SEEDS,
        "f1_scores": f1_scores,
        "top_level_f1_scores": top_scores,
        "other_fields_f1_scores": arg_scores,
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "variance_f1": float(np.var(f1_scores)),
        "mean_top_level_f1": float(np.mean(top_scores)),
        "std_top_level_f1": float(np.std(top_scores)),
        "mean_other_fields_f1": float(np.mean(arg_scores)),
        "std_other_fields_f1": float(np.std(arg_scores)),
        "per_seed": per_seed_results,
    }
    OUTPUT_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nVARIANCE METRICS ACQUIRED:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
