"""Train, merge, and run T4 batch inference for comparison SLMs (SmolLM2, TinyLlama)."""

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]


def run_model(model_name: str, skip_intermediate_eval: bool = True, force_retrain: bool = False) -> None:
    safe_name = model_name.split("/")[-1]
    lora_dir = BASE / "models" / f"{safe_name}_lora"
    merged_dir = lora_dir / "merged"
    out_jsonl = BASE / "reports" / f"{safe_name}_pipeline.jsonl"
    perf_json = BASE / "reports" / f"{safe_name}_perf.json"

    print(f"\n{'*' * 60}\nEvaluating Comparison Model: {model_name}\n{'*' * 60}")
    if force_retrain or not (lora_dir / "adapter_config.json").exists():
        cmd = [
            sys.executable,
            str(BASE / "training" / "train_qwen_lora.py"),
            "--base-model",
            model_name,
            "--output-dir",
            str(lora_dir),
        ]
        if skip_intermediate_eval:
            cmd.append("--skip-intermediate-eval")
        subprocess.run(cmd, check=True)
    else:
        print(f"[{safe_name}] LoRA adapter exists at {lora_dir}; skipping retrain.")

    if force_retrain or not (merged_dir / "config.json").exists():
        subprocess.run(
            [
                sys.executable,
                str(BASE / "training" / "merge_lora.py"),
                "--base-model",
                model_name,
                "--adapter-dir",
                str(lora_dir),
                "--output-dir",
                str(merged_dir),
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
            str(merged_dir),
            "--output-file",
            str(out_jsonl),
            "--metrics-out",
            str(perf_json),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate comparison SLMs on T4.")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--full-eval", action="store_true", help="Evaluate after every epoch during training")
    args = parser.parse_args()

    for model in MODELS_TO_TEST:
        run_model(model, skip_intermediate_eval=not args.full_eval, force_retrain=args.force_retrain)

    print("\nAll model comparisons finished.")


if __name__ == "__main__":
    main()
