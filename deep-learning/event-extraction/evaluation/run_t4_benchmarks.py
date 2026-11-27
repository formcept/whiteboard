"""Unified NVIDIA T4 GPU benchmark runner for ablations, larger LLMs, sensitivity, and CV."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from evaluation.metrics_utils import evaluate_prediction_file, load_schema_validator
from inference.run_batch_inference import run_baseline, run_configured_pipeline

DATA_QWEN_TEST = BASE / "data" / "qwen" / "qwen_test.jsonl"
DATA_DIST_TEST = BASE / "data" / "distilbert" / "distilbert_test.jsonl"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
REPORTS_DIR = BASE / "reports"
MERGED_QWEN = BASE / "models" / "qwen_lora" / "merged"
DISTILBERT_DIR = BASE / "models" / "distilbert"


def _load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]


def _eval_jsonl(test_rows: list, pred_path: Path, validator) -> dict:
    pred_rows = _load_jsonl(pred_path)
    return evaluate_prediction_file(test_rows, pred_rows, validator)


def run_ablations(validator) -> None:
    """Run controlled component ablations and E2E triage evaluation on T4 GPU."""
    test_rows = _load_jsonl(DATA_QWEN_TEST)
    ab_dir = REPORTS_DIR / "ablations"
    ab_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("1_zeroshot_unconstrained", dict(kind="baseline", model="Qwen/Qwen2.5-1.5B-Instruct")),
        ("2_zeroshot_outlines", dict(kind="pipeline", model="Qwen/Qwen2.5-1.5B-Instruct", triage=False, outlines=True, corr=False, filt=False)),
        ("3_lora_unconstrained", dict(kind="pipeline", model=str(MERGED_QWEN), triage=False, outlines=False, corr=False, filt=False)),
        ("4_lora_outlines_no_corr", dict(kind="pipeline", model=str(MERGED_QWEN), triage=False, outlines=True, corr=False, filt=False)),
        ("5_lora_outlines_full_stage2", dict(kind="pipeline", model=str(MERGED_QWEN), triage=False, outlines=True, corr=True, filt=True)),
        ("6_e2e_distilbert_plus_stage2", dict(kind="pipeline", model=str(MERGED_QWEN), triage=True, outlines=True, corr=True, filt=True)),
        ("7_e2e_disambiguated_prompt", dict(kind="pipeline", model=str(MERGED_QWEN), triage=True, outlines=True, corr=True, filt=True, disamb=True)),
    ]

    results = {}
    for name, cfg in configs:
        out_jsonl = ab_dir / f"{name}.jsonl"
        perf_json = ab_dir / f"{name}_perf.json"
        if cfg["kind"] == "baseline":
            perf = run_baseline(test_rows, model_name=cfg["model"], output_file=out_jsonl, metrics_out=perf_json)
        else:
            perf = run_configured_pipeline(
                test_rows=test_rows,
                model_path=cfg["model"],
                output_file=out_jsonl,
                use_triage=cfg["triage"],
                use_outlines=cfg["outlines"],
                use_self_correction=cfg["corr"],
                use_fallback_filter=cfg["filt"],
                disambiguated_prompt=cfg.get("disamb", False),
                metrics_out=perf_json,
            )
        ev = _eval_jsonl(test_rows, out_jsonl, validator)
        results[name] = {
            "metrics": ev["overall"],
            "per_class": ev["per_class"],
            "confusion_matrix": ev["confusion_matrix"],
            "efficiency": perf,
        }

    # Mixed positive + negative End-to-End evaluation using existing qwen_test + distilbert_test negatives
    dist_rows = _load_jsonl(DATA_DIST_TEST)
    neg_rows = [{"text": r["text"], "label": 0, "event_data": None} for r in dist_rows if r["label"] == 0]
    pos_rows = [{"text": r["text"], "label": 1, "event_data": r["event_data"]} for r in test_rows]
    mixed_rows = pos_rows + neg_rows

    e2e_on_jsonl = ab_dir / "mixed_e2e_triage_on.jsonl"
    e2e_on_perf = run_configured_pipeline(
        test_rows=mixed_rows,
        model_path=str(MERGED_QWEN),
        output_file=e2e_on_jsonl,
        use_triage=True,
        use_outlines=True,
        use_self_correction=True,
        use_fallback_filter=True,
        metrics_out=ab_dir / "mixed_e2e_triage_on_perf.json",
    )
    e2e_off_jsonl = ab_dir / "mixed_e2e_triage_off.jsonl"
    e2e_off_perf = run_configured_pipeline(
        test_rows=mixed_rows,
        model_path=str(MERGED_QWEN),
        output_file=e2e_off_jsonl,
        use_triage=False,
        use_outlines=True,
        use_self_correction=True,
        use_fallback_filter=True,
        metrics_out=ab_dir / "mixed_e2e_triage_off_perf.json",
    )
    results["mixed_42_triage_on"] = {
        "eval": _eval_jsonl(mixed_rows, e2e_on_jsonl, validator),
        "efficiency": e2e_on_perf,
    }
    results["mixed_42_triage_off"] = {
        "eval": _eval_jsonl(mixed_rows, e2e_off_jsonl, validator),
        "efficiency": e2e_off_perf,
    }

    (REPORTS_DIR / "ablation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved ablation_metrics.json")


def run_larger_llms(validator) -> None:
    """Evaluate larger Qwen2.5-3B (fp16) and Qwen2.5-7B (4-bit NF4) baselines on T4."""
    test_rows = _load_jsonl(DATA_QWEN_TEST)
    out_dir = REPORTS_DIR / "larger_llms"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    specs = [
        ("Qwen2.5-3B-Instruct (Zero-Shot)", "Qwen/Qwen2.5-3B-Instruct", False, False),
        ("Qwen2.5-3B-Instruct (Zero-Shot + Outlines)", "Qwen/Qwen2.5-3B-Instruct", True, False),
        ("Qwen2.5-7B-Instruct-4bit (Zero-Shot)", "Qwen/Qwen2.5-7B-Instruct", False, True),
        ("Qwen2.5-7B-Instruct-4bit (Zero-Shot + Outlines)", "Qwen/Qwen2.5-7B-Instruct", True, True),
    ]
    for label, hub_id, use_outlines, use_4bit in specs:
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        out_jsonl = out_dir / f"{slug}.jsonl"
        perf_json = out_dir / f"{slug}_perf.json"
        if use_outlines:
            perf = run_configured_pipeline(
                test_rows=test_rows,
                model_path=hub_id,
                output_file=out_jsonl,
                use_triage=False,
                use_outlines=True,
                use_self_correction=False,
                use_fallback_filter=False,
                load_in_4bit=use_4bit,
                metrics_out=perf_json,
            )
        else:
            perf = run_baseline(
                test_rows=test_rows,
                model_name=hub_id,
                output_file=out_jsonl,
                load_in_4bit=use_4bit,
                metrics_out=perf_json,
            )
        ev = _eval_jsonl(test_rows, out_jsonl, validator)
        results[label] = {
            "metrics": ev["overall"],
            "per_class": ev["per_class"],
            "efficiency": perf,
        }

    (REPORTS_DIR / "larger_llm_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved larger_llm_metrics.json")


def run_sensitivity(validator) -> None:
    """Run LoRA rank sensitivity (r=4,8,16,32) and DistilBERT threshold sweep on T4."""
    test_rows = _load_jsonl(DATA_QWEN_TEST)
    sens_dir = REPORTS_DIR / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    rank_results = {}
    for r in (4, 8, 16, 32):
        alpha = 2 * r
        lora_dir = BASE / "models" / f"qwen_lora_r{r}"
        merged_dir = lora_dir / "merged"
        out_jsonl = sens_dir / f"rank_{r}.jsonl"
        perf_json = sens_dir / f"rank_{r}_perf.json"

        subprocess.run(
            [
                sys.executable,
                str(BASE / "training" / "train_qwen_lora.py"),
                "--output-dir",
                str(lora_dir),
                "--rank",
                str(r),
                "--alpha",
                str(alpha),
                "--skip-intermediate-eval",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(BASE / "training" / "merge_lora.py"),
                "--adapter-dir",
                str(lora_dir),
                "--output-dir",
                str(merged_dir),
            ],
            check=True,
        )
        perf = run_configured_pipeline(
            test_rows=test_rows,
            model_path=str(merged_dir),
            output_file=out_jsonl,
            use_triage=False,
            use_outlines=True,
            use_self_correction=True,
            use_fallback_filter=True,
            metrics_out=perf_json,
        )
        train_summary = json.loads((lora_dir / "train_summary.json").read_text(encoding="utf-8"))
        ev = _eval_jsonl(test_rows, out_jsonl, validator)
        rank_results[f"r={r}"] = {
            "rank": r,
            "alpha": alpha,
            "train_summary": train_summary,
            "metrics": ev["overall"],
            "inference_efficiency": perf,
        }
        import shutil
        shutil.rmtree(lora_dir, ignore_errors=True)

    # DistilBERT probability threshold sweep on distilbert_test.jsonl
    dist_rows = _load_jsonl(DATA_DIST_TEST)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR).to(device)
    mdl.eval()

    probs, labels = [], []
    with torch.no_grad():
        for row in dist_rows:
            enc = tok(row["text"], truncation=True, max_length=512, return_tensors="pt").to(device)
            p = torch.softmax(mdl(**enc).logits, dim=-1)[0, 1].item()
            probs.append(p)
            labels.append(int(row["label"]))

    thresh_results = {}
    for tau in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        preds = [1 if p >= tau else 0 for p in probs]
        tp = sum(1 for pr, y in zip(preds, labels) if pr == 1 and y == 1)
        fp = sum(1 for pr, y in zip(preds, labels) if pr == 1 and y == 0)
        tn = sum(1 for pr, y in zip(preds, labels) if pr == 0 and y == 0)
        fn = sum(1 for pr, y in zip(preds, labels) if pr == 0 and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(labels)
        thresh_results[str(tau)] = {
            "threshold": tau,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "pass_rate": (tp + fp) / len(labels),
        }

    out = {"lora_rank_sweep": rank_results, "distilbert_threshold_sweep": thresh_results}
    (REPORTS_DIR / "sensitivity_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved sensitivity_metrics.json")


def run_cv5(validator) -> None:
    """Run 5-fold stratified CV across all 280 positives and 71 negatives in /tmp without altering data/."""
    import random
    from collections import defaultdict

    raw_path = BASE / "data" / "raw" / "splittable_redo.jsonl"
    raw = [json.loads(line) for line in open(raw_path, "r", encoding="utf-8") if line.strip()]
    pos_seen, neg_seen = {}, {}
    for r in raw:
        t = r["text"].strip()
        if int(r.get("has_disruption", 0)) == 1 and t not in pos_seen:
            pos_seen[t] = {"text": t, "label": 1, "event_data": r["event_data"]}
        elif int(r.get("has_disruption", 0)) == 0 and t not in neg_seen:
            neg_seen[t] = {"text": t, "label": 0, "event_data": None}

    rng = random.Random(42)
    by_cls = defaultdict(list)
    for item in sorted(pos_seen.values(), key=lambda x: x["text"]):
        by_cls[item["event_data"]["event_type"]].append(item)

    pos_folds = [[] for _ in range(5)]
    for grp in by_cls.values():
        rng.shuffle(grp)
        for i, item in enumerate(grp):
            pos_folds[i % 5].append(item)

    neg_items = sorted(neg_seen.values(), key=lambda x: x["text"])
    rng.shuffle(neg_items)
    neg_folds = [[] for _ in range(5)]
    for i, item in enumerate(neg_items):
        neg_folds[i % 5].append(item)

    cv_tmp = Path("/tmp/cv_5fold")
    fold_summaries = []
    all_test_rows, all_pred_rows = [], []

    for k in range(5):
        fdir = cv_tmp / f"fold_{k}"
        fdir.mkdir(parents=True, exist_ok=True)
        test_pos = pos_folds[k]
        test_neg = neg_folds[k]
        train_pos = [x for j in range(5) if j != k for x in pos_folds[j]]
        train_neg = [x for j in range(5) if j != k for x in neg_folds[j]]

        q_train_p = fdir / "qwen_train.jsonl"
        q_test_p = fdir / "qwen_test.jsonl"
        d_train_p = fdir / "dist_train.jsonl"
        d_test_p = fdir / "dist_test.jsonl"
        q_train_p.write_text("\n".join(json.dumps({"text": x["text"], "event_data": x["event_data"]}, ensure_ascii=False) for x in train_pos) + "\n")
        q_test_p.write_text("\n".join(json.dumps({"text": x["text"], "event_data": x["event_data"]}, ensure_ascii=False) for x in test_pos) + "\n")
        d_train_p.write_text("\n".join(json.dumps({"text": x["text"], "label": x["label"]}, ensure_ascii=False) for x in (train_pos + train_neg)) + "\n")
        d_test_p.write_text("\n".join(json.dumps({"text": x["text"], "label": x["label"]}, ensure_ascii=False) for x in (test_pos + test_neg)) + "\n")

        lora_dir = fdir / "qwen_lora"
        merged_dir = fdir / "merged"
        dist_dir = fdir / "distilbert"
        pred_out = fdir / "preds.jsonl"

        subprocess.run(
            [sys.executable, str(BASE / "training" / "train_distilbert.py"), "--train-file", str(d_train_p), "--val-file", str(d_test_p), "--test-file", str(d_test_p), "--output-dir", str(dist_dir)],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(BASE / "training" / "train_qwen_lora.py"), "--train-file", str(q_train_p), "--val-file", str(q_test_p), "--output-dir", str(lora_dir), "--skip-val-eval"],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(BASE / "training" / "merge_lora.py"), "--adapter-dir", str(lora_dir), "--output-dir", str(merged_dir)],
            check=True,
        )
        perf = run_configured_pipeline(
            test_rows=test_pos,
            model_path=str(merged_dir),
            output_file=pred_out,
            use_triage=False,
            use_outlines=True,
            use_self_correction=True,
            use_fallback_filter=True,
        )
        ev = _eval_jsonl(test_pos, pred_out, validator)
        fold_summaries.append({"fold": k, "test_count": len(test_pos), "metrics": ev["overall"], "per_class": ev["per_class"], "efficiency": perf})
        all_test_rows.extend(test_pos)
        all_pred_rows.extend(_load_jsonl(pred_out))
        import shutil
        shutil.rmtree(fdir, ignore_errors=True)

    pooled = evaluate_prediction_file(all_test_rows, all_pred_rows, validator)
    f1s = [f["metrics"]["f1"] for f in fold_summaries]
    tops = [f["metrics"]["top_level_f1"] for f in fold_summaries]
    args_f1 = [f["metrics"]["other_fields_f1"] for f in fold_summaries]

    cv_report = {
        "num_folds": 5,
        "total_out_of_fold_positives": len(all_test_rows),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_top_level_f1": float(np.mean(tops)),
        "std_top_level_f1": float(np.std(tops)),
        "mean_other_fields_f1": float(np.mean(args_f1)),
        "std_other_fields_f1": float(np.std(args_f1)),
        "pooled_280_eval": pooled,
        "folds": fold_summaries,
    }
    (REPORTS_DIR / "cv_5fold_metrics.json").write_text(json.dumps(cv_report, indent=2), encoding="utf-8")
    print("Saved cv_5fold_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T4 GPU benchmark suites.")
    parser.add_argument(
        "--suite",
        required=True,
        choices=["ablations", "larger_llms", "comparisons", "variance", "sensitivity", "cv5", "all"],
    )
    args = parser.parse_args()
    validator = load_schema_validator(SCHEMA_FILE)

    if args.suite in ("ablations", "all"):
        run_ablations(validator)
    if args.suite in ("larger_llms", "all"):
        run_larger_llms(validator)
    if args.suite in ("comparisons", "all"):
        subprocess.run([sys.executable, str(BASE / "training" / "run_model_comparisons.py"), "--force-retrain"], check=True)
        subprocess.run([sys.executable, str(BASE / "evaluation" / "compare_extraction.py")], check=True)
    if args.suite in ("variance", "all"):
        subprocess.run([sys.executable, str(BASE / "training" / "run_variance_tests.py")], check=True)
    if args.suite in ("sensitivity", "all"):
        run_sensitivity(validator)
    if args.suite in ("cv5", "all"):
        run_cv5(validator)


if __name__ == "__main__":
    main()
