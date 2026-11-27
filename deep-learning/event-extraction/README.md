# Supply Chain Event Extraction Pipeline

> **Internship Project · Industry Experience Report Reference Implementation**

A resource-efficient, two-stage pipeline for detecting and extracting structured supply chain disruption events from unstructured text, using small language models (SLMs) with LoRA fine-tuning.

---

## Overview

Enterprise organizations monitor large volumes of news and operational documents to detect supply chain risks. Sending every document to a large generative LLM is computationally expensive. This pipeline solves that with a **two-stage triage architecture**:

1. **Stage 1 — Triage (DistilBERT, 66M params):** Fast binary classifier that filters out non-event chunks in **8.17 ms/chunk** on an NVIDIA T4 GPU (93.33% accuracy, 100.00% recall, 94.74% F1 at $\tau = 0.5$).
2. **Stage 2 — Extraction (Qwen2.5-1.5B + LoRA + Outlines):** Structured JSON event extraction with FSM-constrained decoding and field-aware grounding validation, only triggered for chunks that pass triage (**83.71% Overall F1**, **89.78% Top-Level F1**, **80.04% Argument-Level F1**, and **100% schema validity** within **3.40 GB VRAM** on a single NVIDIA T4 GPU).

### Supported Event Types

| Event Type | Captures |
|---|---|
| `FacilityHalt` | Physical production stoppages (fires, strikes, disasters, cyberattacks) |
| `ShipmentDelay` | Transit and carrier delays |
| `SupplierInsolvency` | Bankruptcy, liquidation, and financial restructuring |
| `TariffChange` | Customs duty and trade policy changes |
| `ForceMajeure` | Legal force majeure declarations |

---

## Repository Structure

```
Triage_Pipeline/
│
├── run_all.py                     # End-to-end pipeline runner (stages 0-7)
├── requirements.txt               # Python dependencies
├── wiki_links.txt                 # References for event taxonomy validation
│
├── data/
│   ├── distilbert/          # Binary triage classification datasets
│   │   ├── distilbert_base.jsonl   (196 rows: 125 pos / 71 neg)
│   │   ├── distilbert_train.jsonl  (137 rows: 83 pos / 54 neg)
│   │   ├── distilbert_val.jsonl    (29 rows)
│   │   └── distilbert_test.jsonl   (30 rows)
│   │
│   ├── qwen/                # Structured extraction datasets (positives only)
│   │   ├── qwen_base.jsonl         (175 rows, balanced ~35/class)
│   │   ├── qwen_train.jsonl        (115 rows)
│   │   ├── qwen_val.jsonl          (30 rows, 6/class)
│   │   └── qwen_test.jsonl         (30 rows, 6/class)
│   │
│   ├── raw/
│   │   └── splittable_redo.jsonl   # Master unified dataset (351 rows: 280 pos / 71 neg)
│
├── schemas/
│   └── extraction_schema.json   # JSON Schema governing all extractions
│
├── training/
│   ├── split_dataset.py           # Reads splittable_redo.jsonl → outputs to data/
│   ├── train_distilbert.py        # Fine-tunes DistilBERT for binary triage
│   ├── train_qwen_lora.py         # Fine-tunes Qwen2.5-1.5B with LoRA
│   ├── merge_lora.py              # Merges LoRA adapter into base model
│   ├── run_model_comparisons.py   # Trains/evaluates comparison models (SmolLM2, TinyLlama)
│   └── run_variance_tests.py      # Re-trains Qwen with seeds 42/1337/2026 for variance assessment
│
├── models/
│   ├── distilbert/          # Fine-tuned DistilBERT weights (gitignored: .safetensors)
│   ├── qwen_lora/           # LoRA adapter + merged model (gitignored: .safetensors)
│   ├── SmolLM2-1.7B-Instruct_lora/  # Comparison model (gitignored: .safetensors)
│   └── TinyLlama-1.1B-Chat-v1.0_lora/  # Comparison model (gitignored: .safetensors)
│
├── inference/
│   ├── extract_baseline_test.py   # Baseline extraction testing
│   ├── pipeline_core.py           # Shared chunking, triage, FSM decoding, and field-aware grounding
│   ├── run_batch_inference.py     # Configurable T4 batch inference & profiling runner
│   └── triage_pipeline.py         # Single-sample inference pipeline
│
├── evaluation/
│   ├── compare_extraction.py      # F1 / Precision / Recall / Schema Validity & per-class metrics
│   ├── compare_validation.py      # JSON schema validation check on all outputs
│   ├── evaluate_distilbert.py     # DistilBERT classifier test-set evaluation
│   ├── calculate_parameters.py    # LoRA parameter count and adapter size statistics
│   ├── evaluate_forgetting.py     # Catastrophic forgetting assessment on general tasks
│   ├── metrics_utils.py           # Shared evaluation, ISO-date parsing, and schema validator helpers
│   └── run_t4_benchmarks.py       # Unified T4 GPU benchmark suite (ablations, larger LLMs, sensitivity, CV)
│
├── reports/
│   ├── structured_outputs.jsonl        # Pipeline predictions (30 test samples)
│   ├── baseline_structured_outputs.jsonl  # Zero-shot baseline predictions
│   ├── SmolLM2-1.7B-Instruct_pipeline.jsonl
│   ├── TinyLlama-1.1B-Chat-v1.0_pipeline.jsonl
│   ├── comparison_metrics.csv          # Aggregated model comparison results
│   ├── comparison_metrics.png          # Results bar chart
│   ├── lora_heatmaps.png               # Layer-wise LoRA weight norm heatmap
│   ├── generate_heatmaps.py            # Generates LoRA heatmaps from adapter
│   ├── distilbert_metrics.py           # Calculates precision/recall/F1 for DistilBERT
│   ├── calculate_metrics.py            # Generates comparative evaluation JSON files
│   ├── ablation_metrics.json           # Component ablations & Mixed-42 E2E triage benchmarks
│   ├── larger_llm_metrics.json         # Zero-shot Qwen2.5-3B & 7B-4bit T4 benchmarks
│   ├── per_class_metrics.json          # Per-class F1 & confusion matrices across all models
│   ├── sensitivity_metrics.json        # LoRA rank sweep (r=4,8,16,32) & triage threshold sweep
│   ├── triage_metrics.json             # DistilBERT evaluation results across splits
│   ├── extraction_metrics.json         # Comparative extraction metrics
│   ├── system_performance_metrics.json  # Latencies, load times, and RAM metrics
│   ├── lora_weight_metrics.json        # PEFT matrix weight change norms
│   ├── perplexity_metrics.json         # Perplexity scores for domain drift
│   ├── dataset_metrics.json            # Dataset split statistics
│   └── variance_metrics.json           # F1 variance metrics across seeds (42/1337/2026)
│
├── Appendix.pdf                        # Original supplementary appendix
├── Appendix_camera_ready.tex           # Camera-ready supplementary appendix (LaTeX)
└── Appendix_camera_ready.pdf           # Camera-ready supplementary appendix (PDF)
```

> **Note:** Model weight files (`.safetensors`) are excluded from this repository due to size. Train them following the instructions below, or download from the companion release.

---

## Dataset

The unified master dataset is `data/raw/splittable_redo.jsonl`. It is the single source of truth that feeds both model training stages.

| Stage | Total | Positive | Negative | Positive Rate |
|---|---|---|---|---|
| Raw master (`splittable_redo.jsonl`) | **351** | **280** | **71** | 79.8% |
| DistilBERT base | 196 | 125 | 71 | 63.8% |
| DistilBERT train | 137 | 83 | 54 | 60.6% |
| DistilBERT val | 29 | 24 | 5 | 82.8% |
| DistilBERT test | 30 | 18 | 12 | 60.0% |
| Qwen base | 175 | 175 | — | 100% |
| Qwen train | 115 | 115 | — | 100% |
| Qwen val | 30 | 30 | — | 100% |
| Qwen test | 30 | 30 | — | 100% |

**Qwen base event type distribution** (near-uniform, 35 per class):

| Event Type | Count | % |
|---|---|---|
| FacilityHalt | 36 | 20.6% |
| ShipmentDelay | 36 | 20.6% |
| SupplierInsolvency | 35 | 20.0% |
| TariffChange | 35 | 20.0% |
| ForceMajeure | 33 | 18.9% |

---

## Quickstart

### 1. Install Dependencies
> **Requirement:** Python 3.11 is required.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
You can run the entire pipeline (data preparation, training, merging, inference, comparative model runs, variance runs, evaluation scripts, and report metrics generation) with a single command:

```bash
python run_all.py
```

#### Pipeline Stages
The pipeline execution is divided into the following numbered stages:
* **Stage 0: Data Preparation** — Runs `split_dataset.py` to partition `splittable_redo.jsonl` into stratified splits.
* **Stage 1: Model Training** — Trains the DistilBERT triage gate classifier and the Qwen2.5-1.5B LoRA extractor.
* **Stage 2: LoRA Adapter Merge** — Merges the trained Qwen LoRA adapter into the base model.
* **Stage 3: Inference** — Runs batch inference over the test set using the merged pipeline.
* **Stage 4: Comparison Models** — Trains and runs batch inference on TinyLlama and SmolLM2 for comparison.
* **Stage 5: Variance Testing** — Retrains the Qwen LoRA pipeline with alternative seeds (1337, 2026).
* **Stage 6: Evaluation** — Runs all evaluation scripts (extraction F1, schema validity, DistilBERT metrics, forgetting).
* **Stage 7: Reports and Visualizations** — Computes comparison JSON metrics and plots LoRA adapter weight change heatmaps.

#### Resuming or Running Specific Stages
* Run a single stage only:
  ```bash
  python run_all.py --only 6
  ```
* Resume pipeline execution from a specific stage:
  ```bash
  python run_all.py --from 3
  ```

### 3. Step-by-Step Training & Run Commands
If you prefer running individual scripts manually instead of using `run_all.py`, follow these commands:

```bash
# Step A: Prepare dataset splits
python training/split_dataset.py

# Step B: Train triage classifier
python training/train_distilbert.py

# Step C: Train extraction model (Qwen LoRA)
python training/train_qwen_lora.py

# Step D: Merge the adapter weights
python training/merge_lora.py
```

### 4. Single-Sample Inference
Test the end-to-end pipeline (Triage + Extraction) on a single string snippet:
```bash
python inference/triage_pipeline.py "A major port strike has halted container shipments at Rotterdam for 3 days."
```

**Output:**
```json
{
  "event_type": "ShipmentDelay",
  "source_timestamp": null,
  "text_evidence": "halted container shipments at Rotterdam for 3 days",
  "arguments": {
    "carrier": null,
    "origin": "Rotterdam",
    "destination": null,
    "delay_duration_days": 3,
    "reason": "major port strike"
  },
  "event_id": "EVT-XXXXXXXX"
}
```

### 5. Evaluate Individual Modules
```bash
# Evaluate Extraction (F1 / Recall / Precision)
python evaluation/compare_extraction.py

# Evaluate Triage Classifier (DistilBERT)
python evaluation/evaluate_distilbert.py
```

---

## Key Results

### 1. Stage 1: DistilBERT Triage Classifier Performance
The binary classifier is optimized to prevent false negatives (missed disruption events). It achieves the following performance metrics:

| Split / Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | True Positives | False Positives | True Negatives | False Negatives |
|---|---|---|---|---|---|---|---|---|
| **Test Set ($\tau=0.5$, T4)** | **93.33%** | **90.00%** | **100.00%** | **94.74%** | 18 | 2 | 10 | 0 |
| **Test Set ($\tau \in \{0.1,0.2,0.3\}$)** | 90.00% | 85.71% | 100.00% | 92.31% | 18 | 3 | 9 | 0 |
| **Val Set ($\tau=0.5$)** | 96.55% | 100.00% | 95.83% | 97.87% | 23 | 0 | 5 | 1 |
| **Train Set ($\tau=0.5$)** | 100.00% | 100.00% | 100.00% | 100.00% | 83 | 0 | 54 | 0 |

* **100.00% Recall at the Gate:** The triage classifier guarantees that no actual supply chain events are dropped across $\tau \in [0.1, 0.9]$.
* **80% False-Alert Reduction (Mixed-42 E2E):** On the 42-document mixed stream (30 positives + 12 negatives), triage blocks 10/12 non-event documents, lifting overall F1 from 58.80% to **77.85%** and cutting T4 wall-time by 37% (854.8 s vs. 1355.6 s).

### 2. Stage 2: Event Extraction and Argument Fill (Uniform NVIDIA T4 GPU)

| Model / Configuration | Precision | Recall | F1-Score | Top-Level F1 | Arguments F1 | Schema Validity |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (Zero-Shot) | 46.96% | 47.65% | 46.81% | 58.40% | 39.28% | 23.33% |
| Qwen2.5-3B (Zero-Shot) | 66.43% | 62.94% | 64.45% | 67.81% | 62.68% | 63.33% |
| Qwen2.5-7B-4bit (Zero-Shot) | 65.31% | 65.45% | 65.34% | 68.57% | 63.42% | 90.00% |
| **Qwen2.5-1.5B (LoRA Pipeline)** | **84.22%** | **83.41%** | **83.71%** | **89.78%** | **80.04%** | **100.00%** |
| SmolLM2-1.7B (LoRA) | 72.74% | 69.90% | 70.77% | 78.15% | 65.29% | 100.00% |
| TinyLlama-1.1B (LoRA) | 72.26% | 68.39% | 69.59% | 68.07% | 70.66% | 100.00% |

- **+78.84% relative Overall F1 improvement** over zero-shot 1.5B baseline (46.81% → 83.71%; Top-Level +53.72%, Argument-Level +103.76%)
- **3-Seed Reproducibility (`{42, 1337, 2026}`):** **82.61% ± 1.60%** Overall F1, **88.51% ± 2.08%** Top-Level F1, **78.87% ± 1.36%** Argument-Level F1, 100% schema validity
- **100% schema validity** via Outlines FSM-constrained decoding (vs. 23.33% baseline)
- **Efficient T4 Footprint:** Full pipeline runs within **3.40 GB VRAM** (22% of one T4) and **2.16 GB host RAM**, executing Stage-1 triage in **8.17 ms/chunk**.
- **Full Benchmark Suite:** Run `python evaluation/run_t4_benchmarks.py --suite all` to reproduce ablations, 3B/7B comparisons, rank/threshold sensitivity, and 5-fold CV.

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Rank (r) / Alpha (α) | 16 / 32 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable Parameters | 18,464,768 (~18.46M; 1.196% of base) |
| Training / Inference Dtype | `float16` (T4 GPU) / `bfloat16` (sm >= 80) / `float32` (CPU) |

---

## LoRA Rank Sensitivity (NVIDIA T4 GPU, Seed 42)

| Rank | Trainable Params | Final Loss | Train Time | Train Peak VRAM | Overall F1 | Top-Level F1 | Arg F1 |
|---|---|---|---|---|---|---|---|
| `r=4` | 4.62M | 0.041 | 585.5 s | 9.38 GB | 79.66% | 86.91% | 75.50% |
| `r=8` | 9.23M | 0.033 | 584.4 s | 9.44 GB | 80.97% | 88.22% | 76.08% |
| `r=16` (Default) | 18.46M | 0.026 | 589.4 s | 9.54 GB | 80.39% | 85.57% | 77.08% |
| `r=32` | 36.93M | 0.019 | 605.7 s | 9.77 GB | **89.14%** | **93.74%** | **86.22%** |

---

## Citation

> Paper under preparation. BibTeX entry will be added upon publication.

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
