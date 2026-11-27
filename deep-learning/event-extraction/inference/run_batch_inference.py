"""Configurable T4 batch inference & system profiling runner."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import psutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from inference.pipeline_core import (
    DISAMBIGUATION_ADDENDUM,
    SYSTEM_PROMPT,
    parse_json_completion,
    run_document_pipeline,
)

TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
REPORTS_DIR = BASE / "reports"
PIPELINE_OUT = REPORTS_DIR / "structured_outputs.jsonl"
BASELINE_OUT = REPORTS_DIR / "baseline_structured_outputs.jsonl"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
DISTILBERT_DIR = BASE / "models" / "distilbert"
QWEN_MERGED_DIR = BASE / "models" / "qwen_lora" / "merged"
BASELINE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _load_causal_model(model_path: str, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def run_configured_pipeline(
    test_rows: list,
    model_path: str,
    output_file: Path,
    use_triage: bool = True,
    triage_dir: Path = DISTILBERT_DIR,
    triage_threshold: float = 0.5,
    use_outlines: bool = True,
    use_self_correction: bool = True,
    use_fallback_filter: bool = True,
    field_aware_grounding: bool = True,
    disambiguated_prompt: bool = False,
    load_in_4bit: bool = False,
    metrics_out: Path = None,
) -> dict:
    """Execute extraction pipeline over test_rows and optionally record T4 efficiency stats."""
    triage_model, triage_tokenizer, triage_device = None, None, None
    t_triage_load = 0.0
    if use_triage:
        t0 = time.perf_counter()
        triage_tokenizer = AutoTokenizer.from_pretrained(triage_dir)
        triage_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        triage_model = AutoModelForSequenceClassification.from_pretrained(triage_dir).to(triage_device)
        triage_model.eval()
        t_triage_load = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    qwen_model, qwen_tokenizer = _load_causal_model(str(model_path), load_in_4bit=load_in_4bit)
    t_qwen_load = time.perf_counter() - t0

    generator, outlines_schema = None, None
    if use_outlines:
        from outlines import from_transformers, json_schema

        generator = from_transformers(qwen_model, qwen_tokenizer)
        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        outlines_schema = json_schema(schema)

    prompt_str = SYSTEM_PROMPT + (DISAMBIGUATION_ADDENDUM if disambiguated_prompt else "")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    doc_latencies, triage_ms_list, pass1_list, pass2_list = [], [], [], []
    total_chunks, total_triage_pos, total_corr, total_tokens = 0, 0, 0, 0
    t_wall_start = time.perf_counter()

    with open(output_file, "w", encoding="utf-8") as out_file:
        for row in tqdm(test_rows, desc=f"Infer({Path(str(model_path)).name})"):
            t_doc0 = time.perf_counter()
            extracted_events, st = run_document_pipeline(
                text=row["text"],
                qwen_model=qwen_model,
                qwen_tokenizer=qwen_tokenizer,
                triage_model=triage_model,
                triage_tokenizer=triage_tokenizer,
                triage_device=triage_device,
                triage_threshold=triage_threshold,
                generator=generator,
                outlines_schema=outlines_schema,
                use_self_correction=use_self_correction,
                use_fallback_filter=use_fallback_filter,
                field_aware_grounding=field_aware_grounding,
                system_prompt=prompt_str,
            )
            doc_lat = time.perf_counter() - t_doc0
            doc_latencies.append(doc_lat)
            total_chunks += st["chunks"]
            total_triage_pos += st["triage_positives"]
            total_corr += st["correction_triggered"]
            total_tokens += st["tokens_generated"]
            if st["chunks"] > 0 and use_triage:
                triage_ms_list.append((st["triage_s"] / st["chunks"]) * 1000.0)
            if st["pass1_s"] > 0:
                pass1_list.append(st["pass1_s"])
            if st["pass2_s"] > 0:
                pass2_list.append(st["pass2_s"])

            real_events = [e for e in extracted_events if e.get("event_type") != "NoEvent"]
            if real_events:
                out_file.write(json.dumps(real_events[0], ensure_ascii=False) + "\n")
            elif extracted_events:
                out_file.write(json.dumps(extracted_events[0], ensure_ascii=False) + "\n")
            else:
                out_file.write("null\n")

    wall_total_s = time.perf_counter() - t_wall_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    peak_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    perf = {
        "model": str(model_path),
        "num_docs": len(test_rows),
        "use_triage": use_triage,
        "use_outlines": use_outlines,
        "use_self_correction": use_self_correction,
        "use_fallback_filter": use_fallback_filter,
        "load_in_4bit": load_in_4bit,
        "triage_load_seconds": round(t_triage_load, 4),
        "model_load_seconds": round(t_qwen_load, 4),
        "total_wall_seconds": round(wall_total_s, 3),
        "throughput_docs_per_min": round(len(test_rows) / (wall_total_s / 60.0), 3) if wall_total_s > 0 else 0.0,
        "latency_mean_s": round(float(np.mean(doc_latencies)), 3) if doc_latencies else 0.0,
        "latency_median_s": round(float(np.median(doc_latencies)), 3) if doc_latencies else 0.0,
        "latency_p95_s": round(float(np.percentile(doc_latencies, 95)), 3) if doc_latencies else 0.0,
        "triage_mean_ms_per_chunk": round(float(np.mean(triage_ms_list)), 3) if triage_ms_list else 0.0,
        "pass1_mean_s": round(float(np.mean(pass1_list)), 3) if pass1_list else 0.0,
        "pass2_mean_s": round(float(np.mean(pass2_list)), 3) if pass2_list else 0.0,
        "triage_pass_rate": round(total_triage_pos / max(1, total_chunks), 4),
        "correction_trigger_rate": round(total_corr / max(1, total_triage_pos), 4),
        "total_tokens_generated": int(total_tokens),
        "effective_tokens_per_second": round(total_tokens / wall_total_s, 2) if wall_total_s > 0 else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(perf, indent=2), encoding="utf-8")

    del qwen_model, triage_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return perf


def run_baseline(test_rows: list, model_name: str = BASELINE_MODEL_NAME, output_file: Path = BASELINE_OUT, load_in_4bit: bool = False, metrics_out: Path = None) -> dict:
    """Run zero-shot unconstrained prompt baseline with full JSON schema in system prompt."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tokenizer = _load_causal_model(model_name, load_in_4bit=load_in_4bit)
    t_load = time.perf_counter() - t0
    schema_str = json.dumps(json.loads(SCHEMA_FILE.read_text(encoding="utf-8")), indent=2)
    device = next(model.parameters()).device

    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc_latencies, total_tokens = [], 0
    t_wall_start = time.perf_counter()

    with open(output_file, "w", encoding="utf-8") as out_file:
        for row in tqdm(test_rows, desc=f"Baseline({Path(model_name).name})"):
            t_doc0 = time.perf_counter()
            text = row["text"]
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            chunks, current_chunk, current_len = [], [], 0
            for para in paragraphs:
                para_len = len(tokenizer.tokenize(para))
                if current_len + para_len > 1500 and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk, current_len = [para], para_len
                else:
                    current_chunk.append(para)
                    current_len += para_len
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            if not chunks:
                out_file.write("null\n")
                continue

            extracted_events = []
            for chunk in chunks:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert supply chain disruption event extractor. "
                            "Extract a single event matching the provided JSON schema. "
                            "Respond ONLY with a JSON object. Do not wrap the JSON in markdown code blocks or add any explanatory text.\n\n"
                            "Guidelines:\n"
                            "- Extract only information that is explicitly stated in the source text.\n"
                            "- Do not infer, estimate, or hallucinate any facts.\n"
                            "- If a value is not explicitly mentioned, return null for that field.\n"
                            "- Extract the date of the event and format it as an ISO 8601 string (e.g., YYYY-MM-DDT00:00:00Z) in the source_timestamp field. If only a year is mentioned, use the first day of that year (e.g., YYYY-01-01T00:00:00Z). If only a month and year are mentioned, use the first day of that month (e.g., YYYY-MM-01T00:00:00Z). If no date/time is mentioned, return null.\n"
                            "- Preserve the original meaning of the text.\n"
                            "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
                            "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as bankruptcy, Chapter 11, liquidation, or receivership. Do not classify temporary operational shutdowns, resource depletion (e.g., running out of fuel), or physical facility halts as SupplierInsolvency.\n\n"
                            f"JSON Schema:\n{schema_str}"
                        ),
                    },
                    {"role": "user", "content": chunk},
                ]
                prompt = (
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if hasattr(tokenizer, "apply_chat_template")
                    else f"Extract JSON matching schema:\n{schema_str}\n\nText:\n{chunk}\n"
                )
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    generated = model.generate(
                        **{k: v.to(device) for k, v in inputs.items()},
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = generated[0][inputs["input_ids"].shape[1] :]
                total_tokens += len(gen_ids)
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                parsed = parse_json_completion(completion)
                if parsed is not None:
                    extracted_events.append(parsed)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            doc_latencies.append(time.perf_counter() - t_doc0)

            real_events = [e for e in extracted_events if isinstance(e, dict) and e.get("event_type") != "NoEvent"]
            if real_events:
                out_file.write(json.dumps(real_events[0], ensure_ascii=False) + "\n")
            elif extracted_events:
                out_file.write(json.dumps(extracted_events[0], ensure_ascii=False) + "\n")
            else:
                out_file.write("null\n")

    wall_total_s = time.perf_counter() - t_wall_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    peak_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    perf = {
        "model": model_name,
        "num_docs": len(test_rows),
        "load_in_4bit": load_in_4bit,
        "model_load_seconds": round(t_load, 4),
        "total_wall_seconds": round(wall_total_s, 3),
        "throughput_docs_per_min": round(len(test_rows) / (wall_total_s / 60.0), 3) if wall_total_s > 0 else 0.0,
        "latency_mean_s": round(float(np.mean(doc_latencies)), 3) if doc_latencies else 0.0,
        "latency_median_s": round(float(np.median(doc_latencies)), 3) if doc_latencies else 0.0,
        "latency_p95_s": round(float(np.percentile(doc_latencies, 95)), 3) if doc_latencies else 0.0,
        "total_tokens_generated": int(total_tokens),
        "effective_tokens_per_second": round(total_tokens / wall_total_s, 2) if wall_total_s > 0 else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(perf, indent=2), encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return perf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference on test JSONL.")
    parser.add_argument("--model", default="both", choices=["pipeline", "baseline", "both"])
    parser.add_argument("--model-dir", default=str(QWEN_MERGED_DIR))
    parser.add_argument("--base-model", default=BASELINE_MODEL_NAME)
    parser.add_argument("--input-file", type=Path, default=TEST_FILE)
    parser.add_argument("--output-file", type=Path, default=PIPELINE_OUT)
    parser.add_argument("--baseline-output-file", type=Path, default=BASELINE_OUT)
    parser.add_argument("--triage-dir", type=Path, default=DISTILBERT_DIR)
    parser.add_argument("--triage-threshold", type=float, default=0.5)
    parser.add_argument("--no-triage", action="store_true")
    parser.add_argument("--no-outlines", action="store_true")
    parser.add_argument("--no-self-correction", action="store_true")
    parser.add_argument("--no-fallback-filter", action="store_true")
    parser.add_argument("--legacy-grounding", action="store_true")
    parser.add_argument("--disambiguated-prompt", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args()

    test_rows = [json.loads(line) for line in open(args.input_file, "r", encoding="utf-8") if line.strip()]
    print(f"Loaded {len(test_rows)} test cases from {args.input_file}")

    if args.model in ("pipeline", "both"):
        run_configured_pipeline(
            test_rows=test_rows,
            model_path=args.model_dir,
            output_file=args.output_file,
            use_triage=not args.no_triage,
            triage_dir=args.triage_dir,
            triage_threshold=args.triage_threshold,
            use_outlines=not args.no_outlines,
            use_self_correction=not args.no_self_correction,
            use_fallback_filter=not args.no_fallback_filter,
            field_aware_grounding=not args.legacy_grounding,
            disambiguated_prompt=args.disambiguated_prompt,
            load_in_4bit=args.load_in_4bit,
            metrics_out=args.metrics_out if args.model == "pipeline" else None,
        )

    if args.model in ("baseline", "both"):
        run_baseline(
            test_rows=test_rows,
            model_name=args.base_model,
            output_file=args.baseline_output_file,
            load_in_4bit=args.load_in_4bit,
            metrics_out=args.metrics_out if args.model == "baseline" else None,
        )


if __name__ == "__main__":
    main()
