"""Fine-tune causal SLM with LoRA adapters on supply chain event extraction."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from evaluation.metrics_utils import TOP_LEVEL_KEYS, evaluate_sample, normalize_prediction

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM_PROMPT = (
    "You are an expert supply chain disruption event extractor. "
    "Extract a single event matching the provided JSON schema.\n\n"
    "Guidelines:\n"
    "- Extract only information that is explicitly stated in the source text.\n"
    "- Do not infer, estimate, or hallucinate any facts.\n"
    "- If a value is not explicitly mentioned, return null for that field.\n"
    "- Extract the date of the event and format it as an ISO 8601 string (e.g., YYYY-MM-DDT00:00:00Z) in the source_timestamp field. If only a year is mentioned, use the first day of that year (e.g., YYYY-01-01T00:00:00Z). If only a month and year are mentioned, use the first day of that month (e.g., YYYY-MM-01T00:00:00Z). If no date/time is mentioned, return null.\n"
    "- Preserve the original meaning of the text.\n"
    "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
    "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as bankruptcy, Chapter 11, liquidation, or receivership. Do not classify temporary operational shutdowns, resource depletion (e.g., running out of fuel), or physical facility halts as SupplierInsolvency."
)


def load_valid_rows(filepath: Path) -> list:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("event_data") is not None:
                    rows.append(row)
    return rows


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    # On T4 (compute capability 7.5), native fp16 Tensor Cores are ~2.8x faster than emulated bf16
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for structured event extraction.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=BASE / "data" / "qwen" / "qwen_train.jsonl")
    parser.add_argument("--val-file", type=Path, default=BASE / "data" / "qwen" / "qwen_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "models" / "qwen_lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--skip-intermediate-eval", action="store_true")
    parser.add_argument("--skip-val-eval", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    train_dtype = resolve_dtype(args.dtype)
    print(f"Accelerate utilizing device: {device} | dtype: {train_dtype} | seed: {args.seed} | rank: {args.rank}")

    train_rows = load_valid_rows(args.train_file)
    val_rows = load_valid_rows(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=train_dtype,
        device_map={"": device},
    )
    base_model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora)
    model.enable_input_require_grads()

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.generation_config = generation_config

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_encoded = []
    for row in train_rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": json.dumps(row["event_data"], ensure_ascii=False)})
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids[: args.max_length]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]
        train_encoded.append(
            {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    val_prompts = []
    for row in val_rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        val_prompts.append((prompt, row["event_data"]))

    best_output_dir = args.output_dir / "best"

    def save_checkpoint(target_dir: Path, metrics=None, filename="metrics.json"):
        if accelerator.is_main_process:
            target_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(target_dir)
            tokenizer.save_pretrained(target_dir)
            if metrics:
                (target_dir / filename).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_train_start = time.perf_counter()
    best_top_level_f1 = -float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(range(0, len(train_encoded), args.batch_size), desc=f"Train Epoch {epoch + 1}")
        for start_idx in progress_bar:
            batch = train_encoded[start_idx : start_idx + args.batch_size]
            if not batch:
                continue
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            labels = torch.nn.utils.rnn.pad_sequence(
                [b["labels"] for b in batch], batch_first=True, padding_value=-100
            ).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, batch_count)
        should_eval = not args.skip_val_eval and (not args.skip_intermediate_eval or epoch == args.epochs - 1)
        avg_top_f1, avg_other_f1 = 0.0, 0.0

        if should_eval:
            model.eval()
            val_top_scores, val_other_scores = [], []
            with torch.inference_mode():
                for prompt, gold_event in tqdm(val_prompts, desc=f"Eval Epoch {epoch + 1}", leave=False):
                    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                    prompt_ids = prompt_inputs["input_ids"].to(device)
                    attention_mask = prompt_inputs["attention_mask"].to(device)
                    generated_ids = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                    )
                    generated_text = tokenizer.decode(generated_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True)
                    predicted_event = normalize_prediction(generated_text)
                    _, _, _, top_f1, other_f1 = evaluate_sample(predicted_event, gold_event)
                    val_top_scores.append(top_f1)
                    val_other_scores.append(other_f1)

            avg_top_f1 = float(np.mean(val_top_scores)) if val_top_scores else 0.0
            avg_other_f1 = float(np.mean(val_other_scores)) if val_other_scores else 0.0

        epoch_metrics = {
            "epoch": epoch + 1,
            "top_level_f1": avg_top_f1,
            "other_fields_f1": avg_other_f1,
            "train_loss": avg_train_loss,
        }
        save_checkpoint(args.output_dir, metrics=epoch_metrics, filename=f"epoch_{epoch + 1}_metrics.json")

        if should_eval and avg_top_f1 > best_top_level_f1:
            best_top_level_f1 = avg_top_f1
            best_metrics = {"epoch": epoch + 1, "top_level_f1": avg_top_f1, "other_fields_f1": avg_other_f1}
            save_checkpoint(best_output_dir, metrics=best_metrics, filename="best_metrics.json")

        print(
            f"\n[Epoch {epoch + 1} Summary] Train Loss: {avg_train_loss:.4f} | "
            f"Top Level F1: {avg_top_f1:.4f} | Other Fields F1: {avg_other_f1:.4f}"
        )

    train_wall_s = time.perf_counter() - t_train_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    train_summary = {
        "base_model": args.base_model,
        "seed": args.seed,
        "rank": args.rank,
        "alpha": args.alpha,
        "dtype": str(train_dtype),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_wall_seconds": round(train_wall_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "final_train_loss": avg_train_loss,
    }
    save_checkpoint(args.output_dir, metrics=train_summary, filename="train_summary.json")
    print(f"Training complete in {train_wall_s:.1f}s | Peak VRAM: {peak_vram_mb:.1f} MB | Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
