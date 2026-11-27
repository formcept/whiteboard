"""Fine-tune DistilBERT binary triage classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
MODEL_NAME = "distilbert-base-uncased"


def _load_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _eval_split(model, tokenizer, rows: list, batch_size: int, max_length: int, device) -> tuple:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts = [row["text"] for row in batch_rows]
            labels = torch.tensor([row["label"] for row in batch_rows], dtype=torch.long, device=device)
            enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            preds = torch.argmax(model(**enc).logits, dim=-1)
            preds_all.extend(preds.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    preds_arr = np.asarray(preds_all)
    labels_arr = np.asarray(labels_all)
    acc = float((preds_arr == labels_arr).mean()) if len(labels_arr) else 0.0
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return acc, precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DistilBERT binary triage classifier.")
    parser.add_argument("--train-file", type=Path, default=BASE / "data" / "distilbert" / "distilbert_train.jsonl")
    parser.add_argument("--val-file", type=Path, default=BASE / "data" / "distilbert" / "distilbert_val.jsonl")
    parser.add_argument("--test-file", type=Path, default=BASE / "data" / "distilbert" / "distilbert_test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "models" / "distilbert")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = _load_jsonl(args.train_file)
    val_rows = _load_jsonl(args.val_file)
    test_rows = _load_jsonl(args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    if hasattr(config, "seq_classif_dropout"):
        config.seq_classif_dropout = args.dropout

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        progress = tqdm(range(0, len(train_rows), args.batch_size), desc=f"distilbert epoch {epoch + 1}", leave=False)
        for start in progress:
            batch_rows = train_rows[start : start + args.batch_size]
            texts = [row["text"] for row in batch_rows]
            labels = torch.tensor([row["label"] for row in batch_rows], dtype=torch.long, device=accelerator.device)
            enc = tokenizer(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(accelerator.device) for k, v in enc.items()}
            loss = model(**enc, labels=labels).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            batch_count += 1
            progress.set_postfix(loss=loss.item())

        acc, prec, rec, f1 = _eval_split(model, tokenizer, val_rows, args.batch_size, args.max_length, accelerator.device)
        print("epoch", epoch + 1, "loss", train_loss / max(1, batch_count))
        print("val acc", acc, "precision", prec, "recall", rec, "f1", f1)

    t_acc, t_prec, t_rec, t_f1 = _eval_split(
        model, tokenizer, test_rows, args.batch_size, args.max_length, accelerator.device
    )
    print("test acc", t_acc, "precision", t_prec, "recall", t_rec, "f1", t_f1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
