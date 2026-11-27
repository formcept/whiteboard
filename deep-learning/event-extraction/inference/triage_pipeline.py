"""Single-document end-to-end event extraction inference CLI."""

import json
import os
import sys
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from outlines import from_transformers, json_schema
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from inference.pipeline_core import run_document_pipeline

MODEL_DIR = BASE / "models" / "distilbert"
QWEN_DIR = BASE / "models" / "qwen_lora" / "merged"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python inference/triage_pipeline.py "text_to_extract"')

    text = " ".join(sys.argv[1:])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    triage_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    triage_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    triage_model.eval()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_DIR,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    generator = from_transformers(qwen_model, qwen_tokenizer)
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    outlines_schema = json_schema(schema)

    extracted_events, _ = run_document_pipeline(
        text=text,
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        triage_model=triage_model,
        triage_tokenizer=triage_tokenizer,
        triage_device=device,
        generator=generator,
        outlines_schema=outlines_schema,
    )

    if not extracted_events:
        print("null")
    elif len(extracted_events) == 1:
        print(json.dumps(extracted_events[0], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(extracted_events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
