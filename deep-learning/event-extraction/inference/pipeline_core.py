"""Shared inference utilities for two-stage triage and schema-constrained event extraction."""

import json
import re
import string
import time
import uuid

import torch

SYSTEM_PROMPT = (
    "You are an expert supply chain disruption event extractor. "
    "Extract a single event matching the provided JSON schema.\n\n"
    "Guidelines:\n"
    "- Extract only information that is explicitly stated in the source text. NEVER use external world knowledge to fill in blank fields. If a value is not explicitly written in the text, you MUST return null.\n"
    "- Do not copy any names or placeholders from the examples in the prompt unless they appear word-for-word in the source text.\n"
    "- Do not infer, estimate, or hallucinate any facts.\n"
    "- If a value is not explicitly mentioned, return null for that field.\n"
    "- IF EXPLICIT TIME OR MONTH OR DATE OR EXACT NUMBER FOR YEAR IS NOT PRESENT set the source_timestamp to null.\n"
    "- Extract the date of the event and format it as an ISO 8601 string.\n"
    "- Preserve the original meaning of the text.\n"
    "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
    "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as bankruptcy, Chapter 11 reorganization, insolvency, liquidation, defaulting on debts, receivership, or experiencing severe liquidity crises leading to restructuring.\n\n"
    "Example 1:\n"
    "Text: In other news, global logistics giant [LOGISTICS_COMPANY] announced a 3-day halt of operations at [PORT_LOCATION] due to a severe labor strike starting on [SOURCE_TIMESTAMP].\n"
    'Output: {"event_type": "FacilityHalt", "source_timestamp": "[SOURCE_TIMESTAMP]", "text_evidence": "halt of operations at [PORT_LOCATION] due to a severe labor strike starting on 2023-11-05", "arguments": {"operator": "[LOGISTICS_COMPANY]", "facility_location": "[PORT_LOCATION]", "disruption_type": "Labor_Dispute", "start_date": "2023-11-05T00:00:00Z", "expected_restart_date": null}}\n\n'
    "Example 2:\n"
    "Text: The CEO mentioned they are launching a new product line next week.\n"
    'Output: {"event_type": "NoEvent"}'
)

DISAMBIGUATION_ADDENDUM = (
    "\n- Classify an event as ShipmentDelay when the primary reported impact is delayed transit, shipping congestion, vessel/carrier delays, or rerouted cargo; classify as FacilityHalt ONLY when a specific manufacturing plant, warehouse, or port terminal explicitly suspends or halts operations."
)

ENUM_FIELDS = {"tariff_action", "disruption_type", "legal_action"}
MONTH_NAMES = {
    1: ("january", "jan"),
    2: ("february", "feb"),
    3: ("march", "mar"),
    4: ("april", "apr"),
    5: ("may",),
    6: ("june", "jun"),
    7: ("july", "jul"),
    8: ("august", "aug"),
    9: ("september", "sep", "sept"),
    10: ("october", "oct"),
    11: ("november", "nov"),
    12: ("december", "dec"),
}


def is_numeric_or_percentage(value) -> bool:
    """Return True if value is a numeric literal or formatted numeric/percentage string."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace("$", "").replace(",", "").replace(".", "").strip()
        return cleaned.isdigit()
    return False


_PUNCT_TRANS = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _prepare_chunk_index(chunk: str) -> tuple:
    """Single-pass O(N) chunk normalization returning (lower_text, token_frozenset) for O(1) lookup."""
    chunk_lower = chunk.lower().translate(_PUNCT_TRANS)
    return chunk_lower, frozenset(chunk_lower.split())


def _is_date_grounded(val_str: str, chunk_lower: str, chunk_tokens: frozenset) -> bool:
    """Verify ISO-8601 date strings against year and month/day expressions in chunk."""
    match = re.match(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?(?:[tT].*)?$", val_str.strip())
    if not match:
        return False
    year, month, day = match.group(1), match.group(2), match.group(3)
    if year not in chunk_tokens:
        return False
    if month:
        m_int = int(month)
        month_ok = month in chunk_tokens or str(m_int) in chunk_tokens or any(
            alias in chunk_tokens for alias in MONTH_NAMES.get(m_int, ())
        )
        if not month_ok and not (day == "01" and m_int == 1):
            return False
    return True


def _is_value_grounded(value, chunk_lower: str, chunk_tokens: frozenset, field_aware: bool = True) -> bool:
    if value is None or is_numeric_or_percentage(value):
        return True
    if field_aware and isinstance(value, list):
        if not value:
            return True
        return all(_is_value_grounded(item, chunk_lower, chunk_tokens, field_aware=True) for item in value)

    val_str = str(value).strip().lower()
    if not val_str or val_str == "null":
        return True
    if val_str.startswith("[") and val_str.endswith("]"):
        return False
    if field_aware and _is_date_grounded(val_str, chunk_lower, chunk_tokens):
        return True

    tokens = re.findall(r"[a-z0-9]+", val_str) if field_aware else [w.strip(string.punctuation) for w in val_str.split()]
    val_words = [w for w in tokens if len(w) > 2]
    if not val_words:
        return True
    if field_aware:
        return any(w in chunk_tokens for w in val_words)
    return any(w in chunk_lower for w in val_words)


def get_hallucinated_fields(output, chunk: str, field_aware: bool = True, chunk_index: tuple = None) -> list:
    """Return field names whose extracted values lack grounding in the input chunk."""
    if not isinstance(output, dict):
        return []
    chunk_lower, chunk_tokens = chunk_index or _prepare_chunk_index(chunk)
    hallucinated = []

    args = output.get("arguments", {})
    if isinstance(args, dict):
        for k, v in args.items():
            if k in ENUM_FIELDS:
                continue
            if v is not None and not _is_value_grounded(v, chunk_lower, chunk_tokens, field_aware=field_aware):
                hallucinated.append(k)

    val = output.get("text_evidence")
    if val is not None and not _is_value_grounded(val, chunk_lower, chunk_tokens, field_aware=field_aware):
        hallucinated.append("text_evidence")
    return hallucinated


def filter_hallucinations(output, chunk: str, field_aware: bool = True, chunk_index: tuple = None):
    """Set ungrounded free-text extracted fields to null as a deterministic fallback."""
    if not isinstance(output, dict):
        return output
    chunk_lower, chunk_tokens = chunk_index or _prepare_chunk_index(chunk)

    args = output.get("arguments", {})
    if isinstance(args, dict):
        for k, v in list(args.items()):
            if k in ENUM_FIELDS:
                continue
            if v is not None and not _is_value_grounded(v, chunk_lower, chunk_tokens, field_aware=field_aware):
                args[k] = None

    val = output.get("text_evidence")
    if val is not None and not _is_value_grounded(val, chunk_lower, chunk_tokens, field_aware=field_aware):
        output["text_evidence"] = None
    return output


def _merge_corrected_fields(original: dict, corrected: dict, flagged_fields: list) -> dict:
    """Apply Pass-2 corrections only to flagged fields so grounded Pass-1 fields are never dropped."""
    if not isinstance(corrected, dict):
        return original
    merged = json.loads(json.dumps(original))
    flagged = set(flagged_fields)
    if "text_evidence" in flagged and "text_evidence" in corrected:
        merged["text_evidence"] = corrected["text_evidence"]
    orig_args = merged.get("arguments")
    corr_args = corrected.get("arguments")
    if isinstance(orig_args, dict) and isinstance(corr_args, dict):
        for field in flagged:
            if field in orig_args and field in corr_args:
                orig_args[field] = corr_args[field]
    return merged


def chunk_words(text: str, words_per_chunk: int = 260) -> list:
    """Split text into fixed word-count chunks."""
    words = text.split()
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def parse_json_completion(completion: str):
    """Extract a JSON object from an unconstrained model completion."""
    text = completion.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"```json\s*(\{.*?\})\s*```", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _format_prompt(tokenizer, messages: list, fallback_chunk: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "You are an expert supply chain disruption event extractor. "
        "Extract a single event matching the provided JSON schema.\n\n"
        f"Text:\n{fallback_chunk.strip()}\n"
    )


def _generate_event(
    qwen_model,
    qwen_tokenizer,
    messages: list,
    chunk: str,
    generator=None,
    outlines_schema=None,
    max_new_tokens: int = 512,
):
    prompt = _format_prompt(qwen_tokenizer, messages, chunk)
    if generator is not None and outlines_schema is not None:
        raw_out = generator(
            prompt,
            outlines_schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )
        tok_count = len(qwen_tokenizer(raw_out, add_special_tokens=False).input_ids) if isinstance(raw_out, str) else 0
        parsed = json.loads(raw_out) if isinstance(raw_out, str) else raw_out
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        return parsed, tok_count

    inputs = qwen_tokenizer(prompt, return_tensors="pt")
    device = next(qwen_model.parameters()).device
    with torch.no_grad():
        generated = qwen_model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )
    gen_ids = generated[0][inputs["input_ids"].shape[1] :]
    completion = qwen_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return parse_json_completion(completion), int(len(gen_ids))


def run_document_pipeline(
    text: str,
    qwen_model,
    qwen_tokenizer,
    triage_model=None,
    triage_tokenizer=None,
    triage_device=None,
    triage_threshold: float = 0.5,
    generator=None,
    outlines_schema=None,
    use_self_correction: bool = True,
    use_fallback_filter: bool = True,
    field_aware_grounding: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 512,
) -> tuple:
    """Run triage + extraction + optional grounding correction on one document."""
    chunks = chunk_words(text, 260)
    stats = {
        "chunks": len(chunks),
        "triage_positives": 0,
        "triage_s": 0.0,
        "pass1_s": 0.0,
        "pass2_s": 0.0,
        "correction_triggered": 0,
        "tokens_generated": 0,
    }
    if not chunks:
        return [], stats

    extracted_events = []
    for chunk in chunks:
        label = 1
        if triage_model is not None and triage_tokenizer is not None:
            t0 = time.perf_counter()
            inputs = triage_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            inputs.pop("token_type_ids", None)
            if triage_device is not None:
                inputs = {k: v.to(triage_device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(triage_model(**inputs).logits, dim=-1)
                pos_prob = float(probs[0, 1].item())
                label = 1 if pos_prob >= triage_threshold else 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["triage_s"] += time.perf_counter() - t0

        if label != 1:
            extracted_events.append({"event_type": "NoEvent", "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}"})
            continue

        stats["triage_positives"] += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ]

        t1 = time.perf_counter()
        output, toks1 = _generate_event(
            qwen_model, qwen_tokenizer, messages, chunk, generator, outlines_schema, max_new_tokens
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats["pass1_s"] += time.perf_counter() - t1
        stats["tokens_generated"] += toks1

        chunk_index = _prepare_chunk_index(chunk)
        if use_self_correction and isinstance(output, dict):
            hallucinated_fields = get_hallucinated_fields(
                output, chunk, field_aware=field_aware_grounding, chunk_index=chunk_index
            )
            if hallucinated_fields:
                stats["correction_triggered"] += 1
                correction_instruction = (
                    f"In your previous response, ONLY the following fields lack grounding in the source text: "
                    f"{', '.join(hallucinated_fields)}. "
                    f"Regenerate the JSON keeping all other fields unchanged, and set or correct ONLY these flagged fields "
                    f"using facts explicitly present in the source text (or null if absent)."
                )
                correction_messages = messages + [
                    {"role": "assistant", "content": json.dumps(output)},
                    {"role": "user", "content": correction_instruction},
                ]
                t2 = time.perf_counter()
                corrected, toks2 = _generate_event(
                    qwen_model, qwen_tokenizer, correction_messages, chunk, generator, outlines_schema, max_new_tokens
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stats["pass2_s"] += time.perf_counter() - t2
                stats["tokens_generated"] += toks2
                if isinstance(corrected, dict):
                    output = (
                        _merge_corrected_fields(output, corrected, hallucinated_fields)
                        if field_aware_grounding
                        else corrected
                    )

        if isinstance(output, dict):
            if use_fallback_filter:
                output = filter_hallucinations(
                    output, chunk, field_aware=field_aware_grounding, chunk_index=chunk_index
                )
            output["event_id"] = f"EVT-{uuid.uuid4().hex[:8].upper()}"
            extracted_events.append(output)

    return extracted_events, stats
