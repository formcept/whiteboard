"""Shared extraction evaluation, schema validation, and per-class metric helpers."""

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from jsonschema import Draft7Validator

TOP_LEVEL_KEYS = {"event_type", "source_timestamp", "text_evidence"}
EVENT_CLASSES = [
    "FacilityHalt",
    "ShipmentDelay",
    "SupplierInsolvency",
    "ForceMajeure",
    "TariffChange",
]


def load_schema_validator(schema_path: Path) -> Draft7Validator:
    """Load Draft-07 JSON schema validator from file."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def parse_to_date(val):
    """Parse ISO-style date or YYYY-MM partial date strings into date objects."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "null":
        return None
    if "t" in val_str.lower():
        val_str = re.split(r"[tT]", val_str)[0]

    try:
        return datetime.strptime(val_str, "%Y-%m-%d").date()
    except ValueError:
        pass

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", val_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    match_ym = re.search(r"(\d{4})-(\d{2})", val_str)
    if match_ym:
        try:
            return date(int(match_ym.group(1)), int(match_ym.group(2)), 1)
        except ValueError:
            pass

    return val_str


def compute_token_f1(pred_val, gold_val) -> float:
    """Compute token-level bag-of-words F1 between predicted and gold field values."""
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0

    if isinstance(pred_val, list):
        pred_val = " ".join(str(item) for item in pred_val)
    if isinstance(gold_val, list):
        gold_val = " ".join(str(item) for item in gold_val)

    pred_str = str(pred_val).strip().lower()
    gold_str = str(gold_val).strip().lower()
    if pred_str == gold_str:
        return 1.0
    if not pred_str or not gold_str:
        return 0.0

    pred_tokens = re.findall(r"\w+", pred_str)
    gold_tokens = re.findall(r"\w+", gold_str)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    intersection = sum((pred_counter & gold_counter).values())
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_tokens)
    recall = intersection / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred_val, gold_val) -> float:
    """Compute case-insensitive exact match score."""
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0
    return 1.0 if str(pred_val).strip().lower() == str(gold_val).strip().lower() else 0.0


def normalize_prediction(pred):
    """Normalize prediction payload into a schema-comparable dictionary or None."""
    if pred is None:
        return None
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    if isinstance(pred, dict):
        cleaned = dict(pred)
        cleaned.pop("event_id", None)
        cleaned.pop("triage_label", None)
        cleaned.pop("chunk_text", None)
        return cleaned
    return None


def get_flat_dict(obj) -> dict:
    """Flatten top-level fields and nested arguments dictionary."""
    if not isinstance(obj, dict):
        return {}
    flat = {field: obj.get(field) for field in ("event_type", "source_timestamp", "text_evidence")}
    args = obj.get("arguments", {})
    if isinstance(args, dict):
        flat.update(args)
    return flat


def _score_field(key: str, p_val, g_val) -> float:
    if key == "event_type":
        return compute_exact_match(p_val, g_val)
    if key == "source_timestamp":
        p_date = parse_to_date(p_val)
        g_date = parse_to_date(g_val)
        if p_date is None and g_date is None:
            return 1.0
        if p_date is None or g_date is None:
            return 0.0
        return 1.0 if p_date == g_date else 0.0
    return compute_token_f1(p_val, g_val)


def evaluate_fields_subset(pred_flat: dict, gold_flat: dict, scores: dict, keys: set) -> tuple:
    """Compute precision, recall, and F1 over a subset of flattened schema keys."""
    pred_keys = [k for k in pred_flat if k in keys]
    precision = sum(scores[k] for k in pred_keys) / len(pred_keys) if pred_keys else 1.0

    gold_keys = [k for k in gold_flat if k in keys]
    recall = sum(scores[k] for k in gold_keys) / len(gold_keys) if gold_keys else 1.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_sample(pred_dict, gold_dict) -> tuple:
    """Return (precision, recall, f1, top_level_f1, other_fields_f1) for one sample."""
    pred_is_noevent = isinstance(pred_dict, dict) and pred_dict.get("event_type") == "NoEvent"
    if not gold_dict:
        if not pred_dict or pred_is_noevent:
            return 1.0, 1.0, 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if not pred_dict or pred_is_noevent:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pred_flat = get_flat_dict(pred_dict)
    gold_flat = get_flat_dict(gold_dict)

    all_keys = set(pred_flat.keys()) | set(gold_flat.keys())
    scores = {key: _score_field(key, pred_flat.get(key), gold_flat.get(key)) for key in all_keys}

    precision, recall, f1 = evaluate_fields_subset(pred_flat, gold_flat, scores, all_keys)
    _, _, top_f1 = evaluate_fields_subset(pred_flat, gold_flat, scores, TOP_LEVEL_KEYS)
    _, _, other_f1 = evaluate_fields_subset(pred_flat, gold_flat, scores, all_keys - TOP_LEVEL_KEYS)
    return precision, recall, f1, top_f1, other_f1


def evaluate_prediction_file(test_rows: list, pred_rows: list, validator: Draft7Validator) -> dict:
    """Compute aggregate, per-class, and confusion-matrix metrics for a prediction list."""
    sample_rows = []
    per_class = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))

    for idx, gold_row in enumerate(test_rows):
        gold_event = gold_row.get("event_data")
        gold_cls = gold_event.get("event_type") if isinstance(gold_event, dict) else "NoEvent"

        raw_pred = pred_rows[idx] if idx < len(pred_rows) else None
        norm_pred = normalize_prediction(raw_pred)
        pred_cls = norm_pred.get("event_type") if isinstance(norm_pred, dict) else "Invalid/Null"

        valid = False
        if norm_pred is not None:
            valid = not any(validator.iter_errors(norm_pred))

        p, r, f1, top_f1, other_f1 = evaluate_sample(norm_pred, gold_event)
        row_metrics = {
            "schema_valid": int(valid),
            "precision": p,
            "recall": r,
            "f1": f1,
            "top_level_f1": top_f1,
            "other_fields_f1": other_f1,
            "event_type_acc": float(pred_cls == gold_cls),
        }
        sample_rows.append(row_metrics)
        per_class[gold_cls].append(row_metrics)
        confusion[gold_cls][pred_cls] += 1

    def _mean_block(items):
        if not items:
            return {}
        keys = ["precision", "recall", "f1", "top_level_f1", "other_fields_f1", "schema_valid", "event_type_acc"]
        out = {("schema_validity" if k == "schema_valid" else k): sum(x[k] for x in items) / len(items) for k in keys}
        out["count"] = len(items)
        return out

    pos_indices = [i for i, row in enumerate(test_rows) if row.get("event_data") is not None]
    pos_rows = [sample_rows[i] for i in pos_indices]

    return {
        "overall": _mean_block(sample_rows),
        "positive_only": _mean_block(pos_rows),
        "per_class": {cls: _mean_block(items) for cls, items in sorted(per_class.items())},
        "confusion_matrix": {g: dict(preds) for g, preds in sorted(confusion.items())},
    }
