# evaluate.py
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_and_model import MODEL_ID, SYSTEM_PROMPT
import os

TEST_JSONL ="./data/fatura_test_1250.jsonl"
ADAPTER_DIR="./checkpoints/qwen25-7b-carrier-lora/checkpoint-200"
# MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS=768


def normalize_val(val: Any) -> str:
    """Normalizes strings, numbers, and nulls for fair field-level comparison."""
    if val is None or val == "":
        return "NULL"
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}"
    if isinstance(val, list):
        return json.dumps([normalize_val(x) for x in val], sort_keys=True)
    if isinstance(val, dict):
        return json.dumps(
            {k: normalize_val(v) for k, v in val.items()}, sort_keys=True
        )
    return " ".join(str(val).strip().upper().split())


def score_document_fields(
    pred_json: Dict[str, Any], gt_json: Dict[str, Any]
) -> Tuple[int, int]:
    """Returns (matched_fields, total_ground_truth_fields) for a single document."""
    matched = 0
    total = len(gt_json)
    for key, gt_val in gt_json.items():
        pred_val = pred_json.get(key, None)
        if normalize_val(pred_val) == normalize_val(gt_val):
            matched += 1
    return matched, total


@torch.inference_mode()
def evaluate_model(
    use_lora_adapter: bool = True,
    batch_size: int = 4,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR if use_lora_adapter else MODEL_ID,
        use_fast=True,
        padding_side="left",  # Left-padding is mandatory for batched generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        attn_implementation='sdpa',
        device_map="cuda:0",
    )

    if use_lora_adapter:
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model = model.merge_and_unload()  # Merge LoRA weights into bf16 base for zero-overhead eval
    else:
        model = base_model

    model.eval()

    # Load 1,800 stratified test documents (60 carriers x 30 docs)
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]

    carrier_stats = defaultdict(lambda: {"matched_fields": 0, "total_fields": 0, "valid_json": 0, "docs": 0})
    total_matched_fields = 0
    total_gt_fields = 0
    valid_json_count = 0

    for i in tqdm(range(0, len(test_records), batch_size), desc="Evaluating Test Split"):
        batch = test_records[i : i + batch_size]
        prompts = []
        for rec in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Carrier Code: {rec['carrier_id']}\nDocument OCR Text:\n{rec['ocr_text']}",
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384 - MAX_NEW_TOKENS,
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy deterministic decoding for structured extraction
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Slice off the input prompt tokens
        input_len = inputs["input_ids"].shape[1]
        decoded_outputs = tokenizer.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        for rec, raw_output in zip(batch, decoded_outputs):
            carrier = rec["carrier_id"]
            gt_json = rec["ground_truth_json"]
            carrier_stats[carrier]["docs"] += 1

            try:
                pred_json = json.loads(raw_output.strip())
                if isinstance(pred_json, dict):
                    valid_json_count += 1
                    carrier_stats[carrier]["valid_json"] += 1
                else:
                    pred_json = {}
            except json.JSONDecodeError:
                pred_json = {}

            matched, total = score_document_fields(pred_json, gt_json)
            total_matched_fields += matched
            total_gt_fields += total
            carrier_stats[carrier]["matched_fields"] += matched
            carrier_stats[carrier]["total_fields"] += total

    micro_accuracy = (total_matched_fields / total_gt_fields) * 100.0
    macro_accuracy = (
        sum(
            (s["matched_fields"] / s["total_fields"]) * 100.0
            for s in carrier_stats.values()
        )
        / len(carrier_stats)
    )
    json_validity_rate = (valid_json_count / len(test_records)) * 100.0

    report = {
        "model_variant": "Fine-Tuned LoRA (r=32, alpha=64)" if use_lora_adapter else "Base Qwen2.5-7B-Instruct",
        "total_test_documents": len(test_records),
        "total_carriers": len(carrier_stats),
        "json_validity_rate_pct": round(json_validity_rate, 2),
        "micro_field_accuracy_pct": round(micro_accuracy, 2),
        "macro_carrier_accuracy_pct": round(macro_accuracy, 2),
    }
    print(json.dumps(report, indent=2))
    report["adapter_dir"] = ADAPTER_DIR
    report["test_file"] = TEST_JSONL
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"eval_{os.path.basename(TEST_JSONL).removesuffix('.jsonl')}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_path}")

    return report


if __name__ == "__main__":
    print("--- Running Fine-Tuned LoRA Evaluation ---")
    evaluate_model(use_lora_adapter=True, batch_size=16) # batch size adjusted to 16 