# inference.py
import json
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_and_model import MODEL_ID, SYSTEM_PROMPT

ADAPTER_DIR = "./checkpoints/qwen25-7b-carrier-lora/final_adapter"
MERGED_MODEL_DIR = "./checkpoints/qwen25-7b-carrier-merged-bf16"


class CarrierDocumentExtractor:
    """
    Production PyTorch inference wrapper for long-context (>4k token) carrier extraction.
    """

    def __init__(self, base_model_id: str = MODEL_ID, adapter_dir: str = ADAPTER_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            device_map="cuda:0",
        )

        # Load LoRA adapter and permanently fold (B @ A) * (alpha / r) into base bf16 weights
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model = peft_model.merge_and_unload()
        self.model.eval()

    @torch.inference_mode()
    def extract(
        self, carrier_code: str, ocr_text: str, max_new_tokens: int = 2048
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Carrier Code: {carrier_code}\nDocument OCR Text:\n{ocr_text}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=32768 - max_new_tokens,  # Full 32k native context support
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return json.loads(response_text)

    def save_merged_for_vllm(self, save_dir: str = MERGED_MODEL_DIR):
        """Exports a standalone merged bf16 model ready for vLLM serving."""
        self.model.save_pretrained(save_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Merged bf16 model exported to {save_dir}")


if __name__ == "__main__":
    extractor = CarrierDocumentExtractor()

    # sample_ocr_manifest = (
    #     "MASTER BILL OF LADING - MAERSK LINE (SCAC: MAEU)\n"
    #     "B/L NO: MAEU984120491 | VESSEL: MAERSK MC-KINNEY MOLLER | VOYAGE: 412E\n"
    #     "PORT OF LOADING: SHANGHAI, CN | PORT OF DISCHARGE: ROTTERDAM, NL\n"
    #     "CONTAINER: MSKU4918201 | SEAL: ML99102 | TYPE: 40HC | GROSS WEIGHT: 24,150.00 KG\n"
    #     "FREIGHT CHARGES: OCEAN FREIGHT USD 3,450.00 | BUNKER SURCHARGE USD 420.00 | TOTAL: USD 3,870.00"
    # )
    rec = json.loads(open("data/fatura_test_1250.jsonl").readline())
    print(json.dumps(extractor.extract(rec["carrier_id"], rec["ocr_text"]), indent=2))


    # extracted_payload = extractor.extract(
    #     carrier_code="MAEU",
    #     ocr_text=sample_ocr_manifest,
    # )
    # print(json.dumps(extracted_payload, indent=2))