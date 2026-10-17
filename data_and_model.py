from typing import Dict,Any,Tuple
import torch,os
from transformers  import AutoTokenizer,AutoConfig,AutoModelForCausalLM
from datasets import load_dataset,DatasetDict
from peft import LoraConfig,TaskType,get_peft_model
import json

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT=(
    "You are a logistics document extraction engine. Extract all shipment, carrier, "
    "container, and invoice fields from the provided OCR document into the exact target "
    "JSON schema. If a field is not present in the document text, output null."
)

def load_qwen_lora_for_finetuning(model_id:str=MODEL_ID,
                                  
                                  lora_r:int=32,
                                  lora_alpha:int=64,
                                  lora_dropout: float = 0.05,
                                  enable_yarn_above_32k: bool = False,
                                  ) -> Tuple[torch.nn.Module,AutoTokenizer]:
    """lora_dropout: float = 0.05,
        enable_yarn_above_32k: bool = False,"""

    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

    local_rank=int(os.environ.get("LOCAL_RANK",0))

    #Load tokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=True,padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token= "<|endoftext|>"
        tokenizer.pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")

    #Load Model config( Native 32,768 context)
    config=AutoConfig.from_pretrained(model_id)
    config.use_cache=False #for gradient checkpointing

    #load base model in bf16
    model=AutoModelForCausalLM.from_pretrained(model_id,
                                               config=config,
                                               dtype=torch.bfloat16,
                                            #    attn_implementation='flash_attention_2',
                                                attn_implementation="sdpa",
                                               device_map={"":f"cuda:{local_rank}"},
                                               )

    #enable non renentrant pytorch grad checkpointing
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    #attaching pure LoRA adpater across all 7 linear layers
    lora_config=LoraConfig(r=lora_r,
                           lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout,
                           bias="none",
                           task_type=TaskType.CAUSAL_LM,
                           target_modules=[
                                        "q_proj",
                                        "k_proj",
                                        "v_proj",
                                        "o_proj",
                                        "gate_proj",
                                        "up_proj",
                                        "down_proj",
                                    ],

                           )
    model =get_peft_model(model,lora_config)
    model.print_trainable_parameters()
    return model,tokenizer

def format_to_prompt_completion(example:Dict[str,Any],tokenizer:AutoTokenizer)-> Dict[str,str]:
    """
    Converts each carrier OCR document + TMS ground-truth JSON into TRL's
    prompt-completion format so loss is computed ONLY on the assistant's JSON output.
    Expected input JSONL keys: 'carrier_id', 'ocr_text', 'ground_truth_json'
    """
    prompt_messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {
            "role":"user",
            "content":(f"Carrier Code: {example['carrier_id']}\n"
                f"Document OCR Text:\n{example['ocr_text']}")
        }
    ]
    # Renders "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
    prompt_str=tokenizer.apply_chat_template(prompt_messages,
                                             tokenize=False,
                                             add_generation_prompt=True)

    target_json_str=json.dumps(example["ground_truth_json"], ensure_ascii=False, sort_keys=True)
    completion_str = f"{target_json_str}{tokenizer.eos_token}"
    return {"prompt": prompt_str, "completion": completion_str}

def load_carrier_datasets(
    train_path: str, val_path: str, tokenizer: AutoTokenizer
) -> DatasetDict:
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )
    formatted_datasets = raw_datasets.map(
        lambda ex: format_to_prompt_completion(ex, tokenizer),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=8,
        desc="Formatting 36k carrier docs into Qwen2.5 ChatML prompt/completion",
    )
    return formatted_datasets

