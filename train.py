import json
import os
import torch
from trl import SFTConfig,SFTTrainer
from data_and_model import load_qwen_lora_for_finetuning,load_carrier_datasets

# OUTPUT_DIR = "./checkpoints/qwen25-7b-carrier-lora"
# TRAIN_JSONL = "./data/fatura_train_7500.jsonl"  # built by prepare_fatura.py
# VAL_JSONL = "./data/fatura_val_1250.jsonl"

OUTPUT_DIR  = "./checkpoints/qwen25-7b-fatura-s2-lora"    # new folder, so it doesn't resume Strategy 1
TRAIN_JSONL = "./data/fatura_s2_train_8000.jsonl"
VAL_JSONL   = "./data/fatura_s2_val_1000.jsonl"
# MAX_SEQ_LENGTH = 16384  # Handles >4k multi-page carrier manifests & invoices
MAX_SEQ_LENGTH=2048
def save_infra_metadata(output_dir:str,args:SFTConfig):
    """Logs hardware & training stack lineage alongside the checkpoints."""
    os.makedirs(output_dir,exist_ok=True)
    gpu_name =(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    metadata={
        "compute_environment": {
            "azure_vm_sku": os.environ.get("AZUREML_VM_SIZE", "Standard_NC24ads_A100_v4"),
            "gpu_device": gpu_name,
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        "hyperparameters": {
            "max_seq_length": args.max_length,
            "learning_rate": args.learning_rate,
            "effective_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, torch.cuda.device_count())
            ),
            "num_train_epochs": args.num_train_epochs,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_steps,
            },
    }
    with open(os.path.join(output_dir, "training_infra_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    #load model
    model,tokenizer=load_qwen_lora_for_finetuning(lora_r=32,
                                                  lora_alpha=64,
                                                  lora_dropout=0.05)
    datasets=load_carrier_datasets(TRAIN_JSONL,VAL_JSONL,tokenizer)

    training_args=SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        packing=False,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),

        #batching and optimization
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,

        #pytoch kernal and memory optimization
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=False, # have enough memory left on a100 80gigs
        # gradient_checkpointing_kwargs={"use_reentrant":False},
        dataloader_num_workers=2, #adjusted for colab
        dataloader_pin_memory=True,

        #logging eval and checkpointing
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=50, #adjusted , 7500 docs so 234 steps per epoch
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["mlflow","wandb"],
        run_name="qwen25-7b-fatura-lora-s2",
        seed=42,

    )

    save_infra_metadata(OUTPUT_DIR,training_args)

    #init trainer and execute training 
    trainer=SFTTrainer(model=model,
                       processing_class=tokenizer,
                       train_dataset=datasets["train"],
                       eval_dataset=datasets["validation"],
                       args=training_args)

    from transformers.trainer_utils import get_last_checkpoint   # add to imports

    last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    trainer.train(resume_from_checkpoint=last_ckpt)


    final_adapter_dir=os.path.join(OUTPUT_DIR,"final_adapter")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Final LoRA adapter saved to: {final_adapter_dir}")

if __name__=="__main__":
    main()


