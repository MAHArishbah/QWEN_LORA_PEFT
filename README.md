# Qwen2.5-7B LoRA: invoice field extraction on FATURA

Fine-tunes **Qwen2.5-7B-Instruct** with **LoRA** to turn invoice text with bounding boxes into a fixed JSON schema of 25 fields. Trained on the [FATURA](https://zenodo.org/records/10371464) dataset (10,000 synthetic invoices from 50 layout templates), on one Colab A100, with tracking in Weights & Biases and a self-hosted MLflow server.

## Results

| Test set | What it measures | Adapter | JSON validity | Field accuracy (micro) | Field accuracy (macro, per template) |
|---|---|---|---|---|---|
| **Strategy 1** (1,250 docs) | New invoices from the 50 layouts seen in training | `qwen25-7b-carrier-lora/checkpoint-200` | 100.0% | 99.99% | 99.99% |
| **Strategy 2** (1,000 docs) | Invoices from 10 layouts **never seen** in training | `qwen25-7b-fatura-s2-lora/final_adapter` | 100.0% | 99.99% | 99.99% |

Field accuracy is an exact match after normalizing case and whitespace, over all 25 keys per document, **including correctly predicted `null`s**. See [Limitations](#limitations): these scores are real, but they measure less than they appear to.

## How it works

```
FATURA annotations ──prepare_fatura.py──▶ JSONL {carrier_id, ocr_text, ground_truth_json}
                                                 │
                       data_and_model.py ◀───────┘  ChatML prompt + JSON completion
                                                 │
                              train.py ──────────┤  TRL SFTTrainer, LoRA r=32, loss on JSON only
                                                 │
                     evaluate.py / inference.py  ◀  base model + adapter, greedy decoding
```

**Input (`ocr_text`).** Every annotated text field becomes one line, `[x0,y0,x1,y1] text`, in reading order. Coordinates are scaled to 0–1000 with the origin at the top-left, the LayoutLM convention; FATURA stores PDF points with a bottom-left origin and corners in any order, so they're converted. Line breaks inside a field are shown as ` | `.

```
[551,25,745,46] TAX INVOICE
[551,63,721,77] Date: 20-Mar-2008
[558,170,841,269] Bill to:Denise Perez | 16424 Timothy Mission | Markville, AK 58294 US | ...
[652,878,847,893] TOTAL : 734.33 EUR
```

**Target (`ground_truth_json`).** A fixed set of 25 lowercase keys, `null` when the field is absent. Values are the field text exactly as printed, label included (`"TOTAL : 734.33 EUR"`, not `734.33`). `gst` is a list, because one template prints five GST lines.

`amount_due, bill_to, buyer, conditions, date, discount, due_date, gst, gstin, gstin_buyer, gstin_seller, note, number, payment_details, po_number, seller_address, seller_email, seller_name, seller_site, send_to, sub_total, tax, title, total, total_words`

**`carrier_id`.** FATURA has no carriers, so this is the template ID (`TEMPLATE_01` … `TEMPLATE_50`). It plays the same role: it identifies the document's source format.

**Prompt format.** `apply_chat_template` renders the system and user messages in Qwen's ChatML format, ending with `<|im_start|>assistant\n`. The completion is the JSON (keys sorted, compact, `ensure_ascii=False`) followed by `<|im_end|>`, so the model learns to stop. `completion_only_loss=True` means only the JSON tokens are scored. `evaluate.py` and `inference.py` build the prompt exactly the same way.

## Repository

| File | Purpose |
|---|---|
| [prepare_fatura.py](prepare_fatura.py) | Converts FATURA annotations to train/val/test JSONL (`--strategy 1` or `2`) |
| [data_and_model.py](data_and_model.py) | Loads Qwen + LoRA; formats JSONL into prompt/completion pairs |
| [train.py](train.py) | `SFTTrainer` setup, checkpoint resume, saves the final adapter |
| [evaluate.py](evaluate.py) | Generates JSON for a test file, scores field accuracy, writes `results/eval_*.json` |
| [inference.py](inference.py) | `CarrierDocumentExtractor`: single-document extraction; can export a merged model for vLLM |
| `data/` | Generated JSONL (not committed; rebuild with `prepare_fatura.py`) |

## Data splits

Splits are decided in `prepare_fatura.py` when the JSONL files are created; nothing downstream re-splits.

| Strategy | Train | Val | Test | Source |
|---|---|---|---|---|
| **1**: all layouts seen | 7,500 | 1,250 | 1,250 | FATURA's official `strat1_*.csv`: 150/25/25 instances per template |
| **2**: 10 layouts held out | 8,000 (40 templates) | 1,000 | 1,000 | FATURA's `Strat2_Split.txt` for the templates, then instances 0–99 → val and 100–199 → test |

FATURA's own Strategy 2 file uses the same documents for dev and test. Splitting the held-out templates by instance keeps checkpoint selection (val) separate from the reported score (test). Held-out templates: 2, 4, 7, 10, 12, 25, 32, 39, 49, 50.

Documents are short: 472–1,180 tokens for prompt plus answer (median ~735). The longest answer is 489 tokens.

## Setup

### Local (WSL)

Keep the virtual environment on the Linux filesystem. Inside `/mnt/<drive>` (the Windows-mounted disk), pip and Pylance are very slow.

```bash
python3 -m venv ~/.venvs/fine-tune
~/.venvs/fine-tune/bin/pip install -r requirements.txt
```

### Build the dataset

```bash
mkdir -p ~/datasets/fatura && cd ~/datasets/fatura
curl -L -o FATURA2.zip "https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"   # 690 MB
python3 -m zipfile -e FATURA2.zip .
cd -   # back to the repo

python prepare_fatura.py --strategy 1     # data/fatura_{train_7500,val_1250,test_1250}.jsonl
python prepare_fatura.py --strategy 2     # data/fatura_s2_{train_8000,val_1000,test_1000}.jsonl
```

`--root` defaults to `~/datasets/fatura/invoices_dataset_final`.

## Training on Colab (A100)

1. Upload the `.py` files, `requirements.txt` and `data/` to a Drive folder, e.g. `MyDrive/fine_tune/`.
2. **Runtime → Change runtime type → A100 GPU.** Add `WANDB_API_KEY` (and optionally `HF_TOKEN`) under 🔑 Secrets.
3. Run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/fine_tune
```
```python
!pip install -q -U "transformers>=5.20" "trl>=1.16" "peft>=0.21" accelerate datasets wandb "mlflow==3.16.1" psutil nvidia-ml-py
!pip uninstall -y torchao   # Colab's torchao 0.10 breaks PEFT >= 0.21 ("only versions above 0.16.0 are supported")
```
```python
!MAX_STEPS=10 WANDB_RUN_ID=smoke-1 MLFLOW_EXPERIMENT_NAME=fatura-smoke python train.py   # smoke test
!rm -rf checkpoints
!python train.py
```

Don't reinstall torch on Colab, and don't install `flash-attn`: the model uses PyTorch's built-in `sdpa` attention. Pip's warnings about `cudf-cu13` and `google-adk` conflicts can be ignored; neither package is used.

**End every session with `drive.flush_and_unmount()`.** Writes to `/content/drive` are uploaded to Drive in the background. If the runtime ends first, the last files written are lost; in the Strategy 2 run, `checkpoint-250` and `final_adapter` never reached Drive.

```python
from google.colab import drive
drive.flush_and_unmount()
```

**Resuming.** `train.py` resumes from the newest `checkpoint-N` in `OUTPUT_DIR`, restoring weights, optimizer, scheduler and RNG state. After a disconnect, re-run the setup cells and `!python train.py`. Use a **separate `OUTPUT_DIR` per experiment**, or a new run will resume another run's checkpoint.

### Configuration used

| | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, bf16, `sdpa` attention |
| LoRA | r=32, alpha=64, dropout 0.05, all 7 linear projections; 80.7M trainable params (1.05%) |
| Batch | 4 per device × 8 accumulation = 32 effective |
| Optimizer | `adamw_torch_fused`, lr 3e-4, cosine schedule, 5% warmup, weight decay 0.01, grad clip 1.0 |
| Sequence length | `max_length=2048` (longest document is 1,180 tokens) |
| Gradient checkpointing | off (on an 80 GB A100 it isn't needed and costs ~30% speed) |
| Strategy 1 | 3 epochs = 705 steps planned, eval/save every 100; **stopped at 300, used checkpoint-200** |
| Strategy 2 | 1 epoch = 250 steps, eval/save every 50; `final_adapter` = best checkpoint by `eval_loss` |

### Observed on an A100-SXM4-80GB

| | Batch 2 × 16, checkpointing on | Batch 4 × 8, checkpointing off |
|---|---|---|
| GPU memory | ~18.6 GB | ~62 GB training, ~70 GB peak after eval |
| GPU utilization | 87–100% | 91–100% |
| Speed | — | 6.86 s/step, ~3,400 tokens/s (705 steps ≈ 81 min) |

**Convergence (Strategy 1).** Train loss fell from 0.26 to under 0.001 within the first ~75 steps (a third of epoch 1), with token accuracy at 99.98%. `eval_loss` was 5.8e-05 at step 200 and 1.06e-04 at step 300. Both mean roughly 99.99% probability on the correct token; the small rise followed a training-loss blip around step 275 while the learning rate was still high. Checkpoint-200 was kept. One epoch is enough for this data.

## Experiment tracking

`report_to=["mlflow", "wandb"]`. Trainer's built-in callbacks log params, train metrics every 25 steps, `eval_loss` at each eval, and GPU/system metrics. All configuration is in environment variables set in a Colab cell before `!python train.py`; no logging code is needed.

```python
import os
from google.colab import userdata
os.environ["WANDB_API_KEY"]   = userdata.get("WANDB_API_KEY")
os.environ["WANDB_PROJECT"]   = "fatura-qwen-lora"
os.environ["WANDB_RUN_ID"]    = "fatura-run-1"     # fixed ID + RESUME=allow: a resumed run continues the same chart
os.environ["WANDB_RESUME"]    = "allow"
os.environ["WANDB_LOG_MODEL"] = "false"            # don't upload checkpoints

os.environ["MLFLOW_TRACKING_URI"]    = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "fatura-qwen-lora"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
```

**MLflow server.** MLflow runs on a private GCE VM with only SSH open. The tunnel has to be opened **from the Colab runtime**; a tunnel on your laptop doesn't help, because `localhost` on Colab is Colab's machine. If your Colab and GCP accounts are different Google accounts, sign `gcloud` in as the GCP one:

```python
!gcloud auth login --no-launch-browser            # open the URL as the GCP account, paste the code back
!gcloud config set project <gcp-project>
!gcloud compute ssh <mlflow-vm> --zone <zone> --quiet -- \
    -N -f -L 5000:localhost:5000 -o ServerAliveInterval=60 -o ServerAliveCountMax=10
!curl -s http://localhost:5000/health             # OK
```

Match the client to the server version (`mlflow==3.16.1` here). Without `MLFLOW_TRACKING_URI`, MLflow 3.7+ logs to a local `mlflow.db` that disappears with the Colab session.

Metric definitions: `loss` is cross-entropy on the JSON tokens; `entropy` is the model's average uncertainty over the vocabulary at each answer token (added by TRL); `mean_token_accuracy` is the share of answer tokens whose top prediction is correct, measured with the correct previous tokens given (teacher forcing).

## Evaluation

```python
!TEST_JSONL=./data/fatura_test_1250.jsonl ADAPTER_DIR=./checkpoints/qwen25-7b-carrier-lora/checkpoint-200 python evaluate.py
!TEST_JSONL=./data/fatura_s2_test_1000.jsonl ADAPTER_DIR=./checkpoints/qwen25-7b-fatura-s2-lora/final_adapter python evaluate.py
```

The adapter is merged into the bf16 base model and decoding is greedy, at batch 16 with `max_new_tokens=768`. Reports are saved to `results/eval_<test-file>.json` and record the adapter and test file used.

`final_adapter/` only exists when training **finishes**. If you stop early, point `ADAPTER_DIR` at the best `checkpoint-N`, which `best_model_checkpoint` in that checkpoint's `trainer_state.json` names. Otherwise transformers treats the missing path as a Hub repo ID and fails with "Repo id must be in the form…".

Never evaluate a Strategy 1 adapter on the Strategy 2 test set: Strategy 1 trained on all 50 templates, including the held-out ones.

## Inference

```python
!ADAPTER_DIR=./checkpoints/qwen25-7b-fatura-s2-lora/final_adapter python inference.py   # extracts the first test invoice
```

Or keep the model loaded in the notebook:

```python
from inference import CarrierDocumentExtractor
extractor = CarrierDocumentExtractor(adapter_dir="./checkpoints/qwen25-7b-fatura-s2-lora/final_adapter")
result = extractor.extract(carrier_code="TEMPLATE_02", ocr_text=ocr_text, max_new_tokens=768)
```

`save_merged_for_vllm()` exports a standalone merged bf16 model for serving.

**What a checkpoint contains.** Inference needs only `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` and `training_args.bin` exist only for resuming training. `trainer_state.json` holds the full metric history and is worth keeping as a record.

## Limitations

- **The "OCR" is annotation text.** Inputs come from FATURA's labeled boxes, not from running OCR on the images, so there are no misreads, split lines or stray text. Every answer appears word for word in the prompt: the model learns to **assign text blocks to keys**, not to read documents.
- **Unseen layouts, familiar wording.** The held-out templates use the same field labels as the training templates (`TOTAL`, `INVOICE DATE`, `Bill to`, `Terms & Conditions`…). Only seller names and websites, which have no label, are new. That's why Strategy 1 and Strategy 2 score the same: the model reads the label, not the layout. A real new vendor writing "Amount payable" or "Inv. No." isn't tested here.
- **`null`s inflate the metric.** On average fewer than half of the 25 keys are filled in a document, and a correctly predicted `null` counts as a match. 99.99% over 25,000 fields is about 2–3 wrong fields.
- **Ambiguous labels.** A few labels map to more than one key across templates (`INVOICE` → `number`/`title`, `GSTIN` → `gstin`/`gstin_buyer`/`gstin_seller`, `DUE_AMOUNT` → `amount_due`/`total`); these are the likeliest errors.
- **No line items.** Table rows exist only in FATURA's full-page text, which has no boxes, so they're in neither the input nor the target.
- **Values keep their printed labels.** `"TOTAL : 734.33 EUR"` rather than a parsed amount and currency.
- **Leftover wording from the original project.** `SYSTEM_PROMPT` and some names (`carrier_id`, `load_carrier_datasets`, the `qwen25-7b-carrier-lora` output folder) come from a logistics-document version of this code. They're used consistently in training and inference, so they work as-is.

## Possible next steps

1. Report accuracy on non-null fields only, and save per-document predictions to see the actual errors.
2. Relabel the test set with new wording (`TOTAL :` → `Amount payable:`) to test understanding beyond label matching.
3. Run real OCR (PaddleOCR or Tesseract) on FATURA's images, and train or evaluate on that output.
4. Evaluate on a different public invoice or receipt dataset (SROIE, CORD).
5. Normalize values (amounts, dates, currency) in the target instead of keeping printed labels.

## References

- Limam, Dhiaf, Kessentini (2023). *FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and Understanding.* [arXiv:2311.11856](https://arxiv.org/abs/2311.11856)
- Dataset: [zenodo.org/records/10371464](https://zenodo.org/records/10371464)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [TRL](https://github.com/huggingface/trl) · [PEFT](https://github.com/huggingface/peft)
