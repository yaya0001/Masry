# Llama-3.2-3B LoRA Fine-tuning — Technical Troubleshooting Assistant

Fine-tuning **Meta Llama-3.2-3B-Instruct** on Stack Overflow Q&A data using QLoRA (4-bit) via [Unsloth](https://github.com/unslothai/unsloth). The goal is a domain-adapted assistant that answers technical troubleshooting questions in a Stack Overflow style.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/lora_finetune_local__2_.ipynb)

---

## Overview

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | QLoRA (4-bit quantization + LoRA adapters) |
| Dataset | Stack Overflow Q&A — 79,901 examples |
| Training time | ~4h 47m on NVIDIA RTX 5000 Ada (32 GB) |
| Trainable parameters | 9.2M / 3.2B (0.28%) |
| Final training loss | 2.09 → 1.76 over 2 epochs |

---

## Repository structure

```
├── lora_finetune_local__2_.ipynb   # Main notebook (training + evaluation)
└── README.md
```

> The fine-tuned LoRA adapters are not included in this repo due to file size. See [Loading the fine-tuned model](#loading-the-fine-tuned-model) below.

---

## Quick start (Google Colab)

1. Click the **Open in Colab** badge above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Upload your dataset JSON to Google Drive
4. Update `DATA_PATH` in the *Load dataset* cell to point to your file
5. **Runtime → Run all**

The first cell installs all dependencies automatically:
```python
!pip install -q unsloth trl transformers datasets bitsandbytes peft accelerate rouge-score
```

---

## Dataset format

The notebook expects a JSON file where each record has a `text` field in the following format:

```json
{"text": "<s>[INST] Your question here [/INST] The answer here </s>"}
```

This matches the standard Stack Overflow export format used during training.

---

## Training configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Batch size (per device) | 4 |
| Gradient accumulation | 2 (effective batch = 8) |
| Epochs | 2 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| Precision | bfloat16 |
| Max sequence length | 512 (with packing) |

---

## Evaluation

The notebook includes a perplexity evaluation section. After training, load your saved model and run:

```python
eval_loss, eval_ppl = compute_perplexity(eval_dataset)
print(f"Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")
```

---

## Loading the fine-tuned model

Upload the saved `llama3-3b-finetuned/` folder to your Google Drive, then load it with:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "/content/drive/MyDrive/your_folder/llama3-3b-finetuned",
    max_seq_length = 1024,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
```

---

## Requirements

All installed automatically in the notebook. Manual install:

```bash
pip install unsloth trl transformers datasets bitsandbytes peft accelerate rouge-score
```

Requires a CUDA-capable GPU with at least **15 GB VRAM** (Colab T4 is sufficient).

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for the optimized QLoRA training
- [Meta AI](https://ai.meta.com/) for Llama 3.2
- [Stack Overflow](https://stackoverflow.com/) dataset via Hugging Face
