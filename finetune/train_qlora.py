#!/usr/bin/env python3
"""Fine-tune google/gemma-4-e2b-it bằng QLoRA rồi merge ra model đầy đủ.

Output mặc định:
  outputs/adapter/  — LoRA adapter (trung gian)
  outputs/merged/   — model đã merge (dùng cho vLLM qua compose.merged.yaml)

Cách chạy:
  cd finetune
  python train_qlora.py
  python train_qlora.py --skip-merge   # chỉ lưu adapter
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune + merge Gemma 4 E2B")
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "train.json")
    p.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Optional JSON eval file (mặc định không dùng).",
    )
    p.add_argument("--adapter-dir", type=Path, default=ROOT / "outputs" / "adapter")
    p.add_argument("--merged-dir", type=Path, default=ROOT / "outputs" / "merged")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--skip-merge", action="store_true", help="Không merge sau train")
    return p.parse_args()


def load_token() -> str | None:
    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def build_model_and_tokenizer(model_id: str, token: str | None):
    if not torch.cuda.is_available():
        raise SystemExit("Cần GPU CUDA để train QLoRA. Hãy chạy trên server GPU.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        token=token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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
    # overwrite r/alpha from args in main
    return model, tokenizer, lora


def merge_adapter(
    model_id: str,
    adapter_dir: Path,
    merged_dir: Path,
    token: str | None,
) -> None:
    print(f"[merge] base={model_id} + adapter={adapter_dir} -> {merged_dir}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Merge trên CPU để tránh OOM khi GPU đang đầy sau train.
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        token=token,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    tok.save_pretrained(str(merged_dir))
    print(f"[merge] done -> {merged_dir}")


def main() -> None:
    args = parse_args()
    token = load_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    if not args.train_file.exists():
        raise SystemExit(f"Thiếu file train: {args.train_file}")

    print(f"[train] model={args.model_id}")
    print(f"[train] train_file={args.train_file}")
    model, tokenizer, lora_cfg = build_model_and_tokenizer(args.model_id, token)
    lora_cfg.r = args.lora_r
    lora_cfg.lora_alpha = args.lora_alpha
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=str(args.train_file), split="train")
    eval_ds = None
    if args.eval_file is not None and args.eval_file.exists():
        eval_ds = load_dataset("json", data_files=str(args.eval_file), split="train")

    use_bf16 = torch.cuda.is_bf16_supported()
    sft_args = SFTConfig(
        output_dir=str(args.adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_length,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))
    print(f"[train] adapter saved -> {args.adapter_dir}")

    # Giải phóng GPU trước khi merge trên CPU.
    del trainer
    del model
    torch.cuda.empty_cache()

    if args.skip_merge:
        print("[train] skip merge (--skip-merge)")
        return

    merge_adapter(args.model_id, args.adapter_dir, args.merged_dir, token)
    print("[done] Dùng merged model với: docker compose -f compose.merged.yaml up -d")


if __name__ == "__main__":
    main()
