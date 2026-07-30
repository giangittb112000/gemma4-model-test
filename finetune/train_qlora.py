#!/usr/bin/env python3
"""QLoRA fine-tune → LoRA adapter (serve bằng vLLM --enable-lora).

  docker compose -f compose.train.yaml run --rm train
  → ./models/adapters/query-parser-ft/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def preflight() -> None:
    missing = []
    for mod in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "Thiếu package: "
            + ", ".join(missing)
            + "\nChạy bằng: docker compose -f compose.train.yaml run --rm train"
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Không thấy GPU CUDA.\n"
            "- Tắt vLLM/Ollama đang chiếm GPU: docker compose stop\n"
            "- Kiểm tra: nvidia-smi"
        )

    print(f"[ok] torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")


preflight()

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA → LoRA adapter")
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "train.json")
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/models/adapters/query-parser-ft"),
    )
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def load_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def build_model_and_tokenizer(
    model_id: str, token: str | None, lora_r: int, lora_alpha: int
):
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"[train] loading: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Không load được tokenizer ({exc}).\n"
            "Kiểm tra HF_TOKEN trong .env và quyền google/gemma-4-e2b-it."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            token=token,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Không load được model ({exc}).\n"
            "Tắt vLLM trước khi train; kiểm tra nvidia-smi / VRAM."
        ) from exc

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
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
    return model, tokenizer, lora


def main() -> None:
    args = parse_args()
    token = load_token()
    if not token:
        print(
            "[warn] Không thấy HF_TOKEN — model gated có thể fail.",
            file=sys.stderr,
        )
    else:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    if not args.train_file.exists():
        raise SystemExit(f"Thiếu file train: {args.train_file}")

    print(f"[train] model={args.model_id}")
    print(f"[train] train_file={args.train_file}")
    print(f"[train] adapter_dir={args.adapter_dir}")

    model, tokenizer, lora_cfg = build_model_and_tokenizer(
        args.model_id, token, args.lora_r, args.lora_alpha
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=str(args.train_file), split="train")
    print(f"[train] samples={len(train_ds)}")

    use_bf16 = torch.cuda.is_bf16_supported()
    sft_kwargs = dict(
        output_dir=str(args.adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
    )
    try:
        sft_args = SFTConfig(**sft_kwargs, max_length=args.max_seq_length)
    except TypeError:
        sft_args = SFTConfig(**sft_kwargs, max_seq_length=args.max_seq_length)

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
        )

    trainer.train()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))
    print(f"[done] adapter -> {args.adapter_dir}")
    print("       docker compose up -d  rồi  MODEL_ID=query-parser-ft make test")


if __name__ == "__main__":
    main()
