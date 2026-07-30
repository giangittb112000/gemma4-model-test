#!/usr/bin/env python3
"""QLoRA fine-tune → LoRA adapter (serve bằng vLLM --enable-lora).

  docker compose -f compose.train.yaml run --rm train
  → ./models/adapters/query-parser-ft/
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Giảm phân mảnh VRAM trước khi import torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
            "- Tắt vLLM/Ollama: docker compose stop && sudo systemctl stop ollama\n"
            "- Kiểm tra: nvidia-smi"
        )

    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / (1024**3), total / (1024**3)
    print(
        f"[ok] torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} "
        f"| free {free_gb:.1f}/{total_gb:.1f} GiB"
    )
    if free_gb < 12.0:
        print(
            f"[warn] VRAM trống chỉ {free_gb:.1f} GiB — dễ OOM.\n"
            "       Tắt hết process GPU khác (vLLM/Ollama), rồi: nvidia-smi",
            file=sys.stderr,
        )


preflight()

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
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
    # Prompt trong train.json rất dài — 768 đủ smoke-test, đỡ OOM khi train.
    p.add_argument("--max-seq-length", type=int, default=768)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def load_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def vram(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    print(f"[vram:{tag}] used={used:.2f} GiB  free={free / (1024**3):.2f} GiB")


def count_linear4bit(model) -> int:
    return sum(1 for m in model.modules() if m.__class__.__name__ == "Linear4bit")


def prepare_for_kbit_light(model):
    """Thay peft.prepare_model_for_kbit_training — tránh upcast embedding → OOM 16GB.

    peft mặc định ép mọi tensor fp16/bf16 (kể cả embedding) sang fp32, dễ xin thêm
    ~8GB một lần. Ở đây chỉ freeze + upcast *Norm + bật grad checkpoint.
    """
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        name = module.__class__.__name__.lower()
        if "norm" not in name:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype in (torch.float16, torch.bfloat16):
            module.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _require_grad(_module, _inp, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(_require_grad)

    model.gradient_checkpointing_enable()
    return model


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
    tokenizer.padding_side = "right"

    try:
        # Không truyền torch_dtype cùng quantization_config — tránh load lệch dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": 0},
            token=token,
            attn_implementation="sdpa",
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Không load được model ({exc}).\n"
            "Tắt vLLM/Ollama trước khi train; kiểm tra nvidia-smi / VRAM."
        ) from exc

    n4 = count_linear4bit(model)
    print(f"[train] Linear4bit layers: {n4}")
    if n4 == 0:
        raise SystemExit(
            "Model không vào 4-bit (0 Linear4bit). "
            "Nâng transformers/bitsandbytes trong image rồi build lại."
        )
    vram("after-load")
    gc.collect()
    torch.cuda.empty_cache()

    model = prepare_for_kbit_light(model)
    vram("after-prepare")

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Ít module hơn → ít VRAM gradient; đủ cho smoke-test.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    print(f"[train] max_seq_length={args.max_seq_length}")

    model, tokenizer, lora_cfg = build_model_and_tokenizer(
        args.model_id, token, args.lora_r, args.lora_alpha
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    vram("after-lora")

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
        # Tránh lưu optimizer state vào disk/VRAM không cần thiết cho test.
        save_total_limit=1,
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
    print("       docker compose up -d  rồi  make test MODEL=query-parser-ft Q=\"...\"")


if __name__ == "__main__":
    main()
