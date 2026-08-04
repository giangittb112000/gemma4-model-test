#!/usr/bin/env python3
"""QLoRA fine-tune → LoRA adapter (serve bằng vLLM --enable-lora).

  docker compose -f compose.train.yaml run --rm train
  → ./models/adapters/query-parser-ft/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

from prompt import SYSTEM_PROMPT

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
    # Prompt JSON ngắn (~150–200 tok) — 256 đủ, tiết kiệm VRAM → batch lớn hơn.
    p.add_argument("--max-seq-length", type=int, default=256)
    # Ưu tiên vòng test nhanh: 1 epoch. Full quality có thể --epochs 2|3.
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Giới hạn step (smoke test). -1 = theo epochs.",
    )
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

    # Gemma 4: vision/audio dùng Gemma4ClippableLinear — PEFT không hỗ trợ.
    # List kiểu ["q_proj", ...] sẽ match cả tower multimodal → crash.
    # Regex chỉ language_model (cùng convention peft>=0.19 default cho gemma4).
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)",
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
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as exc:
        if "Gemma4ClippableLinear" not in str(exc):
            raise
        raise SystemExit(
            f"PEFT từ chối Gemma4ClippableLinear ({exc}).\n"
            "Script đã dùng regex language_model — nếu vẫn lỗi: "
            "rebuild image (pip peft mới) hoặc báo lại full traceback."
        ) from exc
    model.print_trainable_parameters()
    vram("after-lora")
    if model.get_nb_trainable_parameters()[0] == 0:
        raise SystemExit(
            "0 trainable params — regex target_modules không khớp architecture. "
            "In vài tên module: "
            + ", ".join(n for n, _ in list(model.named_modules())[:30])
        )

    train_ds = load_dataset("json", data_files=str(args.train_file), split="train")
    # Chuẩn TRL/HF conversational SFT: mỗi mẫu có cột "messages"
    #   [{"role":"system"|"user"|"model", "content": "..."}, ...]
    # Gemma dùng role "model" (không phải "assistant").
    # Tuỳ chọn shorthand: {query, output} → tự ghép messages (dùng SYSTEM_PROMPT).
    if "messages" not in train_ds.column_names:
        if "query" in train_ds.column_names and "output" in train_ds.column_names:

            def to_messages(row: dict) -> dict:
                out = row["output"]
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                return {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f'query: "{row["query"]}"'},
                        {"role": "model", "content": out},
                    ]
                }

            train_ds = train_ds.map(
                to_messages, remove_columns=train_ds.column_names
            )
        else:
            raise SystemExit(
                "train.json cần chuẩn thư viện: [{ \"messages\": ["
                "{\"role\",\"content\"}, ...] }, ...]\n"
                "Hoặc shorthand: [{ \"query\", \"output\" }, ...]"
            )

    print(f"[train] samples={len(train_ds)}")
    # In raw mẫu đầu — đúng thứ SFTTrainer nhận trước khi apply_chat_template.
    sample0 = train_ds[0]["messages"]
    print("[train] raw messages[0] (trl conversational):")
    print(json.dumps(sample0, ensure_ascii=False, indent=2))

    use_bf16 = torch.cuda.is_bf16_supported()
    # ~9725 mẫu / batch2 ≈ 4863 micro-step/epoch — nhanh hơn ~2× so với batch=1;
    # epochs=1 → ~1/3 thời gian so với epochs=3.
    print(
        f"[train] epochs={args.epochs} batch={args.batch_size} "
        f"accum={args.grad_accum} max_seq={args.max_seq_length} "
        f"max_steps={args.max_steps}"
    )
    sft_kwargs = dict(
        output_dir=str(args.adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Gemma 4 multimodal: giữ cột phụ nếu collator/model cần.
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        max_steps=args.max_steps,
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
    print("       docker compose up -d  rồi  make test Q=\"...\"")


if __name__ == "__main__":
    main()
