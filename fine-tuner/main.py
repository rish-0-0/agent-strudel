import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = (
    "You are a Strudel live-coding assistant. Given a description of a groove, "
    "write valid Strudel (strudel.cc) code that produces it. Respond with only "
    "the Strudel code, no explanation."
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "training.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SmolLM2-135M-Instruct on Strudel snippets.")
    p.add_argument("--model", default=MODEL_ID, help="HF model id to fine-tune.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to training.jsonl.")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"), help="Output dir.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-train-samples", type=int, default=None, help="Cap train size (smoke test).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(path, max_train_samples=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    examples = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Write Strudel code for: {r['label']}"},
            ],
            "completion": [
                {"role": "assistant", "content": r["code"]},
            ],
        }
        for r in records
    ]
    ds = Dataset.from_list(examples)
    if max_train_samples is not None:
        ds = ds.select(range(min(max_train_samples, len(ds))))
    return ds


def resolve_dtype_and_precision():
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def load_model(model_id, dtype):
    for attn in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, attn_implementation=attn
            )
            print(f"[model] loaded with attn={attn}, dtype={dtype}")
            return model
        except Exception as e:
            print(f"[model] attn={attn} unavailable: {e}")
    raise RuntimeError("could not load model with any attention implementation")


def main():
    args = parse_args()

    print(f"[data] loading {args.data}")
    train_ds = load_dataset(args.data, args.max_train_samples)
    print(f"[data] train={len(train_ds)}")

    dtype, bf16, fp16 = resolve_dtype_and_precision()
    print(f"[model] downloading/loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_seq_length,
        packing=False,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=False,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[done] model saved to {args.out}")


if __name__ == "__main__":
    main()
