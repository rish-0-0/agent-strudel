import argparse
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(__file__).resolve().parent / "out"
VALIDATE_JS = REPO_ROOT / "training-data-gen" / "validator" / "validate.js"


def parse_args():
    p = argparse.ArgumentParser(description="Generate Strudel from a chat prompt with the fine-tuned model.")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Checkpoint dir or HF model id.")
    p.add_argument("--prompt", default="generate a groovy jazz music having indian fusion")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-validate", action="store_true", help="Skip validate.js check.")
    return p.parse_args()


def load_model(model_path):
    for attn in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attn)
            print(f"[model] loaded with attn={attn}")
            return model
        except Exception as e:
            print(f"[model] attn={attn} unavailable: {e}")
    raise RuntimeError("could not load model")


def validate(code):
    proc = subprocess.run(
        ["node", str(VALIDATE_JS)],
        input=code,
        capture_output=True,
        text=True,
        cwd=str(VALIDATE_JS.parent),
        timeout=30,
    )
    out = proc.stdout.strip()
    return out.startswith("OK"), out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[model] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in input_ids.items()}
    prompt_len = inputs["input_ids"].shape[1]

    print(f"[generate] prompt: {args.prompt!r}")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][prompt_len:]
    code = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n=== generated strudel ===")
    print(code)
    print("=== end ===\n")

    if not args.no_validate:
        print("[validate] running validate.js ...")
        ok, msg = validate(code)
        print(f"[validate] {'OK - compiles and runs' if ok else 'FAILED: ' + msg}")


if __name__ == "__main__":
    main()
