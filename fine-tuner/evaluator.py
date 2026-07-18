import argparse
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

from main import SYSTEM_PROMPT
from generate import load_model as load_ft_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(__file__).resolve().parent / "out"
VALIDATE_JS = REPO_ROOT / "training-data-gen" / "validator" / "validate.js"
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
JUDGE_MODEL = "glm-5.2"

EVAL_SYSTEM = """You are a Strudel live-coding music evaluator. You assess a fine-tuned language model's ability to generate Strudel (strudel.cc) code from a music description.

You have tools:
- generate_strudel(description): generate Strudel code from a description using the fine-tuned model.
- validate_strudel(code): headlessly compile/run the code; returns OK or an ERROR.

Procedure for the given music description:
1. Call generate_strudel with the description.
2. Call validate_strudel on the returned code.
3. Judge the quality.

Scoring (1-10):
- Does the code compile? (use the validation result)
- Does it match the described genre/vibe/instruments?
- Is it musically coherent and non-trivial (layers, rhythm, harmony)?

Your FINAL answer must be exactly:
SCORE: <1-10>
COMPILES: <yes|no>
MATCH: <how well it matches the description>
REASONING: <concise justification>
CODE:
<the generated strudel code>"""

DEFAULT_PROMPTS = [
    "generate a groovy jazz music having indian fusion",
    "a chill lofi hip hop beat with vinyl crackle and mellow chords",
    "driving 4-on-the-floor deep house at 124 bpm with filtered piano",
    "dark minor-key techno with rolling bass and hypnotic hats",
    "warm latin afrobeat groove with percussion and brass stabs",
]

_STATE = {"model": None, "tokenizer": None}


def get_ft_model(model_path):
    if _STATE["model"] is None:
        print(f"[eval] loading fine-tuned model from {model_path}")
        _STATE["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        _STATE["model"] = load_ft_model(model_path)
        _STATE["model"].eval()
    return _STATE["model"], _STATE["tokenizer"]


def make_tools(model_path):
    @tool
    def generate_strudel(description: str) -> str:
        """Generate Strudel code from a natural-language music description using the fine-tuned local model. Returns the raw Strudel code."""
        model, tokenizer = get_ft_model(model_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Write Strudel code for: {description}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        code = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        return code

    @tool
    def validate_strudel(code: str) -> str:
        """Validate Strudel code by running it headlessly. Returns 'OK - compiles and runs' or an error message."""
        import subprocess
        proc = subprocess.run(
            ["node", str(VALIDATE_JS)],
            input=code,
            capture_output=True,
            text=True,
            cwd=str(VALIDATE_JS.parent),
            timeout=30,
        )
        out = proc.stdout.strip()
        return "OK - compiles and runs" if out.startswith("OK") else out

    return [generate_strudel, validate_strudel]


def build_agent(model_path, api_key):
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=ZAI_BASE_URL,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    )
    tools = make_tools(model_path)
    return create_react_agent(llm, tools, prompt=EVAL_SYSTEM)


def parse_args():
    p = argparse.ArgumentParser(description="LangChain evaluator agent: FT model generates, GLM-5.2 judges.")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Fine-tuned checkpoint dir.")
    p.add_argument("--prompt", default=None, help="Single prompt (overrides the default prompt set).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_score(text):
    m = re.search(r"SCORE:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("Z_AI_API_KEY")
    if not api_key:
        raise SystemExit("Z_AI_API_KEY not found in .env")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    executor = build_agent(args.model, api_key)
    print(f"[eval] running {len(prompts)} prompt(s) on {args.model}")

    scores = []
    for i, prompt in enumerate(prompts, 1):
        torch.manual_seed(args.seed)
        print(f"\n[eval] prompt {i}/{len(prompts)}: {prompt}")
        result = executor.invoke({"messages": [{"role": "user", "content": prompt}]})
        out = result["messages"][-1].content
        s = parse_score(out)
        print(f"[eval] score={s}")
        print(out)
        if s is not None:
            scores.append(s)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nFINAL_SCORE: {avg:.2f}")


if __name__ == "__main__":
    main()
