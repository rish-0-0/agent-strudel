"""Overnight fine-tuning orchestrator.

Doubles the dataset and, at each milestone, retrains SmolLM2-135M-Instruct from
base on ALL accumulated data, then runs the GLM-5.2 evaluator. Stops when quality
decreases, the sample cap is hit, the cost cap is hit, or max-hours elapse.

Resumable: state is checkpointed to overnight_state.json after every round, and
training.jsonl is the source of truth for sample count. Re-run to resume.

By default --cap = current sample count, i.e. NO new generation (train+eval only
on pending milestones). Pass --cap N to allow growth up to N total samples.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "training.jsonl"
RUN_LOG = REPO / "data" / "run.log"
STATE_FILE = REPO / "overnight_state.json"
LOG = REPO / "overnight.log"

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DECREASE_TOL = 0.5


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_samples():
    if not DATA.exists():
        return 0
    n = 0
    with DATA.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"rounds": [], "last_score": None, "baseline": None, "total_cost": 0.0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def compute_spent():
    """Cumulative Anthropic spend (USD) parsed from data/run.log.

    Uses each completed run's '=== done ... cost $X ===' total; for a crashed or
    still-running run (no done line), sums its per-attempt 'cost=$X' gen costs.
    """
    if not RUN_LOG.exists():
        return 0.0
    total = 0.0
    run_attempts = 0.0
    with RUN_LOG.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "=== run start:" in line:
                total += run_attempts
                run_attempts = 0.0
            elif "=== done:" in line:
                m = re.search(r"cost \$([\d.]+)", line)
                if m:
                    total += float(m.group(1))
                run_attempts = 0.0
            else:
                for m in re.findall(r"cost=\$([\d.]+)", line):
                    try:
                        run_attempts += float(m)
                    except ValueError:
                        pass
    total += run_attempts
    return total


def parse_cost(stdout):
    total = 0.0
    for m in re.findall(r"cost=\$([\d.]+)", stdout or ""):
        try:
            total += float(m)
        except ValueError:
            pass
    return total


def sh(cmd, timeout=None):
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole process tree (uv spawns python children on Windows).
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            out, err = proc.communicate(timeout=15)
        except Exception:
            out, err = "", ""
        log(f"  TIMEOUT after {timeout}s (tree killed)")
        return None
    if proc.returncode != 0:
        log(f"  FAILED ({proc.returncode}) stderr: {err[-1500:]}")
    else:
        log("  ok")
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_data_gen(delta, milestone, remaining_sec):
    max_per_genre = max(20, milestone // 4)
    cmd = ["uv", "run", "python", "training-data-gen/main.py",
           "--limit", str(delta), "--max-per-genre", str(max_per_genre)]
    timeout = min(delta * 120 + 600, max(remaining_sec - 60, 600))
    return sh(cmd, timeout=timeout)


def run_train(milestone):
    out_dir = str(REPO / "fine-tuner" / f"out_m{milestone}")
    cmd = ["uv", "run", "--project", "fine-tuner", "python", "fine-tuner/main.py",
           "--model", BASE_MODEL, "--data", str(DATA),
           "--out", out_dir, "--epochs", "3"]
    timeout = milestone * 12 + 1800
    return sh(cmd, timeout=timeout), out_dir


def run_eval(model_dir):
    cmd = ["uv", "run", "--project", "fine-tuner", "python", "fine-tuner/evaluator.py",
           "--model", model_dir, "--seed", "42"]
    r = sh(cmd, timeout=3000)
    if r is None or r.returncode != 0:
        return None
    m = re.search(r"FINAL_SCORE:\s*([\d.]+)", r.stdout)
    return float(m.group(1)) if m else None


def parse_args():
    p = argparse.ArgumentParser(description="Overnight fine-tuning orchestrator.")
    p.add_argument("--cap", type=int, default=None,
                   help="Max total samples. Default = current count (no new generation).")
    p.add_argument("--max-cost", type=float, default=60.0, help="Anthropic spend cap (USD).")
    p.add_argument("--max-hours", type=float, default=8.0, help="Wall-clock cap (hours).")
    return p.parse_args()


def main():
    args = parse_args()
    state = load_state()
    start = time.time()
    current = count_samples()
    cap = args.cap if args.cap is not None else current
    if state["baseline"] is None:
        state["baseline"] = current
        save_state(state)
    state["total_cost"] = max(state.get("total_cost", 0.0), compute_spent())
    log(f"=== overnight orchestration start | baseline={state['baseline']} "
        f"current={current} cap={cap} max_cost=${args.max_cost} "
        f"spent=${state['total_cost']:.2f} max_hours={args.max_hours} ===")

    while True:
        elapsed = time.time() - start
        if elapsed > args.max_hours * 3600:
            log(f"max hours ({args.max_hours}h) reached, stopping")
            break
        if state["total_cost"] >= args.max_cost:
            log(f"budget cap ${args.max_cost} reached (spent ${state['total_cost']:.2f}), stopping")
            break

        current = count_samples()
        completed = {state["baseline"]} | {r["milestone"] for r in state["rounds"]}

        if current >= cap:
            log(f"cap ({cap}) reached, stopping")
            break

        if current in completed:
            nxt = min(current * 2, cap)
            if nxt <= current:
                log("no further milestone possible, stopping")
                break
            delta = nxt - current
            log(f"--- round: generate {delta} new (target total {nxt}) ---")
            remaining = args.max_hours * 3600 - elapsed
            r = run_data_gen(delta, nxt, remaining)
            if r is not None:
                state["total_cost"] += parse_cost(r.stdout)
                save_state(state)
                log(f"  spent so far: ${state['total_cost']:.2f}")
            milestone = count_samples()
            if milestone <= current:
                log("no new samples generated, stopping")
                break
            if milestone in completed:
                log(f"milestone {milestone} already completed, looping")
                continue
            log(f"reached {milestone} samples")
        else:
            milestone = current
            log(f"--- resuming pending milestone {milestone}: train + eval ---")

        tr, out_dir = run_train(milestone)
        if tr is None:
            log("training failed/timed out, stopping")
            break

        score = run_eval(out_dir)
        if score is None:
            log("eval failed, stopping")
            break
        log(f"RESULT milestone={milestone} score={score:.2f}")

        prev = state["last_score"]
        state["rounds"].append({"milestone": milestone, "score": score, "model": out_dir})
        if prev is not None and score < prev - DECREASE_TOL:
            log(f"quality decreased ({prev:.2f} -> {score:.2f}), stopping")
            state["last_score"] = score
            save_state(state)
            break
        state["last_score"] = score
        save_state(state)

    log(f"=== done. rounds={json.dumps(state['rounds'])} spent=${state['total_cost']:.2f} ===")


if __name__ == "__main__":
    sys.exit(main())
