"""Training-data generator agent.

Bilevel generation loop (see ARCHITECTURE.md):
  1. Generator (Sonnet-5, low effort, agentic) writes a Strudel snippet and
     self-validates it via the `validate_strudel` tool until it compiles.
  2. Evaluator (Sonnet-5, low effort) strictly judges whether the validated
     snippet is music humans enjoy and writes a descriptive label + genre tag.
  3. A diversity guard rejects near-duplicate labels and caps per-genre counts
     so the dataset stays varied.
  4. Accepted snippets are written to data/training.jsonl as {"label","code"}.
  5. Every N attempts the running state is compacted into a summary that steers
     the generator toward unexplored styles.

Usage:  uv run python training-data-gen/main.py --limit 30 --fresh
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent import GENRE_LIST, evaluate_strudel, generate_strudel, summarize_state

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_api_key() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not found in training-data-gen/.env", file=sys.stderr)
        sys.exit(1)
    return key


@dataclass
class State:
    accepted_labels: list[str] = field(default_factory=list)
    accepted_genres: list[str] = field(default_factory=list)
    genre_counts: Counter = field(default_factory=Counter)
    accepted_scores: Counter = field(default_factory=Counter)
    summary: str = ""
    recent_rejections: list[str] = field(default_factory=list)
    attempts: int = 0
    accepted: int = 0
    rejected_quality: int = 0
    rejected_diversity: int = 0
    total_cost: float = 0.0


_TOKEN = re.compile(r"[a-z0-9]+")


def _label_similarity(a: str, b: str) -> float:
    ta, tb = set(_TOKEN.findall(a.lower())), set(_TOKEN.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _too_similar(label: str, existing: list[str], threshold: float = 0.55) -> bool:
    return any(_label_similarity(label, other) >= threshold for other in existing)


async def run(
    limit: int, max_attempts: int, max_turns: int,
    max_per_genre: int, summarize_every: int, fresh: bool,
) -> None:
    api_key = load_api_key()
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "training.jsonl"
    log_path = DATA_DIR / "run.log"

    if fresh and out_path.exists():
        out_path.unlink()
    if fresh and log_path.exists():
        log_path.unlink()

    logf = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    st = State()
    genre_pool: list[str] = []

    if not fresh and out_path.exists():
        loaded = 0
        with out_path.open("r", encoding="utf-8") as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st.accepted_labels.append(rec.get("label", ""))
                g = rec.get("genre") or "uncategorized"
                st.accepted_genres.append(g)
                st.genre_counts[g] += 1
                st.accepted_scores[rec.get("score", 0)] += 1
                loaded += 1
        if loaded:
            log(f"  preloaded {loaded} existing samples for diversity guard")

    def next_genre() -> str:
        if not genre_pool:
            genre_pool.extend(GENRE_LIST)
            random.shuffle(genre_pool)
        return genre_pool.pop()

    log(f"=== run start: target={limit} max_attempts={max_attempts} "
        f"max_per_genre={max_per_genre} fresh={fresh} ===")

    with out_path.open("a", encoding="utf-8") as outf:
        while st.accepted < limit and st.attempts < max_attempts:
            st.attempts += 1
            target_genre = next_genre()
            log(f"--- attempt #{st.attempts} (accepted {st.accepted}/{limit}) "
                f"target={target_genre} ---")

            gen = await generate_strudel(
                st.accepted_labels, target_genre, st.summary,
                api_key, max_turns=max_turns,
            )
            st.total_cost += gen.cost

            if not gen.ok:
                log(f"  GEN FAILED (no valid code) turns={gen.turns} cost=${gen.cost:.4f}")
                st.recent_rejections.append("generator produced no valid code")
                continue

            log(f"  GEN OK turns={gen.turns} cost=${gen.cost:.4f}; judging...")
            ev = await evaluate_strudel(gen.code, st.accepted_labels, api_key)

            if not ev.accept or ev.score < 8:
                st.rejected_quality += 1
                st.recent_rejections.append(f"score {ev.score}: {ev.reason}")
                log(f"  REJECTED quality (score={ev.score}): {ev.reason}")
                continue

            if _too_similar(ev.label, st.accepted_labels):
                st.rejected_diversity += 1
                st.recent_rejections.append(f"dup label: {ev.label}")
                log(f"  REJECTED diversity (duplicate label): {ev.label}")
                continue

            genre = ev.genre or "uncategorized"
            if st.genre_counts[genre] >= max_per_genre:
                st.rejected_diversity += 1
                st.recent_rejections.append(f"genre cap: {genre}")
                log(f"  REJECTED diversity (genre '{genre}' capped at {max_per_genre})")
                continue

            st.accepted += 1
            st.accepted_labels.append(ev.label)
            st.accepted_genres.append(genre)
            st.genre_counts[genre] += 1
            st.accepted_scores[ev.score] += 1
            st.recent_rejections.clear()
            record = {"label": ev.label, "genre": genre, "score": ev.score, "code": gen.code}
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            outf.flush()
            log(f"  ACCEPTED #{st.accepted} [{genre}|score {ev.score}]: {ev.label}")

            if summarize_every and st.attempts % summarize_every == 0:
                log("  ...compacting state...")
                st.summary = await summarize_state(
                    st.accepted_labels, st.accepted_genres,
                    st.recent_rejections[-10:], api_key,
                )
                log("  state summary updated.")

    log(f"=== done: accepted {st.accepted}/{st.attempts} attempts "
        f"(quality rejects {st.rejected_quality}, diversity rejects {st.rejected_diversity}), "
        f"cost ${st.total_cost:.4f} ===")

    log("--- variety report ---")
    log(f"  accepted score distribution: {dict(sorted(st.accepted_scores.items()))}")
    log(f"  distinct genres: {len(st.genre_counts)}")
    for genre, count in st.genre_counts.most_common():
        log(f"    {genre}: {count}")
    log("  labels:")
    for i, lbl in enumerate(st.accepted_labels, 1):
        log(f"   {i:2d}. {lbl}")
    logf.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Strudel training-data generator")
    ap.add_argument("--limit", type=int, default=30, help="target number of accepted samples")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="attempt cap (0 = 6x limit)")
    ap.add_argument("--max-turns", type=int, default=8, help="max agent turns per generation")
    ap.add_argument("--max-per-genre", type=int, default=3,
                    help="cap per genre (use ~100 for a full per-category run)")
    ap.add_argument("--summarize-every", type=int, default=20, help="compact state every N attempts")
    ap.add_argument("--fresh", action="store_true", help="truncate output files before running")
    args = ap.parse_args()
    max_attempts = args.max_attempts or 6 * args.limit
    asyncio.run(run(args.limit, max_attempts, args.max_turns,
                    args.max_per_genre, args.summarize_every, args.fresh))


if __name__ == "__main__":
    main()
