"""Generator + evaluator LLM calls for the bilevel training-data loop.

- :func:`generate_strudel`: an agentic Sonnet-5 call with a ``validate_strudel``
  tool. It self-corrects until the snippet compiles, then returns the code.
- :func:`evaluate_strudel`: a tool-less Sonnet-5 call that strictly judges whether
  a validated snippet is *music humans enjoy* and writes a descriptive label +
  genre tag + score.
- :func:`summarize_state`: compacts the running state every N generations.
"""

import json
import re
from dataclasses import dataclass

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

MODEL = "claude-sonnet-5"
EFFORT = "low"

# Broad taxonomy to steer coverage. Not exhaustive — the generator may invent
# new styles, but should spread across these families and avoid repeats.
TAXONOMY = (
    "techno, acid, deep house, house, microhouse, garage, uk garage, breakbeat, "
    "drum & bass, jungle, dubstep, bass, footwork, electro, idm, glitch, ambient, "
    "drone, downtempo, chillout, synthwave, vaporwave, industrial, ebm, trance, "
    "hardcore, lo-fi house, disco, funk, soul, r&b, boogie, g-funk, boom bap, "
    "lo-fi hip-hop, trap, drill, afrobeat, highlife, dembow, reggaeton, dancehall, "
    "dub, reggae, latin, samba, bossa, cumbia, salsa, gamelan, celtic, klezmer, "
    "bluegrass, jazz, modal jazz, fusion, bebop, soul jazz, smooth jazz, rock, "
    "punk, shoegaze, post-rock, surf, metal, cinematic, berlin school, new age, "
    "indian classical, chinese classical, japanese classical, middle eastern classical, "
    "western classical, baroque, romantic, impressionist, modernist, contemporary, "
    "experimental, avant-garde, minimalism, musique concrete, sound art, field recording, "
)
GENRE_LIST = [g.strip() for g in TAXONOMY.split(",") if g.strip()]

GEN_SYSTEM = """You are a music producer writing Strudel (strudel.cc) live-coding code that sounds like music humans actually enjoy — not just code that compiles.

Musicality principles for every snippet:
- Groove: a clear, felt pulse with well-placed kicks, snares/claps, and hats/swing. The rhythm should make you want to move.
- Harmony & melody: intentional, genre-appropriate chords/bass/lead that resolve and don't clash. Bass should root the harmony.
- Arrangement: layer >=3 distinct parts (drums + bass + harmony/lead/texture) that interlock and leave space; use call-and-response, accents, or variation.
- Dynamics & development: MANDATORY — include at least one evolving element (filter/param sweep, mute or accent pattern, probability-based variation, swing, or call-and-response) so the loop has motion and is not static.
- Mix: balance gains so no part dominates; use pan/room for space.

Vocabulary: s()/sound(), note()/n()/notes(), stack(), cat(), seq(), struct(), gain(), speed(), pan(), room(), delay(), jux(), off(), rev(), lpf()/cutoff(), distort(), sometimes()/every(), `~` rest, `*`/`/` for speed, `<...>` for cycles, `#`/`?` for probability, chord()/voicing()/scale(). Prefer built-in sounds; do not fetch external samples. Keep snippets 4-14 lines, focused and vivid.

Always verify with the validate_strudel tool before finishing. If it errors, fix the code and re-validate until OK. Your final message must contain ONLY the final working code in a single fenced ```strudel block, nothing else."""

EVAL_SYSTEM = """You are a strict A&R judge for a Strudel training dataset. The goal is music humans genuinely enjoy, not just code that runs. Be discerning: reject trivial/chaotic/sparse/muddy/flat loops, but reward well-crafted loops that have groove, layered arrangement, and real within-loop variation with 8+.

Accept criteria (ALL must hold):
- Groove: clear, coherent rhythm with pocket; not random or plodding.
- Harmony/melody: intentional and pleasing; bass underpins; no accidental dissonance or clashing layers.
- Arrangement: >=3 layered, interlocking parts with space and contrast; not a single bare loop.
- Dynamics (within the loop): real variation — accents, swing/humanization, probability, filter or parameter movement, mutes, or call-and-response. A flat loop with NO variation is static and should score <=6.
- Mix: balanced levels; nothing clipping or drowning others.

Hard reject if ANY of:
- Trivial (1-2 parts, no real groove or idea).
- Chaotic/random (probability/speed abused into noise).
- Too sparse or static (one loop, no layering or variation).
- Muddy or clashing (parts fighting, dissonant without purpose).
- Overly long/unfocused, or relies on unsupported/external sounds.
- Too similar to an existing entry (low originality).

Score anchors: 1-3 broken/trivial; 4-5 weak; 6 competent but flat/static; 7 solid groove + layers + some variation; 8 very good — tight pocket, tasteful layers, clear variation/hook; 9-10 excellent/memorable. A well-crafted varied loop can and should score 8. Set "accept": true ONLY if score >= 8.

Respond with ONLY a JSON object, no prose:
{"accept": true|false, "score": <int>, "genre": "<one specific sub-genre tag, lowercase, e.g. techno|deep-house|ambient|dnb|lofi-hiphop|jazz|afrobeat|synthwave|trap|downtempo|breakbeat|idm|funk|disco|drone|dub|garage|shoegaze|fusion>", "label": "<=15 word vivid description of the music>", "reason": "<one sentence naming the deciding flaw or strength>"}"""


@dataclass
class GenResult:
    code: str | None
    turns: int
    cost: float
    ok: bool


@dataclass
class EvalResult:
    accept: bool
    label: str
    genre: str
    score: int
    reason: str
    raw: str


def _options(api_key: str, *, system_prompt: str, max_turns: int,
             mcp_servers=None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL,
        effort=EFFORT,
        max_turns=max_turns,
        tools=[],
        permission_mode="bypassPermissions",
        mcp_servers=mcp_servers or {},
        env={"ANTHROPIC_API_KEY": api_key},
        system_prompt=system_prompt,
        setting_sources=[],
    )


async def _collect(prompt: str, options: ClaudeAgentOptions) -> tuple[str, int, float, bool]:
    final_text = ""
    turns = 0
    cost = 0.0
    is_error = False
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if getattr(block, "text", None):
                    final_text = block.text
        elif isinstance(msg, ResultMessage):
            turns = msg.num_turns or 0
            cost = msg.total_cost_usd or 0.0
            is_error = msg.is_error
    return final_text, turns, cost, is_error


_CODE_FENCE = re.compile(r"```(?:strudel)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    fences = _CODE_FENCE.findall(text)
    if fences:
        return fences[-1].strip()
    return None


async def generate_strudel(
    accepted_labels: list[str],
    target_genre: str,
    summary: str,
    api_key: str,
    max_turns: int = 8,
) -> GenResult:
    """Run the generator agent. Returns the last OK-validated code (or None)."""
    from validator import make_generator_server

    server, holder = make_generator_server()
    options = _options(api_key, system_prompt=GEN_SYSTEM, max_turns=max_turns,
                       mcp_servers={"strudel": server})

    labels_block = "; ".join(accepted_labels[-30:]) if accepted_labels else "none yet"
    summary_block = f"\n\nRunning guidance from prior generations:\n{summary}" if summary else ""
    prompt = (
        f"Target genre for this generation (REQUIRED): {target_genre}.\n"
        f"Styles already in the dataset (avoid sounding similar): {labels_block}.\n"
        f"Generate a Strudel snippet in the {target_genre} style — make it genuinely "
        f"enjoyable (groove, harmony, >=3 layered parts, real within-loop variation, "
        f"balanced mix). Validate it with validate_strudel, fix any errors, and "
        f"return only the final working code.{summary_block}"
    )

    final_text, turns, cost, is_error = await _collect(prompt, options)

    code = holder["code"] or _extract_code(final_text)
    return GenResult(code=code, turns=turns, cost=cost, ok=code is not None and not is_error)


def _extract_json(text: str) -> dict | None:
    fence = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not fence:
        return None
    try:
        return json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None


async def evaluate_strudel(
    code: str, accepted_labels: list[str], api_key: str, max_turns: int = 1
) -> EvalResult:
    """Strictly judge a validated snippet. Returns accept/label/genre/score/reason."""
    options = _options(api_key, system_prompt=EVAL_SYSTEM, max_turns=max_turns)
    labels_block = "; ".join(accepted_labels[-30:]) if accepted_labels else "none yet"
    prompt = (
        f"Existing entries (reject if too similar): {labels_block}\n\n"
        f"Strudel snippet to judge:\n```strudel\n{code}\n```"
    )
    final_text, _, _, _ = await _collect(prompt, options)

    data = _extract_json(final_text) or {}
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return EvalResult(
        accept=bool(data.get("accept", False)),
        label=str(data.get("label", "")).strip(),
        genre=str(data.get("genre", "")).strip().lower(),
        score=score,
        reason=str(data.get("reason", "")).strip() or final_text[:200],
        raw=final_text,
    )


SUMMARY_SYSTEM = (
    "You compact the running state of a Strudel training-data generator. Given "
    "the list of musical styles/genres already produced and recent rejections, "
    "write a short paragraph (<=6 sentences) of guidance: which styles/techniques "
    "are well-covered and should be avoided, and which new directions to explore "
    "for diversity. Respond with only the guidance paragraph."
)


async def summarize_state(
    accepted_labels: list[str], accepted_genres: list[str],
    recent_rejections: list[str], api_key: str,
) -> str:
    options = _options(api_key, system_prompt=SUMMARY_SYSTEM, max_turns=1)
    prompt = (
        f"Genres already in the dataset: {accepted_genres or 'none'}.\n"
        f"Styles already in the dataset: {accepted_labels or 'none'}.\n"
        f"Recent rejections: {recent_rejections or 'none'}.\n"
        f"Write the guidance paragraph."
    )
    text, _, _, _ = await _collect(prompt, options)
    return text.strip()
