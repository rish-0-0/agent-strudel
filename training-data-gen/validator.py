"""Strudel snippet validation via the headless node validator.

Exposes both a plain async function (:func:`run_validator`) and an in-process
MCP tool (:func:`make_generator_server`) so the generator agent can call
``validate_strudel`` to self-check its snippets.
"""

import asyncio
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

VALIDATE_JS = Path(__file__).resolve().parent / "validator" / "validate.js"
VALIDATOR_TIMEOUT = 30.0


async def run_validator(code: str) -> tuple[bool, str | None]:
    """Run the node validator on ``code``. Returns (ok, error_or_None)."""
    proc = await asyncio.create_subprocess_exec(
        "node",
        str(VALIDATE_JS),
        cwd=str(VALIDATE_JS.parent),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(code.encode()), timeout=VALIDATOR_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, "validator timed out"
    text = out.decode().strip()
    if text.startswith("OK"):
        return True, None
    return False, text or err.decode().strip() or "unknown error"


def make_generator_server() -> tuple[object, dict]:
    """Build a fresh SDK MCP server plus a holder that captures the last
    snippet that validated OK. Returns (server, holder)."""
    holder: dict = {"code": None}

    @tool(
        "validate_strudel",
        "Compile and run a Strudel snippet headlessly to check it is valid. "
        "Pass the full snippet as `code`. Returns OK on success or the error.",
        {"code": str},
    )
    async def validate_strudel(args):
        code = args["code"]
        ok, err = await run_validator(code)
        if ok:
            holder["code"] = code
        msg = "OK - the snippet compiles and runs." if ok else f"ERROR: {err}"
        return {"content": [{"type": "text", "text": msg}], "is_error": not ok}

    server = create_sdk_mcp_server(name="strudel", tools=[validate_strudel])
    return server, holder
