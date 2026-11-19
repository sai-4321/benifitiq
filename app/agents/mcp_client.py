"""MCP client used by the Lookup agent.

It starts the MCP server as a child process (stdio transport), opens ONE session,
calls the tools it needs in order, then closes. In production the server would run
as its own container and we would connect over streamable HTTP instead of stdio.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys

from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import PROJECT_ROOT, get_settings
from app.core.logging_config import get_logger

log = get_logger(__name__)


class ToolCallError(RuntimeError):
    pass


def _server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env["MEMBER_DB_PATH"] = str(get_settings().member_db)  # absolute path for the child process
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server.server"], env=env,
                                 cwd=str(PROJECT_ROOT))


def _parse_result(name: str, result) -> dict:
    if getattr(result, "isError", False):
        text = " ".join(getattr(c, "text", "") for c in result.content)
        raise ToolCallError(f"{name} failed: {text[:300]}")
    structured = getattr(result, "structuredContent", None)
    if structured:
        # FastMCP wraps non-object returns as {"result": ...}; our tools return dicts.
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    for block in result.content:
        if getattr(block, "type", "") == "text":
            return json.loads(block.text)
    raise ToolCallError(f"{name} returned no content")


async def _lookup(member_id: str, service: str | None) -> dict:
    calls: list[dict] = []
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(name: str, args: dict) -> dict:
                data = _parse_result(name, await session.call_tool(name, args))
                calls.append({"tool": name, "args": args})
                return data

            member = await call("get_member_plan", {"member_id": member_id})
            benefits = await call("get_plan_benefits", {"plan_id": member["plan_id"]})
            prior_auth = None
            if service:
                prior_auth = await call("check_prior_auth", {"plan_id": member["plan_id"], "service": service})
    return {"member": member, "benefits": benefits, "prior_auth": prior_auth, "tool_calls": calls}


def _run_sync(coro):
    """Run async code from sync code, even if an event loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@traceable(name="mcp_member_lookup", run_type="tool")
def run_member_lookup(member_id: str, service: str | None = None, timeout: float = 30.0) -> dict:
    """Sync entry point used by the LangGraph lookup node."""
    try:
        return _run_sync(asyncio.wait_for(_lookup(member_id, service), timeout=timeout))
    except ToolCallError:
        raise
    except Exception as exc:  # noqa: BLE001
        # The MCP client runs inside an asyncio TaskGroup, which wraps errors in an
        # ExceptionGroup. Dig out the real error so the agent gets a clean message.
        leaves = _flatten(exc)
        for leaf in leaves:
            if isinstance(leaf, ToolCallError):
                raise leaf from exc
        msg = "; ".join(f"{type(e).__name__}: {e}" for e in leaves)
        raise ToolCallError(f"MCP lookup failed: {msg[:300]}") from exc


def _flatten(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for inner in exc.exceptions:
            out.extend(_flatten(inner))
        return out
    return [exc]
