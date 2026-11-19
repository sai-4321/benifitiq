"""MCP tool server: exposes 3 typed, read-only tools over the Model Context Protocol.

Why MCP instead of letting the agent query the database?
- The agent can only call these 3 tools - no free-form SQL, no other tables.
- Every input has a typed schema (regex pattern / allowed values). Bad input is
  rejected by the server BEFORE any code runs.
- Any MCP client (Claude Desktop, VS Code, another agent) can reuse the same tools.

Run standalone (it talks over stdin/stdout):  python -m app.mcp_server.server
"""
from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.mcp_server import db
from app.schemas import Service

mcp = FastMCP("benefitiq-member-tools", log_level="WARNING")

MemberId = Annotated[str, Field(pattern=r"^M\d{4}$", description="Synthetic member ID, e.g. M1001")]
PlanId = Annotated[str, Field(pattern=r"^PLN-\d{3}$", description="Plan ID, e.g. PLN-002")]


@mcp.tool()
def get_member_plan(member_id: MemberId) -> dict:
    """Return the member's plan ID, plan name, plan type, coverage start and status (active/inactive)."""
    return db.get_member_plan(member_id)


@mcp.tool()
def get_plan_benefits(plan_id: PlanId) -> dict:
    """Return cost-sharing for a plan: deductible, copays (PCP, specialist, ER, urgent care),
    out-of-pocket maximum, coinsurance percent and whether referrals are required."""
    return db.get_plan_benefits(plan_id)


@mcp.tool()
def check_prior_auth(plan_id: PlanId, service: Service) -> dict:
    """Return whether a service needs prior authorization on a plan, with notes."""
    return db.check_prior_auth(plan_id, service)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
